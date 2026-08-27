# DGX Spark 部署 Qwen3.8-Flash-Next 实战手册(2026-08-28 更新:已执行完成)

**[中文](deploy_playbook.zh.md) | [English](deploy_playbook.md)**

> 目标机器:NVIDIA DGX Spark(GB10),128 GB 统一内存
> 状态:**已部署完成并全面提速**——生产配方 = Q3_K_XL + NVMe-PLE + MTP+ngram-map-k4v,262K,78.9 t/s 单流 / 77.5 聚合(2 并发),70GB 内存
> 早期版本(2026-08-26)是"等目标版本出现"的预案,现已全部落地,本文档转为实战记录

## 0. 验收公式(实测核对)

| 项 | 实测值 |
|---|---|
| 磁盘占用(三档模型)| Q3 90GB / IQ3 82GB / Q4 111GB(共 ~283GB,空闲 3.0TB) |
| 运行时峰值 | Q3 配方 262K = **70GB** ≤ 115GB ✅(旧配置 102GB) |
| 无加载反量化膨胀 | ✅ NVMe-PLE:`-lm mmap -ot per_layer_token_embd=CPU`,PLE 不驻留 |
| 质量档位 | Q3 90.4% / IQ3 87.6% / Q4 93.5%(unsloth Divergence-300 数据) |

## 1. 三条运行路径(按优先级,全部本机实测)

### 路径 A:Q3_K_XL 生产配方(首选,262K,78.9 t/s,70GB)🏆

```bash
# 引擎:~/llama-mtp-verified/build(bea3b12d + qwen4exp-mtp-draft-head.patch)
LLAMA_ATTN_ROT_DISABLE=1 ~/llama-mtp-verified/build/bin/llama-server \
  -m ~/models/Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf \
  -lm mmap -ot per_layer_token_embd=CPU \
  -md ~/models/mtp-draft/dzannotti/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf -ngld 999 \
  --spec-type draft-mtp,ngram-map-k4v --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0 \
  --ctx-size 262144 --parallel 2 --host 0.0.0.0 --port 8889 \
  --temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 --presence-penalty 1.5 --jinja
# 就绪后预热 PLE(一次 ~28s):
# GGUF_PY=~/llama.cpp/gguf-py ~/ai-env/bin/python3 ~/smalltask/hf_monitor/warm_table.py <模型分片1>
```

实测:代码复制 78.9 t/s(262K)/ 82.7(8K);散文 21.5-25.8;内存 70GB(余 51GB)。

### 路径 B:速度优先(IQ3_XXS,8K 下 83.0 t/s)

- 引擎/参数同上,模型换 `UD-IQ3_XXS-00001-of-00003.gguf`;或用内嵌版 `UD-IQ3_XXS-MTP/`(免 `-md`,merge-mtp-shard.py 生成)
- 质量 87.6%;窗口验证到 16K

### 路径 C:质量优先(Q4_K_XL,8K 下 70.1 t/s,93.5%)

- ⚠️ **仅 8K 短窗口**(事故 5:262K 加载即整机冻结);大窗口请用路径 A
- 引擎/参数同上,模型换 `UD-Q4_K_XL-00001-of-00004.gguf`

### 简易方案(零依赖,ngram-map-k4v)

```bash
~/llama.cpp/build/bin/llama-server -m <任意分片1> -ngl 999 -t 20 \
  --spec-type ngram-map-k4v --jinja --ctx-size 262144 --port 8889
```

## 2. 构建步骤(已执行,记录备查)

```bash
# MTP 路线(验证树 bea3b12d + 补丁)—— ⚠️ 035e22731 树与补丁不兼容(段错误)
git clone ~/llama.cpp ~/llama-mtp-verified
cd ~/llama-mtp-verified && git checkout bea3b12da
git apply ~/smalltask/hf_monitor/dzannotti-mtp/qwen4exp-mtp-draft-head.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 20

# ngram 路线(035e22731 + canreuse 补丁)
cd ~/llama.cpp && git apply ~/smalltask/hf_monitor/canreuse-qwen4exp.patch
cmake --build build -j 20
```

> 注意:canreuse 补丁依赖 035e22731 的 `llama_memory_hybrid_idx_context`,**不可用于 bea3b12d 树**(编译失败,已回滚)。

## 3. MTP 头注入(免重下模型)

```bash
# 内嵌头(给任意分片 GGUF 附加 MTP 头,硬链接省盘)
GGUF_PY=~/llama.cpp/gguf-py ~/ai-env/bin/python3 \
  ~/projects/qwen38-flash-next-dgx-spark/patches/merge-mtp-shard.py \
  <目标模型分片1> <MTP头gguf> <输出目录>
# 然后 --spec-type draft-mtp 且不带 -md 运行;性能与 -md 差 ~2%
```

## 4. 内存安全铁律(血泪教训,务必遵守)

1. 任何时候只允许**一份模型**驻留(可加 2.44GB MTP 头)
2. 测试前 `free -h`,余量 **≥45GB** 才启动
3. 新组合首次加载前:`sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`
4. **窗口逐级验证:8K→32K→64K→128K→262K,禁止跳级**(事故 5)
5. 看门狗:系统级 `system_watchdog.sh 5 5`(crontab @reboot)+ 测试级 `watchdog.sh <PID> 15 5`
6. 测试实例:独立端口 + setsid 脱离;`pkill -f` 用 `[x]` 防自杀
7. 重启后模型零损坏(已验证 3 次),一键恢复

## 5. 质量红线(沿用)

- 1-bit(IQ1 系):排除(工具调用不可用 + 质量崩坏)
- 2-bit:应急可接受
- **3-bit/4-bit:推荐区间**(Q3 90.4% / IQ3 87.6% / Q4 93.5% 全部实测可跑)
- 5-bit/6-bit/Q8_0:即使 PLE-offload 也超预算(≥117GB),排除
- NVFP4 全量版:运行时未公开,暂不可用(跟踪中)

## 6. 历史版本说明

- 2026-08-26 初版为预案(当时 Q3 未下载、PR 未合并、投机未解锁)
- 2026-08-27 落地:下载攻坚(12 线程并行)、构建、262K 服务、死机教训、投机突破
- 2026-08-28 终版:生产配方 + NVMe-PLE + 组合投机 + 窗口铁律
