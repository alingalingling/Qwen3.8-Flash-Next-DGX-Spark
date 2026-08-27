# MTP 头追踪与注入方案(Qwen3.8-Flash-Next)

**[中文](mtp-tracker.zh.md) | [English](mtp-tracker.md)**

> 背景:投机解码(draft-mtp)可大幅提升解码速度。
> 前提:GGUF 必须携带 MTP 头张量。unsloth 的 Flash-Next 量化版**默认移除** MTP 头(180B → 176.94B 参数,probe_mtp.py 已验证)。

## 当前状态(2026-08-27 夜更新)

| 项 | 状态 |
|---|---|
| llama.cpp qwen4exp PR(#27742)| open、可合并;**54 commits,仍无 MTP 提交**;作者 WIP |
| 前身 PR #27739 | closed,但**已含 MTP + PLE 卸载实现**,社区已提醒作者参考 |
| **dzannotti MTP 头**([仓库](https://huggingface.co/dzannotti/Qwen3.8-Flash-Next-MTP-GGUF))| **重大突破**:独立 `MTP-Q4_K_M.gguf`(2.44GB,标准量化,任意后端)+ `qwen4exp-mtp-draft-head.patch`(459 行,PR #27739 MTP 图调和版 + converter `--mtp` 导出) |
| **本机构建** | 补丁在本机 qwen4exp 分支(基线 bea3b12d 与补丁目标树一致)**干净应用,重编译成功**——llama-server 现支持 `--spec-type draft-mtp`(2026-08-27 19:16 构建) |
| unsloth GGUF | 全部 7 档无 MTP 头;dzannotti 独立头可作为 `-md` 草稿 |
| 独立草稿模型 | 不存在训练过的 qwen4exp 小模型;**MTP 草稿头(4B,官方 BF16 导出)即正确草稿** |

## 作者实测数据(AMD Strix Halo 128GB 统一内存——与 DGX Spark 同级别)

| 后端,目标模型 | 裸跑 | + MTP | 接受率(code / prose) |
|---|---:|---:|---|
| ROCm, UD-Q4_K_XL | 20.3 t/s | **35.8** code / 22.6 prose | 0.90 / 0.74 |
| ROCm, UD-IQ4_XS | 18.0 / 18.6 | 32.8 / 22.1 | 0.84 / 0.68 |
| Vulkan (RADV, Laurent fork), UD-IQ4_XS | 24.2 / 24.3 | **37.2** code / **30.3** prose | 0.88 / 0.82 |

推荐参数:`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75`
(p-min 0.75 比加深深度更重要:草稿头自带 MoE,每个草稿 token 都是真实前向)。
**必须设置 `LLAMA_ATTN_ROT_DISABLE=1`**(上游 quantized-KV 旋转 #21038 不被 qwen4exp attention 路径支持,不设则加载即 abort)。

## 运行方式(独立草稿头)

```bash
LLAMA_ATTN_ROT_DISABLE=1 llama-server \
 -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
 -md Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf -ngld 999 \
 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -c 262144
```

## 注入方案(免重下 82/90GB 模型——merge-mtp-shard.py)

`patches/merge-mtp-shard.py`(来自 dzannotti 仓库,已归档到本项目)可把 MTP 头作为
**任意现有分片 qwen4exp GGUF** 的附加分片:

1. 保留目标模型分片 2..N,重命名为 `-0000N-of-0000(M+1).gguf`
2. 放入重写后的分片 1(元数据:`block_count 49`、`nextn_predict_layers 1`、`compress_ratios` 扩展、`split.count M+1`)和新末分片(29 个头张量,与 Q4_K_M 头同字节)
3. 用 `--spec-type draft-mtp` 且**不带 `-md`** 运行;草稿上下文基于目标模型创建并共享内存(比独立头省 ~0.5GB)

实测:与 `-md` 形式仅差几个百分点(34.5 vs 36.7 t/s,code 场景)——便利性是重点,非速度。

## 本机实测结果(2026-08-27 夜,IQ3_XXS + dzannotti Q4_K_M 头,验证树 bea3b12d)

| 方案 | A: 数数 200 | B: 散文 150 | C: 代码复制修改 | 内存增量 |
|---|---:|---:|---:|---:|
| IQ3 基线 | 26.2 t/s | 25.4 t/s | 25.0 t/s | 0 |
| + MTP(-md 独立头)| **57.3** | 25.2 | **54.6** | ~3 GB |
| + MTP + ngram-mod | **58.1** | **29.6** | **83.0** | ~3 GB |
| + MTP(内嵌头,merge 方案)| 56.3 | — | 53.6 | ~2.5 GB |
| **Q4_K_XL + PLE-offload + MTP+ngram-mod** | 39.1 | 19.3 | **70.1** | ~4 GB(总内存 86GB) |

> Q4 行:质量 93.5%(全档位最高可跑档),NVMe-PLE 把内存压到 82GB 基线;** 仅 8K 窗口验证过,262K 加载会冻结整机(事故 5),生产大窗口用 Q3**;
> 代码复制 70.1 = 自身无投机基线(22.9)的 **3.1 倍**。速度优先选 IQ3(83.0),质量优先选 Q4(70.1)。

- MTP 头草稿统计:接受率 0.62-1.00,平均草稿长度 2.2-4.0(MTP 本身);叠加 ngram-mod 后代码场景平均长度升至 10.8
- **MTP 与 ngram-mod 互补**:MTP 擅长预测上下文之外的规律性输出(数数),ngram-mod 擅长抄上下文(代码复制);
 两者叠加在代码复制上达到 **83.0 t/s = 基线的 3.3 倍**
- 组合用法:`--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-p-min 0.75`
- 内嵌头与 `-md` 性能差 ~2%,便利性优先(省 0.5GB + 免第二个模型句柄)

### 树版本兼容性(重要教训)

- **补丁在 035e22731 树上段错误**(MTP 头单独加载/`-md`/内嵌全部 segfault)——该树含 indexer-cache 重构,与补丁不兼容
- **补丁只在 bea3b12d 树验证过**:`git checkout bea3b12da && git apply qwen4exp-mtp-draft-head.patch` 后构建即正常
- 本机两个构建并存:`~/llama.cpp/build`(035e22731,ngram-mod 路线)与 `~/llama-mtp-verified/build`(bea3b12d,MTP 路线)

## 本机行动计划(2026-08-27 夜:主体已完成 )

1. IQ3_XXS(82GB)下载完成 → 测速链完成(IQ3 基线 / ngram-mod / MTP 官方头 / Q3_K_XL 基线)
2. MTP-Q4_K_M 头(2.44GB)下载完成,验证树重建成功,MTP 全形态实测完成(见上表)
3. merge-mtp-shard.py 已给 IQ3_XXS 内嵌 MTP 头(UD-IQ3_XXS-MTP/,4 分片),实测可用
4. Q4_K_XL(111GB)已下载 + 0xBakeer NVMe-PLE 方案实测完成(82GB、70.1 t/s, 仅 8K 窗口)
5. cafe-llama.cpp fork 路线永久放弃(两次爆内存死机,增量远超估算)

## 监控信号(仓库内 monitor.py 已跟踪)

- llama.cpp PR #27742 出现 MTP 相关 commit / 合并
- unsloth 仓库出现带 MTP 的文件名(如 *-MTP-* 或 mtp.gguf)
- Baekpica 混合量化版通过验证门(其 MQ 版本含 MTP 专家 Q8_0)
- dzannotti 仓库更新(头重量化、更佳补丁)

## 参考

- [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)
- [llama.cpp PR #27739](https://github.com/ggml-org/llama.cpp/pull/27739)(含 MTP 实现)
- [dzannotti/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/dzannotti/Qwen3.8-Flash-Next-MTP-GGUF)(独立 MTP 头 + 补丁 + 合并脚本)
