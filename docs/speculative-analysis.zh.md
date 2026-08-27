# 投机解码全景实测分析(Qwen3.8-Flash-Next on DGX Spark)

**[中文](speculative-analysis.zh.md) | [English](speculative-analysis.md)**


> 平台:NVIDIA DGX Spark(GB10,128 GB 统一内存,273 GB/s)
> 模型:unsloth UD-Q3_K_XL(90 GB,3-bit,176.94B 参数)
> 引擎:llama.cpp qwen4exp 分支(commit 035e22731,CUDA)
> 日期:2026-08-27 | 所有结论均来自本机实测,非纸面推测

---

## 1. 为什么关心投机解码

Qwen3.8-Flash-Next 是 180B 混合 MoE(每次仅激活 ~6B 参数),解码受权重读取带宽与
固定延迟限制,本机实测 **22-24 t/s**。

---

## 2. llama.cpp 投机体系全景(--spec-type 支持 11 种)

本构建(0.3.0-dev,qwen4exp 分支)支持:

```
--spec-type none, draft-simple, draft-eagle3, draft-mtp,
              draft-dflash, draft-dspark,
              ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache
```

| 类别 | 方案 | 草稿来源 | 是否需要额外模型/张量 |
|---|---|---|---|
| 自投机 | draft-mtp | 模型自带 MTP 头 | 需要 GGUF 含 MTP 张量 |
| 引擎草稿 | draft-dflash / draft-dspark | DeltaNet 结构 | 需要模型携带 DND/DFlash 草稿张量 |
| 外部草稿 | draft-simple / draft-eagle3 | 独立小草稿模型 | 需要训练过的同 tokenizer 小模型 |
| 无模型 | ngram-* | 上下文重复片段 | 无需任何额外资源 |

---

## 3. 实测:五种方案逐一验证

### 3.0 基线

| 配置 | 解码速度 | 说明 |
|---|---|---|
| GPU 全卸载(-ngl 999)| **24.2 t/s** | llama-bench tg64,生产配置 |
| 纯 CPU(-ngl 0)| 1.9 t/s | GPU 卸载 = 12.7 倍 |

### 3.1 ngram-simple — 仅重复文本场景

```bash
llama-server ... --spec-type ngram-simple --spec-ngram-simple-size-n 8
```

| 实测项 | 值 |
|---|---|
| 解码速度 | 24.8 t/s(基线 24.2,+2%)|
| 草稿接受率 | 25.6%(11 accepted / 43 generated)|
| 草稿长度 | 平均 12 token |

**结论**:机制确实生效(有接受率统计),但收益微乎其微。原因:n-gram 从上下文
匹配重复片段,只对**代码/日志/格式化输出**等高度重复文本有效;通用对话几乎
没有可匹配的重复模式。且接受率 25.6% 带来的理论 1.34x 被验证路径开销抵消。

**适用**:若工作负载以重复文本为主(如日志分析、代码生成模板),可开启;
通用场景**不值得**。

### 3.2 draft-dflash / draft-dspark — ❌ 静默未生效

```bash
llama-server ... --spec-type draft-dflash   # 或 draft-dspark
```

| 实测项 | 值 |
|---|---|
| 解码速度 | 24.0 / 24.1 t/s(与基线无差异)|
| 投机日志 | **零输出**(无接受率、无草稿活动)|

**结论**:参数被接受,但**静默退化**为无投机。原因:这两种方案需要模型携带
特定的 DND/DFlash 草稿结构张量(类似 SGLang DFlash2 使用的 DND 权重)。
unsloth 的 GGUF 未包含这些张量,llama.cpp 找不到草稿结构就直接跳过。

**关键认知**:`draft-dspark` 的名字虽为 DGX Spark 设计,但同样依赖模型结构——
**不是硬件不够,是模型文件里没有配套张量**。

### 3.3 draft-mtp — ❌ GGUF 无 MTP 头

```bash
llama-server ... --spec-type draft-mtp --spec-draft-n-max 2
```

| 实测项 | 值 |
|---|---|
| GGUF MTP 张量扫描 | **0 条**(probe_mtp.py 实测,3 分片共 1224 张量)|
| 参数核对 | 176.94B = 180B − 4B(MTP 头被移除)|
| unsloth 仓库独立 MTP 模块 | 不存在(全 7 档均无)|

**结论**:draft-mtp 需要模型自带 MTP 头(4B 参数,31 个张量)。unsloth 在量化时
移除了 MTP 头(省 ~2 GB 空间),因此**原样 GGUF 自投机无从谈起**。对比:27B 的 GGUF
**自带** MTP 头,这就是一行 flag 就能提速的原因。

> **更新(2026-08-27 夜)**:「原样」的限制已被打破。[dzannotti/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/dzannotti/Qwen3.8-Flash-Next-MTP-GGUF)
> 发布了**独立 MTP 草稿头**(`MTP-Q4_K_M.gguf`,2.44 GB,标准量化),可作为 `-md` 挂到任意
> Flash-Next GGUF 旁;附补丁 `qwen4exp-mtp-draft-head.patch`(给 qwen4exp 分支加 MTP 图)—
> 本机已干净应用并重编译成功(19:16,`--spec-type draft-mtp` 已支持)。作者在同规格
> 128GB 机器实测:**UD-Q4_K_XL 20.3 → 35.8 t/s(code),接受率 0.90**。
> 运行命令与免重下的注入脚本(`merge-mtp-shard.py`)见 [mtp-tracker.md](mtp-tracker.md)。

### 3.4 外部小草稿(draft-simple)— ❌ 0.2B 微型模型完整实验

**候选**:[inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B](https://huggingface.co/inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B)
(0.33 GB,0.16B 参数,保留 GDN/QSA/PLE/MoE 全部架构组件)

**实验流程(全部实测)**:

| 步骤 | 结果 |
|---|---|
| ① 下载 | 332 MB,秒级完成 |
| ② qwen4exp 转换器转 GGUF | ✅ 成功(113 张量,321 MB)——证明转换器对新架构小模型同样可用 |
| ③ tokenizer 一致性验证 | ✅ **完全一致**(官方 vs 微型,同文本产出完全相同 token ID 序列) |
| ④ 挂载 --model-draft 实测 | ❌ 24.1 t/s,无接受率统计,无提速 |
| ⑤ 单独生成测试 | ❌ **输出空白**——模型权重随机初始化/未训练 |

**结论**:前三步全部通过(格式、tokenizer 都满足草稿条件),死在最后一步——
**模型没训练过,预测全是随机,接受率 ≈ 0**。

**判断草稿模型可用性的快速方法**(沉淀为方法论):
1. **先单独跑草稿模型**,看输出是否连贯——输出空白/乱码 = 未训练,直接淘汰
2. **再比对 tokenizer**(同文本 → 同 token ID 序列)——不一致则 token 错位,不可用
3. 都通过才值得挂载实测

---

## 4. 为什么"未训练"就不能当草稿(接受率数学)

主模型每个位置的真实输出是 248,044 个词上的分布;草稿猜的 token 恰好落在
主模型"最想要"位置的概率:

```
随机草稿(未训练): 接受率 ≈ 1/248,044 ≈ 0.0004%
训练过的草稿:     接受率 ≈ 30-70%(同家族小模型/蒸馏模型)
```

未训练草稿猜 k 个 token,主模型几乎总是**第一个就拒绝** → 投机完全退化,
草稿生成 + 验证的开销纯浪费 → 速度与基线持平甚至略降(实测 24.1 vs 24.2)。

**类比**:投机解码 = 让"快嘴助手"先念一句,专家快速核对。助手必须**真的会说话**
(受过训练)才可能猜到专家的话;未训练模型是"哑巴助手"——结构齐全(有嘴),
但念出来的全是乱码,专家每次都直接摇头。

---

## 5. 因果链总览:为什么当前 24 t/s 是天花板

```
投机解码 = 草稿来源 + 引擎执行路径

草稿来源(三条路全断):
  ├─ MTP 自投机    → GGUF 无 MTP 头(unsloth 量化时移除,实测确认)      ❌
  ├─ DFlash/DSpark → GGUF 无 DND 草稿结构张量(参数被接受但静默退化)   ❌
  └─ 外部小草稿    → 无训练过的同 tokenizer 小模型(0.2B 未训练)       ❌
  (ngram 不挑模型,但仅重复文本有效,通用场景≈0)                       ⚠️

引擎执行路径(未完成):
  └─ llama.cpp qwen4exp 的 MTP 加载/投机 = WIP(PR #27742 无 MTP 提交) ❌

结论:这是时间问题,不是硬件问题
```

---

## 6. 瓶颈分析:24 t/s 离带宽极限(~100 t/s)差在哪

理论:每 token 读取 6B 激活专家 × ~0.45 B ≈ 2.7 GB ÷ 273 GB/s ≈ **~100 t/s**

实测 24 t/s,差距来自三个固定延迟源:

| 延迟源 | 机制 | 可优化性 |
|---|---|---|
| PLE n-gram 随机查表 | 51B 表,每次查表都是缓存未命中的随机内存访问(~100-200ns/次) | 架构固有,难优化 |
| GDN 循环状态 | 逐 token 串行更新,无法并行 | 架构固有 |
| MoE 小矩阵 | 专家中间维仅 640,每个 GEMM 很小,GPU 启动开销占比高 | 依赖引擎内核融合优化 |

---

## 7. 解锁路线图:什么时候能提速

| 触发条件 | 需要什么 | 预期收益 | 当前状态 |
|---|---|---|---|
| **MTP 自投机** | ① llama.cpp PR #27742 完成 MTP 支持 ② 带 MTP 头的 GGUF(unsloth 发布或自行注入) | **1.5-2.5x**(30-50 t/s) | ① WIP(前身 PR #27739 有现成实现可参考)② 无 |
| **SGLang + DFlash2** | 全量 NVFP4 检查点(≤101 GB)+ SGLang qwen4_exp 支持成熟 | **2-4x**(27B 先例 55 t/s) | 全量 NVFP4 未出现 |
| ngram(重复文本场景) | 无 | 特定场景 1.2-1.5x | **现在可用**(见 3.1)|

**MTP 注入方案**(llama.cpp 支持落地后,仓库将提供 inject_mtp.py):
1. 下载官方 BF16(360 GB)
2. convert_hf_to_gguf.py 转 Q8_0 中间件
3. 抽取 31 个 MTP 张量,Q4_0 量化(~2 GB)
4. gguf-py 注入 unsloth 的 UD-Q3_K_XL(更新张量列表 + 追加数据)
5. probe_mtp.py 验证 + `--spec-type draft-mtp --spec-draft-n-max 2` 启动

---

## 8. 测试方法论(可复现)

```
# 解码基线(llama-bench 需在服务停止时运行,避免同时加载两份模型)
llama-bench -m model.gguf -ngl 999 -t 20 -p 0 -n 64 -r 2

# 投机测试:独立端口 + 小上下文,避免影响生产
llama-server -m model.gguf --port 8890 --ctx-size 16384 \
  --spec-type <方案> --jinja

# 读取接受率:服务日志中 "draft acceptance = X (accepted / generated)"
# 读取速度:API 响应的 timings.predicted_per_token_ms

# 判断草稿模型可用性:先单独跑看输出,再比对 tokenizer
llama-cli -m draft.gguf -n 40        # 输出空白/乱码 → 未训练,淘汰
```

---

## 9. 结论

1. **当前组合(UD-Q3_K_XL + llama.cpp)的 24 t/s 是真实天花板**——五条投机路径
   全部实测,全部因"草稿来源缺失"或"引擎支持未完成"而不可用
2. **不是硬件问题**:GB10 的算力/带宽都够,缺的是模型文件里的投机配套
   (MTP 头 / DFlash 张量)与引擎的 MTP 执行路径
3. **最大蛋糕是 MTP 自投机**(1.5-2.5x),两个前置条件都在推进中;
   仓库的 monitor.py 与 mtp-tracker.md 持续跟踪
4. 本文档的所有结论均来自本机实测,欢迎复现与反馈

---

## 参考

- [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) — qwen4exp 支持
- [llama.cpp PR #27739](https://github.com/ggml-org/llama.cpp/pull/27739) — 前身,含 MTP+PLE 卸载实现
- [inference-optimization 0.2B 微型模型](https://huggingface.co/inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B) — 已实测不可作草稿
- 配套工具:scripts/probe_mtp.py(GGUF MTP 头探测)
