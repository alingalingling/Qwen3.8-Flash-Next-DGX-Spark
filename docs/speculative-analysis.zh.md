# 投机解码全景实测分析(Qwen3.8-Flash-Next on DGX Spark)

**[中文](speculative-analysis.zh.md) | [English](speculative-analysis.md)**


> 平台:NVIDIA DGX Spark(GB10,128 GB 统一内存,273 GB/s)
> 模型:unsloth UD-Q3_K_XL / UD-IQ3_XXS / UD-Q4_K_XL(本机实测)
> 引擎:llama.cpp qwen4exp 分支(035e22731 + bea3b12d 两构建)
> 日期:2026-08-27~28 | 所有结论均来自本机实测,非纸面推测

> **阅读导航**:第 1-4 节是早期基线实测(ngram-simple 等,当时结论"24 t/s 是天花板");
> 第 5 节起是演进与解锁——**ngram-mod 与 MTP 头出现后,结论全面改写,最终见第 7/9 节。**

---

## 1. 为什么关心投机解码

Qwen3.8-Flash-Next 是 180B 混合 MoE(每次仅激活 ~6B 参数),解码受权重读取带宽与
固定延迟限制,本机实测裸跑 **22-24 t/s**;投机解锁后代码类任务 **58-83 t/s**。

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

### 3.2 draft-dflash / draft-dspark — 静默未生效

```bash
llama-server ... --spec-type draft-dflash # 或 draft-dspark
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

### 3.3 draft-mtp — GGUF 无 MTP 头

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

### 3.4 外部小草稿(draft-simple)— 0.2B 微型模型完整实验

**候选**:[inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B](https://huggingface.co/inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B)
(0.33 GB,0.16B 参数,保留 GDN/QSA/PLE/MoE 全部架构组件)

**实验流程(全部实测)**:

| 步骤 | 结果 |
|---|---|
| ① 下载 | 332 MB,秒级完成 |
| ② qwen4exp 转换器转 GGUF | 成功(113 张量,321 MB)——证明转换器对新架构小模型同样可用 |
| ③ tokenizer 一致性验证 | **完全一致**(官方 vs 微型,同文本产出完全相同 token ID 序列) |
| ④ 挂载 --model-draft 实测 | 24.1 t/s,无接受率统计,无提速 |
| ⑤ 单独生成测试 | **输出空白**——模型权重随机初始化/未训练 |

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
训练过的草稿: 接受率 ≈ 30-70%(同家族小模型/蒸馏模型)
```

未训练草稿猜 k 个 token,主模型几乎总是**第一个就拒绝** → 投机完全退化,
草稿生成 + 验证的开销纯浪费 → 速度与基线持平甚至略降(实测 24.1 vs 24.2)。

**类比**:投机解码 = 让"快嘴助手"先念一句,专家快速核对。助手必须**真的会说话**
(受过训练)才可能猜到专家的话;未训练模型是"哑巴助手"——结构齐全(有嘴),
但念出来的全是乱码,专家每次都直接摇头。

---

## 5. 因果链演进:从"24 t/s 天花板"到全面解锁(历史回顾)

**2026-08-27 白天的旧因果链**(当时投机两端全断,结论"24 t/s 是天花板"已被当晚实测推翻):

```
投机解码 = 草稿来源 + 引擎执行路径

草稿来源(当时三条路全断):
 ├─ MTP 自投机 → GGUF 无 MTP 头(unsloth 量化时移除,实测确认) → 已解:dzannotti 独立头补回
 ├─ DFlash/DSpark → GGUF 无 DND 草稿结构张量(参数被接受但静默退化) → 仍无
 └─ 外部小草稿 → 无训练过的同 tokenizer 小模型(0.2B 未训练) → 仍无(且 MoE 下无意义)
 (ngram-simple 仅重复文本有效,通用场景≈0) → 已解:ngram-mod 长跨度

引擎执行路径(当时未完成):
 └─ llama.cpp qwen4exp 的 MTP = WIP(PR #27742 无 MTP 提交) → 已解:bea3b12d 树 + 补丁
```

**2026-08-27 夜的解锁结果(全部本机实测)**:

| 方案 | 需要什么 | 实测收益 | 状态 |
|---|---|---|---|
| **ngram-mod** | 内置,零成本 | 代码复制 58.7 t/s(+135%) | 已解锁 |
| **ngram-map-k / map-k4v** | 内置,零成本 | 代码复制 **82.4 / 84.2 t/s**(+230%,接受率 0.849) | 已解锁(2026-08-28,比 mod 再快 40%) |
| **MTP(dzannotti 头)** | 2.44GB 头 + bea3b12d 验证树补丁 | 数数 57.3 / 代码 54.6 | 已解锁 |
| **MTP + ngram-mod 叠加** | 同上组合 | 代码复制 **83.0 t/s**(+232%) | 已解锁 |
| Q4_K_XL + PLE-offload + 组合投机 | 下载 111GB + PLE 卸载参数 | 代码 70.1 t/s(质量 93.5%) | 已解锁(仅 8K) |
| SGLang + DFlash2 | 全量 NVFP4 + 成熟运行时 | — | 运行时未公开,观察中 |

---

## 6. 瓶颈分析:裸跑 22-24 t/s 的三大延迟源(投机解锁后仍存在)

理论:每 token 读取 6B 激活专家 × ~0.45 B ≈ 2.7 GB ÷ 273 GB/s ≈ **~100 t/s**

裸跑 24 t/s 的差距来自三个固定延迟源;**投机解锁后,代码类任务通过并行化 verify 步把有效吞吐拉到 58-83 t/s,但单 token 延迟源不变**:

| 延迟源 | 机制 | 状态 |
|---|---|---|
| PLE n-gram 随机查表 | 51B 表,每次查表都是缓存未命中的随机内存访问 | NVMe-PLE 卸载 + 预热后影响大减;架构固有部分仍存 |
| GDN 循环状态 | 逐 token 串行更新,无法并行 | 架构固有 |
| MoE 小矩阵 | 专家中间维仅 640,每个 GEMM 很小,GPU 启动开销占比高 | canreuse 补丁让 CUDA graph 复用(实测 graphs reused 0→52+,+2.8%) |

> 注:0xBakeer 的"小草稿模型无效"分析也解释了为什么投机必须走长跨度(ngram-mod 均长 37-60 token)而不是小步草稿:top-10/512 MoE 下 k 个草稿 token 激活 k×10 个不同专家,权重流量随 k 线性增长,无摊薄空间。

---

## 7. 解锁路线图:什么时候能提速(2026-08-27 夜:已部分解锁 )

| 触发条件 | 需要什么 | 预期收益 | 当前状态 |
|---|---|---|---|
| **ngram-map-k4v 投机** | 无需任何新东西(llama.cpp 内置) | **复制型任务 3.4x**(84.2 t/s) | **已解锁**——零内存增量,输出逐 token 验证不变(2026-08-28 起替代 ngram-mod) |
| **MTP 自投机** | dzannotti 独立 MTP 头 + bea3b12d 验证树补丁 | **规律性输出 2.2x**(57.3 t/s) | **已解锁**——035e22731 树段错误,须用验证树 |
| **MTP + ngram-map-k4v 叠加** | 同上两者组合 | **复制型任务 4.3x**(108.4 t/s) | **已解锁**——`--spec-type draft-mtp,ngram-map-k4v` |
| **SGLang/vLLM + NVFP4** | 单机 NVFP4 方案(Felliks/MaxLaurence PLE 落盘 / starkweatherdigital 109GB 全量化) | 单流 17.9-42 / 四流聚合 93-132 | 单机跑通但内存 109-120GB 临界;全量化权重上传中 |

**当前推荐生产配置**(代码/agentic 场景):
```bash
# 方案 A(推荐,零内存增量):ngram-map-k4v(2026-08-28 起替代 ngram-mod)
llama-server -m model.gguf -ngl 999 -t 20 --spec-type ngram-map-k4v --jinja

# 方案 B(最强,代码复制 108 t/s):MTP + ngram-map-k4v 叠加(需验证树 + MTP 头)
LLAMA_ATTN_ROT_DISABLE=1 llama-server -m model.gguf \
 -md MTP-Q4_K_M.gguf -ngld 999 \
 --spec-type draft-mtp,ngram-map-k4v --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
 -ngl 999 -fa on -ctk q8_0 -ctv q8_0
```

> 速度形态说明:ngram/MTP 收益完全取决于"输出中有多少来自上下文/规律"——
> 工具驱动的文件编辑、代码补全、列表/参数输出受益巨大;自由散文几乎无增益(仍 ~26 t/s)。
> 测速方法论陷阱(0xBakeer 实测):重复相同 prompt 会把结果虚高 2.8x,必须变换 prompt 再测。

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
llama-cli -m draft.gguf -n 40 # 输出空白/乱码 → 未训练,淘汰
```

---

## 9. 结论(2026-08-27 夜重大更新)

1. **"24 t/s 天花板"已被打破**:ngram-mod(代码复制 58.7 t/s)、MTP(数数 57.3 t/s)、
 **MTP+ngram-mod 叠加(代码复制 83.0 t/s,3.3x)**——全部本机实测,零/极低内存增量
2. **收益形态是"输出可预测性",不是"是不是代码"**:工具驱动的编辑/补全/结构化输出受益巨大,
 自由散文基本无增益(~26 t/s);测速必须变换 prompt(重复 prompt 会虚高 2.8x)
3. **MTP 头已可注入**:dzannotti 标准量化头(2.44GB)+ bea3b12d 验证树补丁;merge-mtp-shard.py
 可内嵌进任意现有 GGUF(实测与 -md 差 ~2%);035e22731 树与补丁不兼容(段错误)
4. **cafe-llama.cpp fork 永久放弃**:两次加载 MTP 草稿爆内存死机
5. 全量 NVFP4(101.7GB,含 MTP 头)已出现(provsalt),SGLang 路线观察中
6. 本文档的所有结论均来自本机实测,欢迎复现与反馈

---

## 参考

- [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) — qwen4exp 支持
- [llama.cpp PR #27739](https://github.com/ggml-org/llama.cpp/pull/27739) — 前身,含 MTP+PLE 卸载实现
- [inference-optimization 0.2B 微型模型](https://huggingface.co/inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B) — 已实测不可作草稿
- 配套工具:scripts/probe_mtp.py(GGUF MTP 头探测)
