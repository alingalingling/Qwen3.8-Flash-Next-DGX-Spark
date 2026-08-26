# 实测基准与对比数据集(NVIDIA DGX Spark,2026-08-27)

**[中文](benchmarks.zh.md) | [English](benchmarks.md)**


> 硬件:GB10 / 128 GB 统一内存 / 273 GB/s / 20 核 ARM(10×Cortex-X925 + 10×A725)
> 引擎:llama.cpp qwen4exp 分支(commit 035e22731,CUDA 后端,GCC 13.3 aarch64)
> 服务:llama-server,OpenAI 兼容 API,端口 8889
> 说明:所有数据为本机实测;标注"官方"的为 unsloth 公布数据,其余为实测

---

## 1. 量化档位全景对比(官方数据 + 本机判定)

| 档位 | 大小 | 与 BF16 一致率 same_top_pct↑ | 质量偏差 mean_kld↓ | 官方总内存需求 | 本机 128 GB 判定 |
|---|---|---|---|---|---|
| UD-Q4_K_XL(4-bit)| 111.3 GB | 93.5% | 0.045 | 112 GB | ❌ 超预算 |
| UD-IQ4_XS(4-bit)| 93.7 GB | 91.1% | 0.079 | 112 GB | ⚠️ 临界(余 ~9 GiB)|
| **UD-Q3_K_XL(3-bit)** | **90.0 GB** | **90.4%** | 0.100 | **90 GB** | ✅✅ **本机选用** |
| UD-IQ3_XXS(3-bit)| 82.0 GB | 87.6% | 0.157 | 90 GB | ✅ 备选(更省内存)|
| UD-Q2_K_XL(2-bit)| 78.9 GB | 85.2% | 0.213 | 79 GB | ✅ 余量大,质量取舍 |
| UD-IQ1_M(1-bit)| 74.5 GB | 82.4% | 0.302 | 75 GB | ❌ 质量崩坏线 |
| UD-IQ1_S(1-bit)| 72.5 GB | 80.2% | 0.375 | 75 GB | ❌ 仅冒烟用 |

> 质量数据来源:unsloth 官方 Divergence-300@32(与 BF16 逐 token 对比)。
> **关键结论:3-bit 已进入高质量区间(IQ3_XXS 与 IQ4_XS 仅差 3.5pp),远未到 1-bit 的崩坏线。**

---

## 2. 引擎/后端对比(同一模型,llama-bench 实测)

| 后端 | 解码速度(tg64)| 相对值 | 说明 |
|---|---|---|---|
| **CUDA 全卸载(-ngl 999)** | **24.22 ± 0.12 t/s** | 1.0x | 生产配置,全部层进 GPU |
| 纯 CPU(-ngl 0,20 线程)| 1.90 t/s | 0.08x | GPU 卸载 = **12.7 倍** |

> 模型识别:qwen4exp A3B Q3_K - Medium,83.80 GiB,176.94B 参数

---

## 3. 上下文配置对比(实测三档)

| 配置 | n_ctx_slot | 服务端 decode | 服务端 prompt(短) | 内存已用 | 内存余量 |
|---|---|---|---|---|---|
| 32K | 32768 | 25.1-25.6 t/s | 73-93 tok/s | ~99 GiB | ~22 GiB |
| 128K | 131072 | 19.9 t/s | 80.8 tok/s | ~93 GiB | ~28 GiB |
| **262K(原生最大)** | **262144** | **22.1 t/s** | **36.5 tok/s** | **~102 GiB** | **~19 GiB** |

**两个关键发现**:

1. **262K 上下文的内存增量只有几 GB**(32K→262K,KV 增加 ~3 GiB)
   ——QSA 稀疏注意力(预算固定 512 块/2048 token)+ GDN 固定状态的红利,日志确认 `kv_unified='true'`
2. **decode 几乎不受上下文窗口影响**(19.9-25.6 t/s 波动属正常),但 **prompt 处理随窗口变大显著变慢**(80.8→36.5 tok/s):
   - llama.cpp 按最大上下文预分配 KV/索引结构,每 token 固定开销随窗口变大(实现层)
   - 真长上下文时 QSA indexer 搜索空间翻倍(架构层)

---

## 4. 速度明细(原始 timings 记录,服务端实测)

### 4.1 冒烟测试(32K 配置,首轮请求含模型加载后首次推理)

| 阶段 | 耗时 | 速度 |
|---|---|---|
| prompt eval | 760.55 ms / 56 tokens | 73.63 tok/s |
| eval(decode)| 3088.34 ms / 80 tokens | 25.58 tok/s |
| 端到端 | 3.86 s | — |

### 4.2 长生成测试(32K 配置,数数 200 tokens)

| 阶段 | 耗时 | 速度 |
|---|---|---|
| prompt eval | 258.30 ms / 24 tokens | 92.92 tok/s |
| eval(decode)| 7921.65 ms / 200 tokens | 25.12 tok/s |
| 端到端 | 8.3 s | 24.1 tok/s(含 prefill)|

### 4.3 128K 配置短请求

| 阶段 | 耗时 | 速度 |
|---|---|---|
| prompt eval | 226.11 ms / 19 tokens | 84.03 tok/s |
| decode | — | 19.9 tok/s |

### 4.4 262K 配置短请求

| 阶段 | 耗时 | 速度 |
|---|---|---|
| prompt eval | — | 36.5 tok/s |
| decode | — | 22.1 tok/s |

---

## 5. 投机解码方案对比(全部实测,当前 24 t/s 是天花板)

| 方案 | 实测 decode | 接受率 | 投机是否生效 | 失败原因 |
|---|---|---|---|---|
| 基线(无投机)| 24.2 t/s | — | — | — |
| ngram-simple | 24.8 t/s | 25.6%(11/43)| ✅ 生效 | 仅重复文本场景有价值 |
| draft-dflash | 24.0 t/s | 无统计 | ❌ 静默退化 | GGUF 无 DND 草稿结构 |
| draft-dspark | 24.1 t/s | 无统计 | ❌ 静默退化 | 同上 |
| draft-mtp | 不可用 | — | ❌ | GGUF 无 MTP 头(1224 张量扫描 0 命中)|
| 0.2B 外部草稿 | 24.1 t/s | 无统计 | ❌ | 未训练,接受率≈0 |

> 详细分析(机制/数学/因果链/解锁路线)见 **speculative-analysis.md**

---

## 6. 内存占用明细

| 项 | 值 |
|---|---|
| 模型权重(mmap)| ~90 GB |
| KV/状态(262K 配置)| ~4-6 GB |
| 运行时/系统/页面缓存 | ~8-10 GB |
| 峰值(262K)| ~102 GiB / 128 GB |
| 运行中可用 | ~19-22 GiB |

---

## 7. 与参考数据的对比(27B 同机先例)

| 模型 | 引擎/方案 | 解码速度 |
|---|---|---|
| Qwen3.8-27B NVFP4 | SGLang + DFlash2 投机(k=12)| **55.3 t/s** |
| Qwen3.8-27B NVFP4 | SGLang 无投机(带宽极限 273GB/s ÷ 20.4GB)| ~13.4 t/s |
| Qwen3.8-Flash-Next 180B | llama.cpp 无投机(本机)| **22-24 t/s** |

> 参考:Flash-Next 无投机实测已接近 27B 无投机带宽极限的 2 倍(6B 激活 × 更小权重的红利);
> 若 MTP 投机落地(1.5-2.5x),预期 30-50 t/s,将明显超越 27B+DFlash2 组合。

---

## 8. 测试方法论(可复现)

```bash
# 解码基线(llama-bench 需在服务停止时运行,避免同时加载两份模型)
llama-bench -m model.gguf -ngl 999 -t 20 -p 0 -n 64 -r 2

# 服务端速度:API 响应的 timings 字段
curl -s http://127.0.0.1:8889/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"..."}],"max_tokens":200}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['timings'])"

# 投机测试:独立端口 + 小上下文(避免影响生产)
llama-server -m model.gguf --port 8890 --ctx-size 16384 --spec-type <方案> --jinja

# 内存:free -h(服务运行中)
```

---

## 9. 结论速览

1. **UD-Q3_K_XL 是 128 GB 机器的性价比最优档**(90.4% 质量 / 90 GB 内存需求 / 余量充足)
2. **262K 上下文免费**(架构红利),但大窗口会拖慢短提示词 prefill(实现税+架构税)
3. **解码 22-24 t/s 是当前天花板**,瓶颈为 PLE 查表延迟 + GDN 串行 + MoE 小矩阵
4. **提速路线:MTP 自投机(1.5-2.5x)→ SGLang DFlash2(2-4x)**,前置条件见 mtp-tracker.md
