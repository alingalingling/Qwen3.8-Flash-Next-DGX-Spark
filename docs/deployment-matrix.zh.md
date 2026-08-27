# 全市场单机部署方案总对比(2026-08-28 盘点)

**[中文](deployment-matrix.zh.md) | [English](deployment-matrix.md)**

> 口径:能在**单台 128GB 级机器**(DGX Spark / Ascent GX10 等 GB10 平台)上部署 Qwen3.8-Flash-Next
> 的全部公开方案;速度分代码类/普通文本;状态截至 2026-08-28,所有数字来自作者实测(非估算)。

## A. GGUF + llama.cpp 路线(唯一本机全家桶验证)

| 方案 | 量化/质量 | 单流速度 | 并发聚合 | 上下文 | 内存 | 投机 | 状态 |
|---|---|---|---|---|---|---|---|
| **本机生产配方 Q3+PLE+MTP+map-k4v** | Q3_K_XL 90.4% | 代码 78.9 / 散文 21.5 | **77.5(2 并发)** | **262K** | **70GB** | MTP+map-k4v | ✅ 本机实测 |
| 本机速度配方 IQ3+MTP+map-k4v | IQ3_XXS 87.6% | **代码 108.4** | 93-104(2 并发,8K) | 16K | 86GB | MTP+map-k4v | ✅ 本机实测 |
| 本机零依赖 IQ3+map-k4v | IQ3_XXS 87.6% | 代码 84.2 / 散文 27.8 | 同上 | 16K | 83GB | map-k4v | ✅ 本机实测 |
| 本机质量配方 Q4+PLE+MTP+map-k4v | Q4_K_XL 93.5% | 代码 70.1 | — | **8K 仅** | 86GB | MTP+map-k4v | ✅ 本机实测(⚠️短窗) |
| [0xBakeer](https://github.com/0xBakeer/qwen38-flash-next-spark) Q4+PLE+ngram-mod | Q4_K_XL 93.5% | 代码 46-74 / 散文 22-23 | —(parallel 1) | 262K | ~77GB 常驻 | ngram-mod | ✅ 单机验证(首创 NVMe-PLE) |

## B. SGLang + NVFP4 路线(单机可跑,PLE 落盘)

| 方案 | 量化/质量 | 单流速度 | 并发聚合 | 上下文 | 内存 | 投机 | 状态 |
|---|---|---|---|---|---|---|---|
| [Felliks](https://github.com/Felliks/qwen38-flash-next-one-dgx-spark) RadixArk+PLE 落盘 | NVFP4 专家版 4-bit 级 | 32.7(普通) | **93-103(4 流)** | 262K | 120.45GB | MTP-213 | ✅ Ascent GX10 验证(同 GB10) |
| [MaxLaurence](https://github.com/MaxLaurence/qwen38-flash-next-sglang-dgx-spark) 同思路 | 同上 | 17.9 散文 / 42.1 复制 | 39.3 / 132.2(4 流) | 262K | 109GB | NEXTN/MTP | ✅ DGX Spark 实测 |
| RadixArk 原版 135GB | NVFP4 专家版 | — | — | — | **236GB** | — | ❌ 需 GB300 |
| Lewfkrad W4-PLE ~114GB | PLE 压 W4 | — | — | — | ~114GB+开销 | — | ❌ SM120 验证,运行时未公开 |

## C. vLLM + 全量化 NVFP4 路线

| 方案 | 量化/质量 | 单流速度 | 并发 | 上下文 | 内存 | 投机 | 状态 |
|---|---|---|---|---|---|---|---|
| [starkweatherdigital](https://huggingface.co/starkweatherdigital/qwen3.8-flash-next-nvfp4) 109GB | **全量化**(PLE 也 4-bit) | **24.6(MTP,80% 接受率)** / 16.8 无 | 16 seq | **32K** | ~120GB | MTP(自带) | ⏳ DGX Spark 实测过,权重上传中,镜像公开 |
| provsalt 101.6GB | 全量化(PLE NVFP4) | — | — | 4K | 临界(自测 14GB swap) | — | ❌ 插件未公开 |

## D. 其他平台/不可用(记录备查)

| 方案 | 说明 | 判定 |
|---|---|---|
| MLX 系列(jedisct1 oQ4e / Vontra / inferencerlabs / Sawfwair) | Apple Silicon 专属格式 | ❌ 非本平台 |
| ROCmFP4 系列(kingjones777 / agentionai / MrLordCat) | AMD Strix Halo 专属(Vulkan/ROCm) | ❌ 非本平台 |
| Baekpica 混合量化 98.5GB + ds4 运行时 | 自研运行时(v0.6.3-dfm),验证门未过 | ❌ 不可用 |
| axiomofmind W4A16-NVFP4-GGUF 168GB | 专家 NVFP4 + BF16 注意力单文件 GGUF | ❌ PLE-offload 后仍 ~140GB |
| PixelML Dual-DGX-Spark / tonyd2wild 双机 | 双机方案(后者结构化 50-55 t/s) | ❌ 非单机 |
| 27B + SGLang + DFlash2 | 小模型参照 55.3 t/s | 参照(不同模型) |

## 完整部署步骤(可跑方案)

### B1. Felliks(SGLang,262K+MTP-213)

```bash
git clone https://github.com/Felliks/qwen38-flash-next-one-dgx-spark
cd qwen38-flash-next-one-dgx-spark
./run-spark.sh prepare   # docker build 固定镜像 + 下载 RadixArk(rev 7b71922…)+ 转 51.2GB PLE mmap + 256 行校验
./run-spark.sh serve     # --memory 116g 硬上限;SGLang MTP-213(2 步/top-k1/3 草稿)
./run-spark.sh smoke / status / logs / stop   # http://127.0.0.1:8000/v1
```

### B2. MaxLaurence(SGLang,recipe 固定)

```bash
git clone https://github.com/MaxLaurence/qwen38-flash-next-sglang-dgx-spark
cd qwen38-flash-next-sglang-dgx-spark   # recipe.lock.json 全哈希固定,5 个 fail-closed 源转换
```

### C1. starkweatherdigital(vLLM,109GB 全量化)

```bash
docker pull docker.io/jstarkg/vllm-gb10-flashnext:0.28-sm121-r3   # 预构建镜像 9GB;等 HF 权重上传
VLLM_PLE_NVFP4=1 vllm serve /path/to/flashnext-nvfp4 \
  --served-model-name qwen3.8-flash-next --quantization modelopt_fp4 \
  --moe-backend marlin --enforce-eager --gpu-memory-utilization 0.92 \
  --max-model-len 32768 --max-num-batched-tokens 8192 --max-num-seqs 16 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"qwen3_8_flash_next_mtp","num_speculative_tokens":1}'
```

> ⚠️ 三个方案全部内存临界(109-120/128GB,余 8-12GB):尝试前 drop_caches、加载曲线监控、挂看门狗;
> 与本仓库 GGUF 生产配方(70GB,余 51GB)相比余量小,个人单机首选 GGUF。

## 结论排名

- **单机单请求(代码类)**:本机 Q3 生产配方 78.9(262K)/ IQ3 108.4(16K) > 0xBakeer 46-74 > Felliks/MaxLaurence 33-42 > starkweatherdigital 24.6
- **并发(4 流级)**:MaxLaurence 132 / Felliks 93-103 > 本机 2 流 77.5-104(4 流未测,内存允许)
- **内存安全**:本机 70GB(余 51GB)>> 其余方案 109-120GB(余 8-12GB)
- **性价比结论**:个人单机 = 本机 GGUF 配方;多人并发服务 = Felliks/MaxLaurence SGLang;全量化质量 = starkweatherdigital(权重上传中)
