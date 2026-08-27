# Single-Machine Deployment Matrix — full market comparison (2026-08-28)

**[English](deployment-matrix.md) | [中文](deployment-matrix.zh.md)**

> Scope: every public way to run Qwen3.8-Flash-Next on a **single 128 GB-class machine**
> (DGX Spark / Ascent GX10, GB10 platform); speed split by code-type / prose; status as of
> 2026-08-28; all numbers from authors' own measurements (not estimates).

## A. GGUF + llama.cpp route (only fully-verified family on this machine)

| Setup | Quant / quality | Single-stream | Concurrent aggregate | Context | Memory | Speculation | Status |
|---|---|---|---|---|---|---|---|
| **This machine: Q3+PLE+MTP+map-k4v (production)** | Q3_K_XL 90.4% | code 78.9 / prose 21.5 | **77.5 (2 concurrent)** | **262K** | **70GB** | MTP+map-k4v | ✅ measured here |
| This machine: IQ3+MTP+map-k4v (speed) | IQ3_XXS 87.6% | **code 108.4** | 93-104 (2 concurrent, 8K) | 16K | 86GB | MTP+map-k4v | ✅ measured here |
| This machine: IQ3+map-k4v (zero-dep) | IQ3_XXS 87.6% | code 84.2 / prose 27.8 | same | 16K | 83GB | map-k4v | ✅ measured here |
| This machine: Q4+PLE+MTP+map-k4v (quality) | Q4_K_XL 93.5% | code 70.1 | — | **8K only** | 86GB | MTP+map-k4v | ✅ measured here (⚠️ short window) |
| [0xBakeer](https://github.com/0xBakeer/qwen38-flash-next-spark) Q4+PLE+ngram-mod | Q4_K_XL 93.5% | code 46-74 / prose 22-23 | — (parallel 1) | 262K | ~77GB resident | ngram-mod | ✅ single-machine (NVMe-PLE pioneer) |

## B. SGLang + NVFP4 route (runnable single-machine, PLE on disk)

| Setup | Quant / quality | Single-stream | Concurrent aggregate | Context | Memory | Speculation | Status |
|---|---|---|---|---|---|---|---|
| [Felliks](https://github.com/Felliks/qwen38-flash-next-one-dgx-spark) RadixArk+PLE-on-NVMe | NVFP4 expert-only, 4-bit class | 32.7 (general) | **93-103 (4 streams)** | 262K | 120.45GB | MTP-213 | ✅ Ascent GX10 (same GB10) |
| [MaxLaurence](https://github.com/MaxLaurence/qwen38-flash-next-sglang-dgx-spark) same idea | same | 17.9 prose / 42.1 copy | 39.3 / 132.2 (4 streams) | 262K | 109GB | NEXTN/MTP | ✅ DGX Spark measured |
| RadixArk original 135GB | NVFP4 expert-only | — | — | — | **236GB** | — | ❌ needs GB300 |
| Lewfkrad W4-PLE ~114GB | PLE packed W4 | — | — | — | ~114GB+overhead | — | ❌ SM120-validated, runtime unpublished |

## C. vLLM + full-NVFP4 route

| Setup | Quant / quality | Single-stream | Concurrency | Context | Memory | Speculation | Status |
|---|---|---|---|---|---|---|---|
| [starkweatherdigital](https://huggingface.co/starkweatherdigital/qwen3.8-flash-next-nvfp4) 109GB | **full NVFP4 (PLE also 4-bit)** | **24.6 (MTP, 80% acc)** / 16.8 none | 16 seq | **32K** | ~120GB | MTP (built-in) | ⏳ measured on DGX Spark; weights uploading; public image |
| provsalt 101.6GB | full NVFP4 (PLE NVFP4) | — | — | 4K | critical (14GB swap in its own test) | — | ❌ plugin unpublished |

## D. Other platforms / not usable (for reference)

| Setup | Note | Verdict |
|---|---|---|
| MLX series (jedisct1 oQ4e / Vontra / inferencerlabs / Sawfwair) | Apple Silicon only | ❌ wrong platform |
| ROCmFP4 series (kingjones777 / agentionai / MrLordCat) | AMD Strix Halo only (Vulkan/ROCm) | ❌ wrong platform |
| Baekpica mixed-quant 98.5GB + ds4 runtime | custom runtime (v0.6.3-dfm), validation gates not passed | ❌ unusable |
| axiomofmind W4A16-NVFP4-GGUF 168GB | single-file GGUF, NVFP4 experts + BF16 attention | ❌ ~140GB even with PLE-offload |
| PixelML Dual-DGX-Spark / tonyd2wild 2-machine | two-machine setups (latter: 50-55 t/s structured) | ❌ not single-machine |
| 27B + SGLang + DFlash2 | smaller-model reference 55.3 t/s | reference only |

## Ranking

- **Single-machine, single request (code-type)**: this machine Q3 production 78.9 (262K) / IQ3 108.4 (16K) > 0xBakeer 46-74 > Felliks/MaxLaurence 33-42 > starkweatherdigital 24.6
- **Concurrency (4-stream class)**: MaxLaurence 132 / Felliks 93-103 > this machine 2-stream 77.5-104 (4-stream untested, memory allows)
- **Memory safety**: this machine 70GB (51GB headroom) >> all others 109-120GB (8-12GB headroom)
- **Bottom line**: personal single-machine = this repo's GGUF recipe; multi-user serving = Felliks/MaxLaurence SGLang; full-quant quality = starkweatherdigital (weights uploading)
