# NVFP4 Single-Machine Recipe Manual (2026-08-28)

**[English](nvfp4-recipes.md) | [中文](nvfp4-recipes.zh.md)**

> Complete steps for the three public ways to run Qwen3.8-Flash-Next NVFP4 on a single
> 128 GB GB10 machine. ⚠️ All are memory-critical (109-120/128 GB) with far less headroom
> than this repo's GGUF production recipe (70 GB) — evaluate before trying.

## 1. Felliks (SGLang, RadixArk 135GB + PLE-on-NVMe, 262K + MTP-213)

```bash
git clone https://github.com/Felliks/qwen38-flash-next-one-dgx-spark
cd qwen38-flash-next-one-dgx-spark
./run-spark.sh prepare   # docker build pinned image + download RadixArk (rev 7b71922…) + build 51.2GB PLE mmap + verify
./run-spark.sh serve     # --memory 116g hard cap; SGLang MTP-213 config
./run-spark.sh smoke / status / logs / stop
```

- Measured (Ascent GX10, same GB10): 32.7 t/s single; 93-103 aggregate (4 streams); 262K; 120.45GB; 9.5-min cold load
- Mechanism: two SHA-256-guarded patches = PLE exact-FP8 mmap on NVMe (256MiB 4-way row cache) + SM121 QSA SDPA fallback (FA4 varlen broken on GB10)

## 2. MaxLaurence (SGLang, same RadixArk, measured on DGX Spark)

```bash
git clone https://github.com/MaxLaurence/qwen38-flash-next-sglang-dgx-spark
cd qwen38-flash-next-sglang-dgx-spark
# recipe.lock.json fully hash-pinned; PLE exact-FP8 demand-paged from NVMe
```

- Measured (DGX Spark): 17.9 prose / 42.06 copy-friendly t/s single; 39.3 / 132.2 (4 streams); 262K; 109GB used / 12GB free; 501s cold start

## 3. starkweatherdigital (vLLM, 109GB full quant, PLE also 4-bit)

```bash
docker pull docker.io/jstarkg/vllm-gb10-flashnext:0.28-sm121-r3   # prebuilt image, 9GB
# wait for HF weights: starkweatherdigital/qwen3.8-flash-next-nvfp4 (109GB)
VLLM_PLE_NVFP4=1 vllm serve /path/to/flashnext-nvfp4 \
  --served-model-name qwen3.8-flash-next \
  --quantization modelopt_fp4 --moe-backend marlin --enforce-eager \
  --gpu-memory-utilization 0.92 --max-model-len 32768 \
  --max-num-batched-tokens 8192 --max-num-seqs 16 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"qwen3_8_flash_next_mtp","num_speculative_tokens":1}'
```

- Measured (DGX Spark): 24.6 t/s single with MTP (80% acceptance) / 16.8 without; 32K window; ~120GB
- The only "full-quant (PLE also 4-bit) + fully public runtime" setup; weights still uploading

## 4. Unusable setups quick reference

| Setup | Why not |
|---|---|
| RadixArk 135GB original | PLE re-inflated to BF16 → 236GB, needs GB300 |
| Inferact / primitive-ai 171-186GB | PLE uncompressed, over budget |
| provsalt 101.6GB | qwen38-nvfp4-ple plugin unpublished |
| Lewfkrad W4-PLE ~114GB | SM120-validated, not GB10; runtime unpublished |
| local-inference-lab 4p89 | WIP |
| axiomofmind W4A16 GGUF 168GB | over budget |
| MLX / ROCmFP4 series | Apple / AMD only |
