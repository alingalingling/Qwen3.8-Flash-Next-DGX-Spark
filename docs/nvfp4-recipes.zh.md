# NVFP4 单机部署配方手册(2026-08-28)

**[中文](nvfp4-recipes.zh.md) | [English](nvfp4-recipes.md)**

> 三个能在单台 128GB GB10 机器上跑 Qwen3.8-Flash-Next NVFP4 的公开方案完整步骤。
> ⚠️ 全部内存临界(109-120/128GB),与本仓库 GGUF 生产配方(70GB)相比余量小,尝试前请评估。

## 1. Felliks(SGLang,RadixArk 135GB + PLE 落盘,262K + MTP-213)

```bash
git clone https://github.com/Felliks/qwen38-flash-next-one-dgx-spark
cd qwen38-flash-next-one-dgx-spark
./run-spark.sh prepare   # docker build 固定镜像 + 下载 RadixArk(rev 7b71922…) + 转 51.2GB PLE mmap + 校验
./run-spark.sh serve     # --memory 116g 硬上限;SGLang MTP-213 配置
./run-spark.sh smoke / status / logs / stop
```

- 实测(Ascent GX10,同 GB10):单流 32.7 t/s;四流聚合 93-103;262K;120.45GB;冷加载 9.5 分钟
- 机制:两个 SHA-256 守卫补丁 = PLE exact-FP8 mmap 落 NVMe(256MiB 4 路行缓存)+ SM121 QSA SDPA 回退(FA4 varlen 在 GB10 编译不正确)

## 2. MaxLaurence(SGLang,同 RadixArk,DGX Spark 实测)

```bash
git clone https://github.com/MaxLaurence/qwen38-flash-next-sglang-dgx-spark
cd qwen38-flash-next-sglang-dgx-spark
# recipe.lock.json 全哈希固定;PLE exact-FP8 demand-paged from NVMe
```

- 实测(DGX Spark):散文 17.9 / 复制友好 42.06 t/s 单流;四流 39.3 / 132.2;262K;109GB 用/12GB 余;冷启动 501s

## 3. starkweatherdigital(vLLM,109GB 全量化,PLE 也 4-bit)

```bash
docker pull docker.io/jstarkg/vllm-gb10-flashnext:0.28-sm121-r3   # 预构建镜像,9GB
# 等 HF 权重上传:starkweatherdigital/qwen3.8-flash-next-nvfp4(109GB)
VLLM_PLE_NVFP4=1 vllm serve /path/to/flashnext-nvfp4 \
  --served-model-name qwen3.8-flash-next \
  --quantization modelopt_fp4 --moe-backend marlin --enforce-eager \
  --gpu-memory-utilization 0.92 --max-model-len 32768 \
  --max-num-batched-tokens 8192 --max-num-seqs 16 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"qwen3_8_flash_next_mtp","num_speculative_tokens":1}'
```

- 实测(DGX Spark):24.6 t/s 单流 + MTP(80% 接受率)/ 16.8 无;32K 窗口;~120GB
- 唯一"全量化(PLE 也 4-bit)+ 运行时全公开"方案;权重上传中

## 4. 不可用方案速查

| 方案 | 原因 |
|---|---|
| RadixArk 135GB 原版 | PLE 反量化 BF16 → 236GB,需 GB300 |
| Inferact / primitive-ai 171-186GB | PLE 不压缩,超预算 |
| provsalt 101.6GB | qwen38-nvfp4-ple 插件未公开 |
| Lewfkrad W4-PLE ~114GB | SM120 验证,非 GB10,运行时未公开 |
| local-inference-lab 4p89 | WIP |
| axiomofmind W4A16 GGUF 168GB | 超预算 |
| MLX / ROCmFP4 系列 | Apple / AMD 专属 |
