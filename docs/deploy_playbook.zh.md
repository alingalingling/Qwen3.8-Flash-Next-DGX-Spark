# DGX Spark 部署 Qwen3.8-Flash-Next 执行手册(预写,等待目标版本)

**[中文](deploy_playbook.zh.md) | [English](deploy_playbook.md)**


> 目标机器:NVIDIA DGX Spark(GB10),128 GB 统一内存,Blackwell FP4 原生支持
> 目标版本:≤95 GB(NVFP4/GGUF/FP8 全量化,非 MLX,≥3-bit),未出现,此手册为预案

## 0. 验收公式(新版本出现时核对)
- 磁盘 safetensors+gguf 合计 ≤ 95 GB(理想 ≤ 85 GB)
- 无"加载时反量化回 BF16"的膨胀(PLE 表必须保持 FP8/低比特或可 offload)
- 非 MLX(Apple 专属)、非 IQ1/oQ1/Q1(1-bit 质量差)
- 运行时峰值 = 磁盘大小 + KV/激活(~15-25 GB)+ 系统(~8 GB)≤ 115 GB

## 1. 三条运行路径(按优先级)

### 路径 A:GGUF + llama.cpp(首选,等 PR #27742 合并)
- 前提:llama.cpp 合并 qwen4_exp 支持(PR #27742,daniehanchen,mergeable=True)
- 构建:git clone https://github.com/ggml-org/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j
- 运行:./build/bin/llama-server -m Qwen3.8-Flash-Next-XX.gguf --ctx-size 32768 -ngl 99
- PLE n-gram 表可 --no-mmap 或按 llama.cpp 新参数 offload 到 RAM(PR 里作者采用 Gemma-3N 方案,单张大表可 offload)
- 预估内存:2-bit 85 GB / 3-bit 100 GB(unsloth 官方表),上下文建议 16K-64K

### 路径 B:SGLang + 全量 NVFP4(等 RadixArk/社区出全量化版)
- pip install sglang 或使用官方容器;需 qwen4_exp 支持
- python -m sglang.launch_server --model-path <全量NVFP4> --quantization modelopt_fp4 --context-length 32768 --mem-fraction-static 0.80

### 路径 C:unsloth 运行时(Unsloth Desktop,GGUF 可先跑)
- 若 llama.cpp 未合并而 unsloth 已传档位,可先用 Unsloth Desktop 跑

## 2. 本机待办(现在就能做)
1. 修复 nvidia-smi NVML: Unknown Error(驱动 580.173.02 已加载,检查容器/NVML 库版本)
2. 装好 llama.cpp 构建环境(gcc 13.3 已有,CMake 需装)
3. 预留磁盘 150 GB+(当前可用 3.4 TB,足够)
4. 上下文管理:262K 全量不可能,目标 16K-64K

## 3. 质量红线
- 1-bit(IQ1/UD-IQ1/oQ1/Q1):排除(用户明确)
- 2-bit:可接受但需实测(oQ2 曾出现质量 hold,IQ2 系列待测)
- 3-bit/4-bit:推荐区间
- NVFP4 全量化:理想
