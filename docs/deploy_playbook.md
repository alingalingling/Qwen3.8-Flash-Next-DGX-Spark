# Deployment Playbook — Qwen3.8-Flash-Next on DGX Spark (pre-written, awaiting target version)

**[English](deploy_playbook.md) | [中文](deploy_playbook.zh.md)**

> Target machine: NVIDIA DGX Spark (GB10), 128 GB unified memory, native Blackwell FP4
> Target version: ≤95 GB (full quant: NVFP4/GGUF/FP8, non-MLX, ≥3-bit) — not yet released; this manual is a contingency plan

## 0. Fit-check formula (verify when a new version appears)
- Total safetensors+gguf on disk ≤ 95 GB (ideally ≤ 85 GB)
- No load-time BF16 re-inflation (PLE table must stay FP8/low-bit or be offloadable)
- Not MLX (Apple-only), not IQ1/oQ1/Q1 (1-bit quality is poor)
- Runtime peak = disk size + KV/activations (~15-25 GB) + system (~8 GB) ≤ 115 GB

## 1. Three runtime paths (by priority)

### Path A: GGUF + llama.cpp (preferred; wait for PR #27742 merge)
- Prereq: llama.cpp qwen4_exp support (PR #27742, daniehanchen, mergeable=True)
- Build: git clone https://github.com/ggml-org/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j
- Run: ./build/bin/llama-server -m Qwen3.8-Flash-Next-XX.gguf --ctx-size 32768 -ngl 99
- The PLE n-gram table can use --no-mmap or offload to RAM per llama.cpp's new parameters (PR author uses the Gemma-3N approach: one large offloadable table)
- Estimated memory: 2-bit 85 GB / 3-bit 100 GB (unsloth official table); suggested context 16K-64K

### Path B: SGLang + full NVFP4 (wait for RadixArk/community full quant)
- pip install sglang or use the official container; needs qwen4_exp support
- python -m sglang.launch_server --model-path <full-NVFP4> --quantization modelopt_fp4 --context-length 32768 --mem-fraction-static 0.80

### Path C: unsloth runtime (Unsloth Desktop, GGUF can run first)
- If llama.cpp hasn't merged but unsloth has published levels, Unsloth Desktop can run them in the meantime

## 2. Local to-dos (can do now)
1. Fix nvidia-smi NVML: Unknown Error (driver 580.173.02 is loaded; check container/NVML library versions)
2. Prepare the llama.cpp build environment (gcc 13.3 present; CMake needed)
3. Reserve 150 GB+ disk (3.4 TB currently free — plenty)
4. Context management: full 262K is not realistic; target 16K-64K

## 3. Quality red lines
- 1-bit (IQ1/UD-IQ1/oQ1/Q1): excluded (explicitly)
- 2-bit: acceptable but needs testing (oQ2 had a quality hold; IQ2 series untested)
- 3-bit/4-bit: recommended range
- Full NVFP4: ideal
