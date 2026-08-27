# Qwen3.8-Flash-Next on DGX Spark — Single-Node Deployment & Tuning

**[English](README.md) | [中文](README.zh.md)**

> **Deploy the quantized version of the 180B hybrid-MoE model on a single NVIDIA DGX Spark (GB10, 128 GB unified memory, no multi-GPU, no cluster) and serve it stably**
> Includes: one-click deployment on the llama.cpp qwen4exp branch, a GGUF MTP-head probe,
> an HF model monitor, full measured benchmarks, and an MTP injection plan.

![status](https://img.shields.io/badge/status-deployed-green) ![license](https://img.shields.io/badge/license-Apache--2.0-blue) ![model](https://img.shields.io/badge/model-Qwen3.8--Flash--Next-orange) ![single](https://img.shields.io/badge/single-node-DGX_Spark-blue)

> **Project freshness note**
>
> This project was created on **the release day of Qwen3.8-Flash-Next (2026-08-26)**; every deployment step, measurement, and conclusion reflects **the state of the community ecosystem at that time**:
> llama.cpp qwen4_exp support (PR #27742 branch), all unsloth Dynamic 3.0 quant levels, and every inference engine and quant repo that existed then.
>
> The ecosystem is still growing fast: llama.cpp upstream support is about to merge, **MTP speculative decoding is now within reach** (a standalone MTP draft head + patch is available and built locally — see docs/mtp-tracker.md), and Baekpica's mixed-quant (including the SSD-PLE variant) plus a full NVFP4 quant are in validation/release.
> **This project will therefore keep updating** — as the ecosystem evolves, the README conclusions, docs data, tool scripts, and roadmap will be revised accordingly.

## What's in this repo

In one sentence: **we deployed a 180B model that officially requires dual GB300 GPUs on a single DGX Spark, and open-sourced the whole process (how to install, how fast, where it's stuck, how to speed it up).**

- **Deploy it**: [docs/deploy_playbook.md](docs/deploy_playbook.md) — deployment manual: complete commands from download/compile to launch, plus the data for checking whether a version fits your machine.
- **Real numbers**: [docs/benchmarks.md](docs/benchmarks.md) — all measured comparisons: how to pick among 7 quant levels, how much faster GPU is than CPU, speeds at 32K/128K/262K context, memory footprints, and raw per-test timings.
- **Make it faster?**: [docs/speculative-analysis.md](docs/speculative-analysis.md) — all five speculative-decoding paths (ngram / DFlash / DSpark / MTP / small draft) **measured end to end**: why 24 t/s is the current ceiling, where the bottlenecks are, which path may reach 40-60 t/s, and when it unlocks.
- **Track acceleration progress** ：see [docs/mtp-tracker.md](docs/mtp-tracker.md): why the MTP head (the key to speculation speedup) was stripped from the quant, how to inject it back, llama.cpp's support status, and three "go" signals.
- **Run it now** → use the `scripts/` tools: `run_qwen38_q3.sh` starts/stops the server, `probe_mtp.py` checks in 3 seconds whether your GGUF can speculate, `monitor.py` watches HF for new releases.

## Background

Qwen3.8-Flash-Next is the preview of the Qwen4 architecture (180B-parameter hybrid MoE, 512 experts with top-10 routing, GDN linear attention + QSA sparse attention + a 51B PLE n-gram lookup table, 262K native context).
The official validation environment is dual GB300/B300; this repo proves **a single DGX Spark can deploy its quantized version**.

This repo documents and open-sources all tools and measurements for deploying it on **a single NVIDIA DGX Spark (GB10, 128 GB unified memory)** — including the current pain point (missing MTP) and its solution.

## Measured results

| Item | Value |
|---|---|
| Model | unsloth UD-Q3_K_XL (90 GB, 90.4% quality retention) |
| Engine | llama.cpp qwen4exp branch (PR #27742) |
| Context | **262K (native max)** |
| Decode speed | **22-24 t/s** (GPU offload; CPU-only is 1.9 t/s) |
| Memory | ~102 / 128 GB |

## Speculative speedups (measured 2026-08-27 night)

| Setup | Code copy/edit | Prose | Memory | Quality |
|---|---:|---:|---:|---:|
| Baseline (no speculation) | 25.0 t/s | 25.4 t/s | 83 GB | 87.6% |
| **ngram-mod** (free) | **58.7 t/s** | 26.1 t/s | 83 GB | 87.6% |
| **MTP + ngram-mod** (verified tree) | **83.0 t/s** | 29.6 t/s | 86 GB | 87.6% |
| **Q4_K_XL + PLE-offload + MTP+ngram** | **70.1 t/s** | 19.3 t/s | 86 GB | **93.5%** |
| **🏆 Q3_K_XL + PLE + MTP+ngram (262K production)** | **78.9 t/s** | 21.5 t/s | **70 GB** | 90.4% |
| Reference: 2×DGX Spark NVFP4+MTP4 (community) | 50-55 t/s | ~33 t/s | two machines | 4-bit class |

> 🚀 A single DGX Spark with stacked speculation beats the community's two-machine NVFP4+MTP4
> setup on structured output. NVMe-PLE (`-ot per_layer_token_embd=CPU -lm mmap`) serves the PLE
> table from the NVMe page cache: Q4_K_XL at 82 GB (official requirement 112 GB), Q3_K_XL at
> 70 GB (262K window). Details in [docs/benchmarks.md](docs/benchmarks.md) and
> [docs/mtp-tracker.md](docs/mtp-tracker.md).

## Quant levels

| Level | Size | Same-top-1% vs BF16 | Memory req. | Verdict @ 128 GB |
|---|---|---|---|---|
| UD-Q4_K_XL (4-bit) | 111.3 GB | 93.5% | 112 GB | Over budget |
| UD-IQ4_XS (4-bit) | 93.7 GB | 91.1% | 112 GB | Marginal |
| **UD-Q3_K_XL (3-bit)** | **90.0 GB** | **90.4%** | **90 GB** | **Chosen** |
| UD-IQ3_XXS (3-bit) | 82.0 GB | 87.6% | 90 GB | Alternative |
| UD-Q2_K_XL (2-bit) | 78.9 GB | 85.2% | 79 GB | Tradeoffs |
| UD-IQ1 series (1-bit) | 72.5-74.5 GB | 80-82% | 75 GB | Quality collapse |

## Docs (suggested reading order)

| Doc | Contents | For whom |
|---|---|---|
| [docs/deploy_playbook.md](docs/deploy_playbook.md) | Deployment manual: fit-check formula, three runtime paths, quality red lines | People deploying |
| [docs/benchmarks.md](docs/benchmarks.md) | Full measured comparisons: quants / engines / context / speculation / raw timings | People validating or comparing |
| [docs/speculative-analysis.md](docs/speculative-analysis.md) | ⭐ Full measured speculation analysis: five paths, acceptance math, causal chain, unlock roadmap | People tuning or researching |
| [docs/mtp-tracker.md](docs/mtp-tracker.md) | MTP-head tracking: llama.cpp PR status, injection plan, monitor signals | People following acceleration progress |

## Quick start

```bash
# 1. Download the model (90 GB)
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --include "UD-Q3_K_XL/*" --local-dir ~/models/Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL

# 2. Build llama.cpp (qwen4exp branch, PR #27742)
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cd ~/llama.cpp
git fetch origin pull/27742/head:qwen4exp && git checkout qwen4exp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(nproc)

# 3. Start the server (OpenAI-compatible, port 8889, 262K context)
bash scripts/run_qwen38_q3.sh start

# 4. Verify
curl http://127.0.0.1:8889/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8","messages":[{"role":"user","content":"Hello"}],"max_tokens":100,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Layout

```
├── scripts/
│   ├── run_qwen38_q3.sh      # Server start/stop script (start/stop/status/logs)
│   ├── qwen38-q3.service     # systemd autostart template
│   ├── probe_mtp.py          # ⭐ GGUF MTP-head probe (pure Python, zero deps)
│   └── monitor.py            # HF quant-repo monitor (incl. llama.cpp PR status)
├── docs/
│   ├── deploy_playbook.md        # Deployment manual (fit-check / 3 paths / red lines)
│   ├── benchmarks.md             # Measured benchmarks (speed/context/memory/speculation)
│   ├── speculative-analysis.md   # ⭐ Full measured speculation analysis (5 paths + causal chain + roadmap)
│   └── mtp-tracker.md            # MTP-head tracking + injection plan
└── LICENSE                   # Apache-2.0
```

## Key scripts

### probe_mtp.py — can your GGUF speculate?
Speculative decoding (draft-mtp) requires the GGUF to contain an MTP head. unsloth stripped it from the Flash-Next quants (180B → 176.94B params), while the 27B GGUF keeps it. Check in 3 seconds:
```bash
python3 scripts/probe_mtp.py /path/to/model.gguf   # or a shard directory
```

### monitor.py — watch for new versions
Dual-channel HF scan (official quantized tag + name search), auto-classified as
`DEPLOYABLE / UPLOADING / TOO_BIG / MLX_ONLY / Q1_BAD`, plus llama.cpp PR status.

## Known findings

1. **🚀 Speculative decoding is now UNLOCKED (2026-08-27 night)**: ngram-mod gives a free 2.3×
   on code-type tasks; **MTP head + ngram-mod stacked reaches 3.3× (83 t/s)**; prose stays ~26 t/s
   (gain = output predictability). The old "24 t/s ceiling" finding is obsolete —
   see docs/speculative-analysis.md.
2. **Start background services with setsid**: prevents the terminal timeout from killing the service along with the command.
3. **Open the context boldly**: QSA sparse KV keeps the 262K memory cost to a few GB; but **large windows slow down short-prompt processing** (36.5 vs 80.8 tok/s at 262K vs 128K) — use 128K for short conversations.
4. **Memory-safety rules**: only one model resident at a time; the cafe-llama.cpp fork OOM'd twice (permanently abandoned); system watchdog (system_watchdog.sh, 5GB/5s) auto-starts via crontab @reboot.

## Roadmap

- [ ] llama.cpp PR #27742 merge (upstream; now 54 commits, still no MTP commit)
- [x] MTP support → done locally via dzannotti head + bea3b12d verified tree (MTP+ngram-mod = 83 t/s)
- [x] 0xBakeer NVMe-PLE recipe verified (Q4_K_XL at 82 GB; final production recipe Q3 = 78.9 t/s @262K / 70 GB)
- [ ] Evaluate Baekpica mixed-quant once validation gates pass (ds4 runtime)
- [ ] Full NVFP4: provsalt (101.6 GB, PLE also NVFP4) / Lewfkrad W4-PLE / 4p89 all appeared, but runtimes are unpublished/unverified — not usable yet; triggers tracked in the plan doc §10.2

## Acknowledgements

- [unsloth](https://huggingface.co/unsloth) — Dynamic 3.0 GGUF
- [llama.cpp](https://github.com/ggml-org/llama.cpp) PR #27742 (daniehanchen)

## License

Apache-2.0. Model weights are under the Qwen Community License; fetch them from the original repos — this repo does not bundle model files.
