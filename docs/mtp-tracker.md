# MTP-Head Tracking & Injection Plan (Qwen3.8-Flash-Next)

**[English](mtp-tracker.md) | [中文](mtp-tracker.zh.md)**

> Background: speculative decoding (draft-mtp) can substantially speed up decode.
> Prereq: the GGUF must carry MTP-head tensors. unsloth's Flash-Next quants **strip the MTP head by default** (180B → 176.94B params, verified by probe_mtp.py).

## Current status (2026-08-27, night update)

| Item | Status |
|---|---|
| llama.cpp qwen4exp PR (#27742) | open, mergeable; **42 commits**, still no MTP commit; author WIP |
| Predecessor PR #27739 | closed, but **already includes MTP + PLE-offload implementation**; community reminded the author to reference it |
| **dzannotti MTP head** ([repo](https://huggingface.co/dzannotti/Qwen3.8-Flash-Next-MTP-GGUF)) | **BREAKTHROUGH**: standalone `MTP-Q4_K_M.gguf` (2.44 GB, standard quant, any backend) + `qwen4exp-mtp-draft-head.patch` (459 lines, reconciled PR #27739 MTP graph + converter `--mtp` export) |
| **Local build (this machine)** | patch applied **cleanly** on our qwen4exp branch (baseline `bea3b12d` matches the patch's target tree); **rebuilt successfully** — `llama-server` now supports `--spec-type draft-mtp` (build 2026-08-27 19:16) |
| unsloth GGUF | all 7 levels lack the MTP head; standalone head from dzannotti works as `-md` draft |
| Standalone draft model | no trained qwen4exp small model exists; **MTP draft head (4B, from official bf16 checkpoint) is the right draft** |

## Author-measured numbers (AMD Strix Halo, 128 GB unified — same class as DGX Spark)

| backend, target | bare | + MTP | acceptance (code / prose) |
|---|---:|---:|---|
| ROCm, UD-Q4_K_XL | 20.3 t/s | **35.8** code / 22.6 prose | 0.90 / 0.74 |
| ROCm, UD-IQ4_XS | 18.0 / 18.6 | 32.8 / 22.1 | 0.84 / 0.68 |
| Vulkan (RADV, Laurent's fork), UD-IQ4_XS | 24.2 / 24.3 | **37.2** code / **30.3** prose | 0.88 / 0.82 |

Recommended flags: `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75`
(p-min 0.75 matters more than depth: the head carries its own MoE, so every drafted token is a real forward pass).
**Must set `LLAMA_ATTN_ROT_DISABLE=1`** (upstream quantized-KV rotation #21038 is not supported by the qwen4exp attention path; the server aborts at load without it).

## Run (standalone draft head)

```bash
LLAMA_ATTN_ROT_DISABLE=1 llama-server \
  -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -md Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf -ngld 999 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -c 262144
```

## Injection plan (no re-download of the 82/90 GB model — merge-mtp-shard.py)

`patches/merge-mtp-shard.py` (from dzannotti's repo, archived in this project) attaches the
head as an extra shard of **any existing split qwen4exp GGUF**:

1. keep the target's shards 2..N and rename them `-0000N-of-0000(M+1).gguf`
2. drop in the rewritten shard 1 (metadata: `block_count 49`, `nextn_predict_layers 1`,
   `compress_ratios` extended, `split.count M+1`) and the new last shard (29 head tensors,
   same bytes as the Q4_K_M head above)
3. run with `--spec-type draft-mtp` and **no `-md`**; the draft context is created against
   the target and shares its memory (~0.5 GB less than the separate head)

Measured: within a few percent of the `-md` form (34.5 vs 36.7 t/s on code) — the
convenience is the point, not speed.

## This machine's plan

1. IQ3_XXS (82 GB) download in progress (~25 MB/s) → then the benchmark chain runs 4 groups:
   IQ3 baseline / IQ3 + cafe fork + quimmedes draft / Q3_K_XL baseline / **IQ3 + official MTP head** (new group)
2. MTP-Q4_K_M head (2.44 GB) downloading
3. After validation, optionally inject the head into IQ3_XXS (or even Q3_K_XL) via merge-mtp-shard.py

## Monitor signals (tracked by monitor.py in this repo)

- llama.cpp PR #27742 gets an MTP-related commit / merges
- unsloth repo gains MTP-bearing filenames (e.g., *-MTP-* or mtp.gguf)
- Baekpica's mixed-quant passes its validation gates (its MQ version includes MTP experts at Q8_0)
- dzannotti's repo updates (head requants, better patches)

## References

- [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)
- [llama.cpp PR #27739](https://github.com/ggml-org/llama.cpp/pull/27739) (includes MTP implementation)
- [dzannotti/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/dzannotti/Qwen3.8-Flash-Next-MTP-GGUF) (standalone MTP head + patch + merge script)
