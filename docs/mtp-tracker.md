# MTP-Head Tracking & Injection Plan (Qwen3.8-Flash-Next)

**[English](mtp-tracker.md) | [中文](mtp-tracker.zh.md)**

> Background: speculative decoding (draft-mtp) can substantially speed up decode.
> Prereq: the GGUF must carry MTP-head tensors. unsloth's Flash-Next quants **strip the MTP head by default** (180B → 176.94B params, verified by probe_mtp.py).

## Current status (2026-08-27)

| Item | Status |
|---|---|
| llama.cpp qwen4exp PR (#27742) | open, mergeable; **20 commits, no MTP commit**; author WIP |
| Predecessor PR #27739 | closed, but **already includes MTP + PLE-offload implementation**; community reminded the author to reference it |
| unsloth GGUF | all 7 levels lack the MTP head; no standalone MTP module in the repo |
| Standalone draft model | no trained qwen4exp small model exists (the 0.2B test artifact is untrained) |

## Unlock conditions (any one enables speculation)

1. **llama.cpp PR finishes MTP support** + unsloth publishes an MTP-bearing quant → directly use `--spec-type draft-mtp --spec-draft-n-max 2`
2. **Inject the MTP head yourself** (after llama.cpp support): see plan below

## Injection plan (execute after llama.cpp MTP support lands)

```bash
# 1) Download official BF16 (360 GB; ensure disk space)
hf download Qwen/Qwen3.8-Flash-Next --local-dir ~/models/Qwen3.8-Flash-Next-BF16

# 2) Convert to GGUF with the qwen4exp converter (needs the PR-branch convert_hf_to_gguf.py)
python3 convert_hf_to_gguf.py ~/models/Qwen3.8-Flash-Next-BF16 \
  --outfile ~/models/flashnext-bf16.gguf --outtype q8_0   # 180 GB intermediate

# 3) Extract the 31 MTP tensors (*mtp*), quantize to Q4_0 (~2 GB)

# 4) Inject the MTP tensors into unsloth's UD-Q3_K_XL GGUF with the gguf-py library
#    (update the metadata tensor list + append data + rewrite shards)

# 5) Verify and launch
python3 scripts/probe_mtp.py ~/models/Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL/
llama-server -m model.gguf --spec-type draft-mtp --spec-draft-n-max 2 ...
```

> Note: the complete injection script (inject_mtp.py) will be provided after llama.cpp MTP support merges;
> first check whether unsloth has released an MTP-bearing version (prefer the official one, no injection needed).

## Monitor signals (tracked by monitor.py in this repo)

- llama.cpp PR #27742 gets an MTP-related commit / merges
- unsloth repo gains MTP-bearing filenames (e.g., *-MTP-* or mtp.gguf)
- Baekpica's mixed-quant passes its validation gates (its MQ version includes MTP experts at Q8_0)

## References

- [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)
- [llama.cpp PR #27739](https://github.com/ggml-org/llama.cpp/pull/27739) (includes MTP implementation)
