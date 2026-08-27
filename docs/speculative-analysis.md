# Full Measured Speculative-Decoding Analysis (Qwen3.8-Flash-Next on DGX Spark)

**[English](speculative-analysis.md) | [中文](speculative-analysis.zh.md)**

> Platform: NVIDIA DGX Spark (GB10, 128 GB unified memory, 273 GB/s)
> Model: unsloth UD-Q3_K_XL (90 GB, 3-bit, 176.94B params)
> Engine: llama.cpp qwen4exp branch (commit 035e22731, CUDA)
> Date: 2026-08-27 | Every conclusion comes from local measurement, not armchair reasoning

---

## 1. Why care about speculative decoding

Qwen3.8-Flash-Next is a 180B hybrid MoE (only ~6B params active per token); decode is limited by weight-read bandwidth and fixed latency, measured locally at **22-24 t/s**.

---

## 2. llama.cpp speculation panorama (--spec-type supports 11 kinds)

This build (0.3.0-dev, qwen4exp branch) supports:

```
--spec-type none, draft-simple, draft-eagle3, draft-mtp,
 draft-dflash, draft-dspark,
 ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache
```

| Category | Method | Draft source | Needs extra model/tensors? |
|---|---|---|---|
| Self-speculation | draft-mtp | model's own MTP head | GGUF must contain MTP tensors |
| Engine draft | draft-dflash / draft-dspark | DeltaNet structure | model must carry DND/DFlash draft tensors |
| External draft | draft-simple / draft-eagle3 | separate small draft model | trained same-tokenizer small model |
| Model-free | ngram-* | repeated context fragments | nothing extra |

---

## 3. Measured: all five methods, one by one

### 3.0 Baseline

| Config | Decode | Note |
|---|---|---|
| GPU full offload (-ngl 999) | **24.2 t/s** | llama-bench tg64, production config |
| CPU only (-ngl 0) | 1.9 t/s | GPU offload = 12.7× |

### 3.1 ngram-simple — repetitive-text only

```bash
llama-server ... --spec-type ngram-simple --spec-ngram-simple-size-n 8
```

| Measured item | Value |
|---|---|
| Decode speed | 24.8 t/s (baseline 24.2, +2%) |
| Draft acceptance | 25.6% (11 accepted / 43 generated) |
| Draft length | 12 tokens avg |

**Verdict**: the mechanism works (acceptance stats appear) but the gain is negligible. Reason: n-gram matches repeated fragments from the context, so it only helps **highly repetitive text** (code/logs/formatted output); general conversation has almost nothing to match. Also, the theoretical 1.34× from 25.6% acceptance is eaten by the verification path overhead.

**When to use**: if your workload is dominated by repetitive text (log analysis, code-gen templates), turn it on; for general use it's **not worth it**.

### 3.2 draft-dflash / draft-dspark — silent no-op

```bash
llama-server ... --spec-type draft-dflash # or draft-dspark
```

| Measured item | Value |
|---|---|
| Decode speed | 24.0 / 24.1 t/s (no difference from baseline) |
| Speculation log | **zero output** (no acceptance, no draft activity) |

**Verdict**: the flags are accepted but **silently degrade** to no speculation. Reason: these methods require the model to carry specific DND/DFlash draft-structure tensors (like the DND weights SGLang's DFlash2 uses). unsloth's GGUF doesn't include them, so llama.cpp finds no draft structure and just skips.

**Key insight**: despite the name, `draft-dspark` also depends on model structure — **it's not a hardware limitation, the tensors simply aren't in the model file**.

### 3.3 draft-mtp — GGUF has no MTP head

```bash
llama-server ... --spec-type draft-mtp --spec-draft-n-max 2
```

| Measured item | Value |
|---|---|
| GGUF MTP tensor scan | **0 hits** (probe_mtp.py measured; 1224 tensors across 3 shards) |
| Parameter check | 176.94B = 180B − 4B (MTP head removed) |
| unsloth standalone MTP module | none (all 7 levels lack it) |

**Verdict**: draft-mtp needs the model to carry an MTP head (4B params, 31 tensors). unsloth stripped it during quantization (saving ~2 GB), so **self-speculation is out of the question** *for the stripped GGUFs as-is*. By contrast, the 27B GGUF **keeps** its MTP head — that's why a single flag speeds it up.

> **UPDATE (2026-08-27 night)**: the "as-is" caveat no longer applies. [dzannotti/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/dzannotti/Qwen3.8-Flash-Next-MTP-GGUF) ships a **standalone MTP draft head** (`MTP-Q4_K_M.gguf`, 2.44 GB, standard quant) usable as `-md` next to any Flash-Next GGUF, plus a patch (`qwen4exp-mtp-draft-head.patch`) that adds the MTP graph to the qwen4exp branch — applied and built successfully on this machine (19:16, `--spec-type draft-mtp` supported). Author-measured on the same 128 GB class machine: **UD-Q4_K_XL 20.3 → 35.8 t/s (code), acceptance 0.90**. See [mtp-tracker.md](mtp-tracker.md) for run commands and the no-re-download injection script (`merge-mtp-shard.py`).

### 3.4 External small draft (draft-simple) — full experiment with the 0.2B tiny model

**Candidate**: [inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B](https://huggingface.co/inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B)
(0.33 GB, 0.16B params, keeps all GDN/QSA/PLE/MoE architecture components)

**Experiment (all measured)**:

| Step | Result |
|---|---|
| ① Download | 332 MB, seconds |
| ② Convert with qwen4exp converter | success (113 tensors, 321 MB) — proves the converter also handles small new-arch models |
| ③ Tokenizer consistency | **identical** (official vs tiny produce the exact same token-ID sequences for the same text) |
| ④ Mount as --model-draft, measure | 24.1 t/s, no acceptance stats, no speedup |
| ⑤ Standalone generation test | **blank output** — weights are randomly initialized / untrained |

**Verdict**: steps ①-③ all passed (format and tokenizer satisfy draft conditions); it dies at the last step — **the model was never trained, predictions are random, acceptance ≈ 0**.

**Quick method for judging whether a draft model is usable** (distilled into methodology):
1. **Run the draft standalone first** — if output is blank/garbage, it's untrained; discard it
2. **Then compare tokenizers** (same text → same token-ID sequence) — mismatch means token misalignment; unusable
3. Only if both pass is it worth mounting for a live test

---

## 4. Why an "untrained" model can't be a draft (acceptance math)

At each position the target model's true output is a distribution over 248,044 tokens; the probability that a draft's guess lands on the target model's most-desired token:

```
Random draft (untrained): acceptance ≈ 1/248,044 ≈ 0.0004%
Trained draft: acceptance ≈ 30-70% (same-family small model / distilled model)
```

When an untrained draft guesses k tokens, the target almost always **rejects the very first one** → speculation fully degrades, the draft-generation + verification overhead is pure waste → speed matches baseline or is slightly lower (measured 24.1 vs 24.2).

**Analogy**: speculation is a "fast talker" reciting a line for an expert to verify quickly. The talker must **actually know how to speak** (be trained) to guess what the expert would say; an untrained model is a "mute assistant" — all the parts are there (a mouth), but everything it utters is gibberish, and the expert shakes its head every time.

---

## 5. Causal-chain evolution: from the "24 t/s ceiling" to full unlock (history)

**The 2026-08-27 daytime causal chain** (both ends of speculation broken then; the "24 t/s ceiling"
conclusion was overturned by that night's measurements):

```
Speculation = draft source + engine execution path

Draft source (all three paths were broken):
 ├─ MTP self-spec → GGUF has no MTP head (stripped by unsloth, measured) → solved: dzannotti head
 ├─ DFlash/DSpark → GGUF has no DND draft tensors (flags accepted, silent) → still none
 └─ External draft → no trained same-tokenizer small model (0.2B untrained) → still none (and pointless in MoE)
 (ngram-simple only works on repetitive text, ≈0 for general) → solved: ngram-mod long spans

Engine execution path (incomplete then):
 └─ llama.cpp qwen4exp MTP load/speculate = WIP (PR #27742 has no MTP commit) → solved: bea3b12d tree + patch
```

**What unlocked that night (all measured locally)**:

| Method | Requires | Measured gain | Status |
|---|---|---|---|
| **ngram-mod** | built-in, zero cost | code copy 58.7 t/s (+135%) | unlocked |
| **ngram-map-k / map-k4v** | built-in, zero cost | code copy **82.4 / 84.2 t/s** (+230%, acc 0.849) | unlocked (2026-08-28, +40% over mod) |
| **MTP (dzannotti head)** | 2.44GB head + bea3b12d verified-tree patch | counting 57.3 / code 54.6 | unlocked |
| **MTP + ngram-mod stacked** | the two above | code copy **83.0 t/s** (+232%) | unlocked |
| Q4_K_XL + PLE-offload + stacked | download 111GB + PLE-offload flags | code 70.1 t/s (93.5% quality) | unlocked ( 8K only) |
| SGLang + DFlash2 | full NVFP4 + mature runtime | — | runtime unpublished, watchlisting |

---

## 6. Bottleneck analysis: the three fixed-latency sources behind bare 22-24 t/s (still present after the unlock)

Theory: per token read 6B active experts × ~0.45 B ≈ 2.7 GB ÷ 273 GB/s ≈ **~100 t/s**

Bare 24 t/s; the gap comes from three fixed-latency sources. **After the unlock, code-type tasks
reach 58-83 t/s by parallelizing the verify step, but the per-token latency sources remain:**

| Latency source | Mechanism | Status |
|---|---|---|
| PLE n-gram random lookups | 51B table; every lookup is a cache-missing random access | mitigated by NVMe-PLE offload + table warm-up; residual is architectural |
| GDN recurrent state | serial per-token update, cannot parallelize | architecture-inherent |
| Small MoE GEMMs | expert intermediate dim is only 640; each GEMM is tiny; GPU launch overhead dominates | canreuse patch enables CUDA graph reuse (graphs reused 0→52+, +2.8%) |

> Note: 0xBakeer's "small draft model is useless" analysis also explains why speculation must go
> long-span (ngram-mod mean 37-60 tokens) instead of small-step drafts: in top-10/512 MoE, k draft
> tokens activate up to k×10 different experts, weight traffic scales with k, nothing to amortize.

---

## 7. Unlock roadmap: when can it get faster (2026-08-27 night: partially unlocked )

| Trigger | What's needed | Expected gain | Current status |
|---|---|---|---|
| **ngram-map-k4v speculation** | nothing (built into llama.cpp) | **3.4× on copy-type tasks** (84.2 t/s) | **UNLOCKED** — zero extra memory; every token verified (since 2026-08-28, replaces ngram-mod) |
| **MTP self-speculation** | dzannotti standalone MTP head + bea3b12d verified-tree patch | **2.2× on regular output** (57.3 t/s) | **UNLOCKED** — segfaults on 035e22731; must use the verified tree |
| **MTP + ngram-map-k4v stacked** | the two above combined | **4.3× on copy-type tasks** (108.4 t/s) | **UNLOCKED** — `--spec-type draft-mtp,ngram-map-k4v` |
| **SGLang/vLLM + NVFP4** | single-machine NVFP4 (Felliks/MaxLaurence PLE-on-NVMe / starkweatherdigital 109GB full-quant) | 17.9-42 single / 93-132 4-stream | runnable but 109-120GB memory; full-quant weights uploading |

**Recommended production configs** (code/agentic workloads):
```bash
# Option A (recommended, zero extra memory): ngram-map-k4v (since 2026-08-28, replaces ngram-mod)
llama-server -m model.gguf -ngl 999 -t 20 --spec-type ngram-map-k4v --jinja

# Option B (strongest, 108 t/s code copy): MTP + ngram-map-k4v stacked (needs verified tree + MTP head)
LLAMA_ATTN_ROT_DISABLE=1 llama-server -m model.gguf \
 -md MTP-Q4_K_M.gguf -ngld 999 \
 --spec-type draft-mtp,ngram-map-k4v --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
 -ngl 999 -fa on -ctk q8_0 -ctv q8_0
```

> Gain shape: ngram/MTP benefit scales with "how much of the output comes from the context
> or regular patterns" — tool-driven file edits, code completion, lists/args benefit hugely;
> free-form prose gets almost nothing (~26 t/s). Measurement trap (0xBakeer): repeating the
> same prompt inflates results 2.8× — vary prompts when benchmarking.

---

## 8. Reproducible methodology

```
# Decode baseline (run with the server stopped to avoid loading two model copies)
llama-bench -m model.gguf -ngl 999 -t 20 -p 0 -n 64 -r 2

# Speculation tests: separate port + small context, don't disturb production
llama-server -m model.gguf --port 8890 --ctx-size 16384 \
 --spec-type <method> --jinja

# Read acceptance: "draft acceptance = X (accepted / generated)" in the server log
# Read speed: timings.predicted_per_token_ms in the API response

# Judge a draft model: run it standalone first, then compare tokenizers
llama-cli -m draft.gguf -n 40 # blank/garbage output → untrained, discard
```

---

## 9. Conclusions (major update 2026-08-27 night)

1. **The "24 t/s ceiling" is broken**: ngram-mod (58.7 t/s code copy), MTP (57.3 t/s counting),
 **MTP+ngram-mod stacked (83.0 t/s code copy, 3.3×)** — all measured locally, zero/near-zero extra memory
2. **The gain tracks output predictability, not "is it code"**: tool-driven edits/completion/
 structured output benefit hugely; free-form prose gets almost nothing (~26 t/s); measurement
 must vary prompts (repeats inflate 2.8×)
3. **MTP head injection works**: dzannotti standard-quant head (2.44 GB) + bea3b12d verified-tree
 patch; merge-mtp-shard.py embeds it into any existing GGUF (~2% vs `-md`); tree 035e22731 is
 incompatible (segfault)
4. **cafe-llama.cpp fork permanently abandoned**: twice OOM loading MTP drafts
5. Full NVFP4 (101.7 GB, with MTP head) now exists (provsalt); SGLang route watchlisted
6. All conclusions in this document come from local measurement; reproduction and feedback welcome

---

## References

- [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) — qwen4exp support
- [llama.cpp PR #27739](https://github.com/ggml-org/llama.cpp/pull/27739) — predecessor, includes MTP+PLE-offload implementation
- [inference-optimization 0.2B tiny model](https://huggingface.co/inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B) — measured, not usable as a draft
- Companion tool: scripts/probe_mtp.py (GGUF MTP-head probe)
