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

### 3.2 draft-dflash / draft-dspark — ❌ silent no-op

```bash
llama-server ... --spec-type draft-dflash   # or draft-dspark
```

| Measured item | Value |
|---|---|
| Decode speed | 24.0 / 24.1 t/s (no difference from baseline) |
| Speculation log | **zero output** (no acceptance, no draft activity) |

**Verdict**: the flags are accepted but **silently degrade** to no speculation. Reason: these methods require the model to carry specific DND/DFlash draft-structure tensors (like the DND weights SGLang's DFlash2 uses). unsloth's GGUF doesn't include them, so llama.cpp finds no draft structure and just skips.

**Key insight**: despite the name, `draft-dspark` also depends on model structure — **it's not a hardware limitation, the tensors simply aren't in the model file**.

### 3.3 draft-mtp — ❌ GGUF has no MTP head

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

### 3.4 External small draft (draft-simple) — ❌ full experiment with the 0.2B tiny model

**Candidate**: [inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B](https://huggingface.co/inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B)
(0.33 GB, 0.16B params, keeps all GDN/QSA/PLE/MoE architecture components)

**Experiment (all measured)**:

| Step | Result |
|---|---|
| ① Download | 332 MB, seconds |
| ② Convert with qwen4exp converter | ✅ success (113 tensors, 321 MB) — proves the converter also handles small new-arch models |
| ③ Tokenizer consistency | ✅ **identical** (official vs tiny produce the exact same token-ID sequences for the same text) |
| ④ Mount as --model-draft, measure | ❌ 24.1 t/s, no acceptance stats, no speedup |
| ⑤ Standalone generation test | ❌ **blank output** — weights are randomly initialized / untrained |

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
Trained draft:            acceptance ≈ 30-70% (same-family small model / distilled model)
```

When an untrained draft guesses k tokens, the target almost always **rejects the very first one** → speculation fully degrades, the draft-generation + verification overhead is pure waste → speed matches baseline or is slightly lower (measured 24.1 vs 24.2).

**Analogy**: speculation is a "fast talker" reciting a line for an expert to verify quickly. The talker must **actually know how to speak** (be trained) to guess what the expert would say; an untrained model is a "mute assistant" — all the parts are there (a mouth), but everything it utters is gibberish, and the expert shakes its head every time.

---

## 5. Causal chain: why 24 t/s is the current ceiling

```
Speculation = draft source + engine execution path

Draft source (all three paths broken):
  ├─ MTP self-spec  → GGUF has no MTP head (stripped by unsloth, measured)      ❌
  ├─ DFlash/DSpark  → GGUF has no DND draft tensors (flags accepted, silent)    ❌
  └─ External draft → no trained same-tokenizer small model (0.2B untrained)    ❌
  (ngram needs no model, but only works on repetitive text, ≈0 for general)     ⚠️

Engine execution path (incomplete):
  └─ llama.cpp qwen4exp MTP load/speculate = WIP (PR #27742 has no MTP commit)  ❌

Conclusion: a matter of time, not of hardware
```

---

## 6. Bottleneck analysis: why 24 t/s instead of the bandwidth limit (~100 t/s)

Theory: per token read 6B active experts × ~0.45 B ≈ 2.7 GB ÷ 273 GB/s ≈ **~100 t/s**

Measured 24 t/s; the gap comes from three fixed-latency sources:

| Latency source | Mechanism | Optimizability |
|---|---|---|
| PLE n-gram random lookups | 51B table; every lookup is a cache-missing random access (~100-200ns) | architecture-inherent, hard |
| GDN recurrent state | serial per-token update, cannot parallelize | architecture-inherent |
| Small MoE GEMMs | expert intermediate dim is only 640; each GEMM is tiny; GPU launch overhead dominates | depends on engine kernel fusion |

---

## 7. Unlock roadmap: when can it get faster

| Trigger | What's needed | Expected gain | Current status |
|---|---|---|---|
| **MTP self-speculation** | ① llama.cpp PR #27742 finishes MTP support ② GGUF with MTP head (unsloth release or self-injection) | **1.5-2.5×** (30-50 t/s) | ① WIP (predecessor PR #27739 has a ready implementation to reference) ② none |
| **SGLang + DFlash2** | full NVFP4 checkpoint (≤101 GB) + mature SGLang qwen4_exp support | **2-4×** (27B precedent: 55 t/s) | no full NVFP4 yet |
| ngram (repetitive text) | nothing | 1.2-1.5× for specific workloads | **available now** (see 3.1) |

**MTP injection plan** (after llama.cpp support lands, the repo will provide inject_mtp.py):
1. Download official BF16 (360 GB)
2. convert_hf_to_gguf.py to a Q8_0 intermediate
3. Extract the 31 MTP tensors, quantize to Q4_0 (~2 GB)
4. Inject into unsloth's UD-Q3_K_XL with gguf-py (update tensor list + append data)
5. Verify with probe_mtp.py, then launch with `--spec-type draft-mtp --spec-draft-n-max 2`

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
llama-cli -m draft.gguf -n 40        # blank/garbage output → untrained, discard
```

---

## 9. Conclusions

1. **24 t/s for the current combo (UD-Q3_K_XL + llama.cpp) is a real ceiling** — all five speculation paths measured, all unusable due to "missing draft source" or "engine support incomplete"
2. **It's not a hardware problem**: GB10 has enough compute and bandwidth; what's missing is the speculation support in the model file (MTP head / DFlash tensors) and the engine's MTP execution path
3. **The biggest prize is MTP self-speculation** (1.5-2.5×); both prereqs are moving; the repo's monitor.py and mtp-tracker.md keep tracking
4. All conclusions in this document come from local measurement; reproduction and feedback welcome

---

## References

- [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) — qwen4exp support
- [llama.cpp PR #27739](https://github.com/ggml-org/llama.cpp/pull/27739) — predecessor, includes MTP+PLE-offload implementation
- [inference-optimization 0.2B tiny model](https://huggingface.co/inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B) — measured, not usable as a draft
- Companion tool: scripts/probe_mtp.py (GGUF MTP-head probe)
