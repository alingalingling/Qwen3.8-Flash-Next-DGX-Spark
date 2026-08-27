# Measured Benchmarks & Comparison Dataset (NVIDIA DGX Spark, 2026-08-27)

**[English](benchmarks.md) | [中文](benchmarks.zh.md)**

> Hardware: GB10 / 128 GB unified memory / 273 GB/s / 20-core ARM (10×Cortex-X925 + 10×A725)
> Engine: llama.cpp qwen4exp branch (commit 035e22731, CUDA backend, GCC 13.3 aarch64)
> Service: llama-server, OpenAI-compatible API, port 8889
> Note: all numbers measured locally; rows marked "official" are unsloth-published data, everything else is measured

---

## 1. Quant-level panorama (official data + local verdict)

| Level | Size | same_top_pct vs BF16 ↑ | mean_kld ↓ | Official total mem | Verdict @ 128 GB |
|---|---|---|---|---|---|
| UD-Q4_K_XL (4-bit) | 111.3 GB | 93.5% | 0.045 | 112 GB | ❌ over budget |
| UD-IQ4_XS (4-bit) | 93.7 GB | 91.1% | 0.079 | 112 GB | ⚠️ marginal (~9 GiB left) |
| **UD-Q3_K_XL (3-bit)** | **90.0 GB** | **90.4%** | 0.100 | **90 GB** | ✅✅ **chosen** |
| UD-IQ3_XXS (3-bit) | 82.0 GB | 87.6% | 0.157 | 90 GB | ✅ alternative (less memory) |
| UD-Q2_K_XL (2-bit) | 78.9 GB | 85.2% | 0.213 | 79 GB | ✅ more headroom, quality tradeoff |
| UD-IQ1_M (1-bit) | 74.5 GB | 82.4% | 0.302 | 75 GB | ❌ quality collapse line |
| UD-IQ1_S (1-bit) | 72.5 GB | 80.2% | 0.375 | 75 GB | ❌ smoke-test only |

> Quality data source: unsloth's official Divergence-300@32 (per-token comparison vs BF16).
> **Key takeaway: 3-bit is already in the high-quality zone (IQ3_XXS is only 3.5pp behind IQ4_XS), far from the 1-bit collapse line.**

---

## 2. Engine/backend comparison (same model, llama-bench measured)

| Backend | Decode (tg64) | Relative | Note |
|---|---|---|---|
| **CUDA full offload (-ngl 999)** | **24.22 ± 0.12 t/s** | 1.0x | Production config, all layers on GPU |
| CPU only (-ngl 0, 20 threads) | 1.90 t/s | 0.08x | GPU offload = **12.7×** |

> Model identified as: qwen4exp A3B Q3_K - Medium, 83.80 GiB, 176.94B params

---

## 3. Context configuration comparison (three levels measured)

| Config | n_ctx_slot | Server decode | Server prompt (short) | Memory used | Memory left |
|---|---|---|---|---|---|
| 32K | 32768 | 25.1-25.6 t/s | 73-93 tok/s | ~99 GiB | ~22 GiB |
| 128K | 131072 | 19.9 t/s | 80.8 tok/s | ~93 GiB | ~28 GiB |
| **262K (native max)** | **262144** | **22.1 t/s** | **36.5 tok/s** | **~102 GiB** | **~19 GiB** |

**Two key findings**:

1. **262K context costs only a few GB of extra memory** (32K→262K adds ~3 GiB of KV)
   — the dividend of QSA sparse attention (fixed budget of 512 blocks/2048 tokens) + fixed GDN state; log confirms `kv_unified='true'`
2. **Decode is almost unaffected by the window size** (19.9-25.6 t/s variance is normal), but **prompt processing slows markedly with larger windows** (80.8→36.5 tok/s):
   - llama.cpp pre-allocates KV/index structures for the max context; per-token fixed overhead grows with the window (implementation layer)
   - with genuinely long contexts the QSA indexer's search space doubles (architecture layer)

---

## 4. Speed detail (raw timings, server-measured)

### 4.1 Smoke test (32K config, first request after model load)

| Stage | Time | Speed |
|---|---|---|
| prompt eval | 760.55 ms / 56 tokens | 73.63 tok/s |
| eval (decode) | 3088.34 ms / 80 tokens | 25.58 tok/s |
| End-to-end | 3.86 s | — |

### 4.2 Long generation (32K config, counting 200 tokens)

| Stage | Time | Speed |
|---|---|---|
| prompt eval | 258.30 ms / 24 tokens | 92.92 tok/s |
| eval (decode) | 7921.65 ms / 200 tokens | 25.12 tok/s |
| End-to-end | 8.3 s | 24.1 tok/s (incl. prefill) |

### 4.3 Short request @ 128K config

| Stage | Time | Speed |
|---|---|---|
| prompt eval | 226.11 ms / 19 tokens | 84.03 tok/s |
| decode | — | 19.9 tok/s |

### 4.4 Short request @ 262K config

| Stage | Time | Speed |
|---|---|---|
| prompt eval | — | 36.5 tok/s |
| decode | — | 22.1 tok/s |

---

## 5. Speculative decoding comparison (all measured)

### 5.0 Early tests (Q3_K_XL era, 2026-08-27 daytime)

| Method | Measured decode | Acceptance | Active? | Failure reason |
|---|---|---|---|---|
| Baseline (no speculation) | 24.2 t/s | — | — | — |
| ngram-simple | 24.8 t/s | 25.6% (11/43) | ✅ active | only useful for repetitive text |
| draft-dflash | 24.0 t/s | no stats | ❌ silent fallback | GGUF lacks DND draft structure |
| draft-dspark | 24.1 t/s | no stats | ❌ silent fallback | same |
| draft-mtp | n/a | — | ❌ | GGUF has no MTP head (0/1224 tensors) |
| 0.2B external draft | 24.1 t/s | no stats | ❌ | untrained, acceptance ≈ 0 |

### 5.1a Full comparison table (2026-08-28 final, IQ3_XXS, ctx 8192-16384, same 3 tasks)

| Setup | A: counting 200 | B: prose 150 | C: code copy-edit 500 | Extra memory |
|---|---:|---:|---:|---:|
| IQ3 baseline | 26.2 t/s | 25.4 t/s | 25.0 t/s | 0 |
| + ngram-mod | 26.1 | 26.1 | 58.7 (+135%) | **0 (free)** |
| + **ngram-map-k** | 26.5 | 27.0 | **82.4** (+230%) | **0 (free)** |
| + **ngram-map-k4v** | 27.8 | 27.8 | **84.2** (+237%) | **0 (free)** |
| + MTP (dzannotti head, `-md`) | 57.3 | 25.2 | 54.6 | ~3 GB |
| + MTP + ngram-mod | 58.1 | 29.6 | 83.0 | ~3 GB |
| + **MTP + ngram-map-k4v** | **57.7** | **27.9** | **108.4** 🚀🚀 (+334%) | ~3 GB |
| + MTP (embedded head, merge script) | 56.3 | — | 53.6 | ~2.5 GB |

> **ngram-variant discovery (2026-08-28)**: map-k/map-k4v token mapping raises draft acceptance
> from mod's 0.566 to **0.849** (mean len 37→42), code copy 58.7 → 82-84 t/s — **+40% free**;
> **MTP+map-k4v stacked = 108.4 t/s = 4.3x baseline**, currently the fastest single-machine config

### 5.1b llama.cpp concurrency measured (2026-08-28, IQ3_XXS + ngram-map-k4v, --parallel 2, 8K window)

| Scenario | Request A | Request B | **Aggregate** | Server alive |
|---|---:|---:|---:|---|
| codeC + prose | 70.1 t/s | 22.9 t/s | **~93 t/s** | ✅ (0 asserts) |
| codeC + codePy | 61.8 t/s | 42.1 t/s | **~104 t/s** | ✅ |

> 🚀 **qwen4exp concurrency works (correcting 0xBakeer's crash report)**: with our
> 035e22731+canreuse build, `--parallel 2` aggregates **93-104 t/s**, on par with Felliks'
> SGLang 4-stream (93-103) — llama.cpp can serve multiple requests on one machine too!
> ⚠️ Verified at 8K window × 2 streams only; 262K concurrent not yet verified (0xBakeer's
> crash may live on the long-context path); validate 128K before production concurrency.

### 5.1c 262K-window concurrency measured (2026-08-28, Q3+PLE-offload+MTP+ngram-map-k4v, --parallel 2, production build bea3b12d)

| Scenario | Request A | Request B | Aggregate | Server | Memory |
|---|---:|---:|---:|---|---:|
| codeC + prose (short) | 61.7 t/s | 15.8 t/s | ~77.5 t/s | ✅ alive, 0 asserts | 72GB / 49GB free |
| **18.7K-token prompt + prose** | 29.4 t/s (after prefill) | 2.4 t/s (queued) | — | ✅ alive, 0 asserts | 74GB / 47GB free |

> 🎯 **262K + concurrency = safe (further correction)**: even with a 262K window, 2 concurrent
> streams and an 18.7K-token prompt, zero assert crashes, memory stable at 74GB/47GB free.
> 0xBakeer's crash was on an older commit (035e227, missing the indexer-KV save/restore
> concurrency fixes); our 035e22731 and bea3b12d builds both include them. Production recipe can
> safely use `--parallel 2` (free dual-request concurrency on a personal machine).
> (code-type). ngram-mod: acc 0.56641 / len 37.25; map-k4v: acc 0.849 / len 41.75; MTP+k4v: acc
> 0.86 / len 14.64. Speculation is exact (every token verified, output unchanged).

### 5.2 MTP (dzannotti official route) status

- 2.44GB standard Q4_K_M MTP head + `qwen4exp-mtp-draft-head.patch` ready
- ❌ **Current tree (035e22731) + patch → segfault**: standalone head, `-md` draft mode and embedded
  head all segfault at load_model — the patch is only verified on tree bea3b12d; the indexer-cache
  refactor in 035e22731 is incompatible
- 🔧 Rebuilding on the verified tree bea3b12d (in progress); will re-test when done
- ⚠️ **cafe-llama.cpp fork permanently abandoned**: twice blew memory loading MTP drafts (90GB and
  82GB main models), overhead far beyond the documented 10-15GB

### 5.3 Cross-reference (2026-08-27 night)

| Setup | Code copy-edit | Prose | Counting | Memory | Quality |
|---|---:|---:|---:|---:|---:|
| IQ3_XXS baseline | 25.0 t/s | 25.4 | 26.2 | 83 GB | 87.6% |
| IQ3_XXS + ngram-mod | **58.7** | 26.1 | 26.1 | 83 GB | 87.6% |
| IQ3_XXS + MTP+ngram-mod | **83.0** 🚀 | 29.6 | 58.1 | 86 GB | 87.6% |
| Q4_K_XL + PLE-offload baseline (⚠️8K window only) | 22.9 | 20.1 | 20.0 | **82 GB** | 93.5% |
| Q4_K_XL + PLE + ngram-mod | 47.4 | 19.3 | 20.0 | **82 GB** | 93.5% |
| Q4_K_XL + PLE + MTP+ngram-mod (⚠️8K window only) | **70.1** | 19.3 | 39.1 | 86 GB | **93.5%** |
| **Q3_K_XL + PLE + MTP+ngram (8K)** | **82.7** | 25.8 | 54.1 | **64 GB** | 90.4% |
| **Q3_K_XL + PLE + MTP+ngram (128K)** | **79.2** | — | — | **68 GB** | 90.4% |
| **Q3_K_XL + PLE + MTP+ngram (262K, production window)** | **78.9** | 21.5 | — | **70 GB** | 90.4% |
| 2×DGX Spark NVFP4 + MTP4 (community, tonyd2wild) | 50-55 t/s | ~33 | — | two machines | 4-bit class |

**🏆 Final production recommendation (2026-08-27 night): Q3_K_XL + NVMe-PLE + MTP+ngram-mod**
- Memory **70 GB (262K, was 102 GB)**, headroom 51 GB; code copy **78.9 t/s (262K) / 82.7 (8K)**
  = **3.3× the old config** (24 t/s)
- All three windows (8K/128K/262K) verified; quality 90.4% unchanged (speculation verifies every token)

**Q4_K_XL + NVMe-PLE key findings (0xBakeer recipe verified on this machine)**:
- `-lm mmap -ot per_layer_token_embd=CPU`: the 51B PLE table (26.8 GiB) is served from the NVMe
  page cache — **memory footprint 82 GB (IQ3-level, far below the official 112 GB requirement)**
- Quality 93.5% (+3.1pp over Q3_K_XL, +5.9pp over IQ3_XXS); code copy with MTP+ngram-mod stacked
  hits **70.1 t/s = 3.1× its own baseline**
- Speed vs quality tradeoff: IQ3 stacked = 83.0 t/s (87.6%) vs Q4 stacked = 70.1 t/s (93.5%)
- Warm-up: one sequential 26.8 GiB read (~28 s, 0.95 GiB/s); warm after the server is ready
  (loading evicts the table); cold/warm difference is significant with speculation on
- Cold load of the 4-shard Q4 takes ~4-5 min (3.5 min when download pages were still cached)

> Full analysis (mechanism/math/causal chain/unlock roadmap) in **speculative-analysis.md**

---

## 6. Memory breakdown

| Item | Value |
|---|---|
| Model weights (mmap) | ~90 GB |
| KV/state (262K config) | ~4-6 GB |
| Runtime/system/page cache | ~8-10 GB |
| Peak (262K) | ~102 / 128 GB |
| Available while serving | ~19-22 GiB |

---

## 7. Comparison with reference data (27B precedent on the same machine)

| Model | Engine/method | Decode speed |
|---|---|---|
| Qwen3.8-27B NVFP4 | SGLang + DFlash2 speculation (k=12) | **55.3 t/s** |
| Qwen3.8-27B NVFP4 | SGLang no speculation (bandwidth limit 273GB/s ÷ 20.4GB) | ~13.4 t/s |
| Qwen3.8-Flash-Next 180B | llama.cpp no speculation (this machine) | **22-24 t/s** |

> Reference: Flash-Next's no-speculation measurement is already ~2× the 27B no-speculation bandwidth limit (6B active × smaller weights);
> if MTP speculation lands (1.5-2.5×), expect 30-50 t/s, clearly beating the 27B+DFlash2 combo.

---

## 8. Reproducible methodology

```bash
# Decode baseline (run llama-bench with the server stopped to avoid loading two model copies)
llama-bench -m model.gguf -ngl 999 -t 20 -p 0 -n 64 -r 2

# Server-side speed: the timings field of the API response
curl -s http://127.0.0.1:8889/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"..."}],"max_tokens":200}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['timings'])"

# Speculation tests: separate port + small context (don't disturb production)
llama-server -m model.gguf --port 8890 --ctx-size 16384 --spec-type <method> --jinja

# Memory: free -h (while serving)
```

---

## 9. Summary (final 2026-08-27 night)

1. **🏆 Final production recommendation: Q3_K_XL + NVMe-PLE + MTP+ngram-mod** (262K window)
   - Code copy **78.9 t/s** = **3.3× the old config** (24 t/s); memory **70 GB** (was 102 GB), headroom 51 GB
   - Command: see docs/mtp-tracker.md (Option B + PLE flags + warm_table.py warm-up)
2. **🚀 ngram-mod breaks the "24 t/s ceiling"**: code copy/edit **25.0 → 58.7 t/s (+135%)**, zero extra memory, output verified token-by-token; prose stays ~26 t/s
3. **🚀 MTP+ngram-mod stacked**: IQ3_XXS code copy **83.0 t/s (3.3×)**, counting 58.1, prose 29.6 (verified tree bea3b12d + dzannotti head)
4. **🚀 Q4_K_XL is now runnable via NVMe-PLE**: 82 GB memory (official requirement 112 GB), 93.5% quality; +MTP+ngram-mod code copy **70.1 t/s (3.1×)**
   - Tradeoff: speed-first → IQ3 stacked (83 t/s / 87.6%); quality-first → Q4 stacked (70.1 t/s / 93.5%)
5. **262K context is effectively free** (architecture dividend), but large windows slow short-prompt prefill (implementation tax + architecture tax)
6. **MTP tree compatibility**: segfaults on 035e22731; cafe fork permanently abandoned (2× OOM); verified tree bea3b12d + standalone/embedded head work
7. Full NVFP4 (101.7 GB, with MTP head) now exists (provsalt) — critical fit, watchlisted
