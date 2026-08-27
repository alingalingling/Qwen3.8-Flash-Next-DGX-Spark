# Deployment Playbook — Qwen3.8-Flash-Next on DGX Spark (2026-08-28: executed)

**[English](deploy_playbook.md) | [中文](deploy_playbook.zh.md)**

> Target machine: NVIDIA DGX Spark (GB10), 128 GB unified memory
> Status: **deployed and fully accelerated** — production recipe = Q3_K_XL + NVMe-PLE +
> MTP+ngram-map-k4v, 262K context, 78.9 t/s single / 77.5 aggregate (2 concurrent), 70 GB memory
> The 2026-08-26 edition was a contingency plan ("awaiting target version"); everything has
> since landed — this document is now the executed playbook.

## 0. Fit-check (measured)

| Item | Measured |
|---|---|
| Disk (three quants) | Q3 90GB / IQ3 82GB / Q4 111GB (~283GB total; 3.0TB free) |
| Runtime peak | Q3 recipe @262K = **70GB** ≤ 115GB ✅ (old config: 102GB) |
| No load-time BF16 re-inflation | ✅ NVMe-PLE: `-lm mmap -ot per_layer_token_embd=CPU`, PLE not resident |
| Quality levels | Q3 90.4% / IQ3 87.6% / Q4 93.5% (unsloth Divergence-300) |

## 1. Three runtime paths (by priority, all measured on this machine)

### Path A: Q3_K_XL production recipe (preferred, 262K, 78.9 t/s, 70GB) 🏆

```bash
# Engine: ~/llama-mtp-verified/build (bea3b12d + qwen4exp-mtp-draft-head.patch)
LLAMA_ATTN_ROT_DISABLE=1 ~/llama-mtp-verified/build/bin/llama-server \
  -m ~/models/Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf \
  -lm mmap -ot per_layer_token_embd=CPU \
  -md ~/models/mtp-draft/dzannotti/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf -ngld 999 \
  --spec-type draft-mtp,ngram-map-k4v --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0 \
  --ctx-size 262144 --parallel 2 --host 0.0.0.0 --port 8889 \
  --temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 --presence-penalty 1.5 --jinja
# After ready, warm the PLE table (one ~28s pass):
# GGUF_PY=~/llama.cpp/gguf-py ~/ai-env/bin/python3 ~/smalltask/hf_monitor/warm_table.py <model shard 1>
```

Measured: code copy 78.9 t/s (262K) / 82.7 (8K); prose 21.5-25.8; memory 70GB (51GB headroom).

### Path B: speed-first (IQ3_XXS, 83.0 t/s at 8K)

- Same engine/flags, model = `UD-IQ3_XXS-00001-of-00003.gguf`; or the embedded-head build
  `UD-IQ3_XXS-MTP/` (no `-md` needed; made with merge-mtp-shard.py)
- Quality 87.6%; window verified up to 16K

### Path C: quality-first (Q4_K_XL, 70.1 t/s at 8K, 93.5%)

- ⚠️ **8K short window only** (incident 5: loading at 262K froze the machine); use Path A for large windows
- Same engine/flags, model = `UD-Q4_K_XL-00001-of-00004.gguf`

### Simple option (zero dependencies, ngram-map-k4v)

```bash
~/llama.cpp/build/bin/llama-server -m <any shard 1> -ngl 999 -t 20 \
  --spec-type ngram-map-k4v --jinja --ctx-size 262144 --port 8889
```

## 2. Build steps (executed; kept for reference)

```bash
# MTP route (verified tree bea3b12d + patch) — ⚠️ patch segfaults on tree 035e22731
git clone ~/llama.cpp ~/llama-mtp-verified
cd ~/llama-mtp-verified && git checkout bea3b12da
git apply ~/smalltask/hf_monitor/dzannotti-mtp/qwen4exp-mtp-draft-head.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 20

# ngram route (035e22731 + canreuse patch)
cd ~/llama.cpp && git apply ~/smalltask/hf_monitor/canreuse-qwen4exp.patch
cmake --build build -j 20
```

> Note: the canreuse patch depends on 035e22731's `llama_memory_hybrid_idx_context`; it does
> NOT compile on bea3b12d (reverted there).

## 3. MTP-head injection (no model re-download)

```bash
# Embedded head (attach the head to any split GGUF; hard links save disk)
GGUF_PY=~/llama.cpp/gguf-py ~/ai-env/bin/python3 \
  ~/projects/qwen38-flash-next-dgx-spark/patches/merge-mtp-shard.py \
  <target model shard 1> <MTP head gguf> <output dir>
# Then run with --spec-type draft-mtp and no -md; within ~2% of the -md form
```

## 4. Memory-safety iron rules (hard-won lessons)

1. Only **one model** resident at a time (a 2.44GB MTP head is allowed)
2. `free -h` before tests; **≥45GB available** required to start
3. First load of any new combo: `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`
4. **Escalate windows stepwise: 8K→32K→64K→128K→262K, never skip** (incident 5)
5. Watchdogs: system-level `system_watchdog.sh 5 5` (crontab @reboot) + per-test `watchdog.sh <PID> 15 5`
6. Test instances: separate port + setsid; `[x]` bracket for pkill -f self-kill protection
7. Models survive reboots intact (verified 3×); one-command restore

## 5. Quality red lines

- 1-bit (IQ1 series): excluded (broken tool calls + quality collapse)
- 2-bit: acceptable in emergencies
- **3-bit/4-bit: recommended range** (Q3 90.4% / IQ3 87.6% / Q4 93.5% all measured runnable)
- 5-bit/6-bit/Q8_0: over budget even with PLE-offload (≥117GB), excluded
- Full NVFP4: runtime unpublished, not usable yet (tracking)

## 6. Version history

- 2026-08-26: contingency plan (Q3 not downloaded, PR not merged, speculation locked)
- 2026-08-27: execution — parallel downloader, builds, 262K serving, crash lessons, speculation unlock
- 2026-08-28: final — production recipe + NVMe-PLE + stacked speculation + window iron rules
