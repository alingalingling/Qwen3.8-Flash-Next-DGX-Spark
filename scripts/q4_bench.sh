#!/usr/bin/env bash
# Q4_K_XL + NVMe-PLE 方案测速(0xBakeer 配方,安全版)
# 用法: bash q4_bench.sh [warm|nowarm]   (默认 warm: 预热 PLE 表再测)
set -u
WATCHDOG="/home/lingyan/smalltask/hf_monitor/watchdog.sh"
SERVER="/home/lingyan/llama.cpp/build/bin/llama-server"
Q4_MODEL="/home/lingyan/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
PORT=8898
CTX=16384
RESULT="/tmp/q4_bench_results.txt"
WARM="${1:-warm}"
PLE_FLAGS="-lm mmap -ot per_layer_token_embd=CPU"

log() { echo "$(date '+%F %T') $*" | tee -a "$RESULT"; }

precheck() {
  local n=$(ps aux | grep "[l]lama-server" | wc -l)
  if [ "$n" -gt 0 ]; then log "❌ 已有 $n 个 llama-server 在运行,中止"; exit 1; fi
  local avail=$(free -g | awk 'NR==2{print $7+0}')
  if [ "$avail" -lt 45 ]; then log "❌ 可用内存 ${avail}GB < 45GB,中止"; exit 1; fi
  log "✅ 预检通过: 无运行中服务, 可用 ${avail}GB"
}

run_test() {
  local label="$1"; shift
  local avail=$(free -g | awk 'NR==2{print $7+0}')
  if [ "$avail" -lt 40 ]; then log "❌ 组间检查: 可用 ${avail}GB < 40GB,跳过"; return 1; fi
  log ""
  log "========== 测试: $label =========="
  setsid nohup "$SERVER" -m "$Q4_MODEL" --host 0.0.0.0 --port $PORT --ctx-size $CTX -ngl 999 -t 20 --jinja "$@" > /tmp/q4_server.log 2>&1 < /dev/null &
  local spid=$!
  "$WATCHDOG" "$spid" 15 5 & local wdog=$!
  log "服务 PID=$spid, 看门狗 15GB/5s, 等待就绪..."
  local ready=0
  for i in $(seq 1 300); do
    if curl -sf --max-time 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$spid" 2>/dev/null || { log "❌ 服务进程已死"; break; }
    local nowavail=$(free -g | awk 'NR==2{print $7+0}')
    if [ "$nowavail" -lt 15 ]; then log "🚨 加载期内存守卫: 可用 ${nowavail}GB → kill"; kill -9 "$spid"; sleep 3; break; fi
    sleep 2
  done
  if [ "$ready" != "1" ]; then
    log "❌ 未就绪(300次轮询)"; tail -5 /tmp/q4_server.log >> "$RESULT"
    kill "$wdog" 2>/dev/null; kill -9 "$spid" 2>/dev/null; sleep 3
    return 1
  fi
  log "✅ 就绪, 内存: $(free -g | awk 'NR==2{print "已用"$3"G 可用"$7"G"}')"
  if [ "$WARM" = "warm" ]; then
    log "--- 服务就绪后预热 PLE 表(避免被加载挤掉) ---"
    GGUF_PY=/home/lingyan/llama.cpp/gguf-py /home/lingyan/ai-env/bin/python3 /home/lingyan/smalltask/hf_monitor/warm_table.py "$Q4_MODEL" | tee -a "$RESULT"
  fi
  log "--- A: 数数 200 tokens ---"
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"输出:1,2,3,4,5,6,7,8,9,10, 然后继续数到200,逗号分隔"}],"max_tokens":200,"chat_template_kwargs":{"enable_thinking":false},"temperature":0}' \
    | python3 -c "
import json,sys
d=json.load(sys.stdin); t=d['timings']
print('   prompt: %.1f tok/s | decode: %.1f tok/s'%(t['prompt_per_second'], 1000/t['predicted_per_token_ms']))
" | tee -a "$RESULT"
  log "--- B: 散文 150 tokens ---"
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"写一段关于秋天的散文"}],"max_tokens":150,"chat_template_kwargs":{"enable_thinking":false},"temperature":0.7}' \
    | python3 -c "
import json,sys
d=json.load(sys.stdin); t=d['timings']
print('   prompt: %.1f tok/s | decode: %.1f tok/s'%(t['prompt_per_second'], 1000/t['predicted_per_token_ms']))
" | tee -a "$RESULT"
  log "--- C: 代码复制修改 500 tokens ---"
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"请把下面 C 程序中的 `result = a + b` 改成 `result = a * b + c`,然后完整输出整个修改后的程序(不要省略任何行):\n\n#include <stdio.h>\n#include <stdlib.h>\n\nstatic int compute(int a, int b, int c) {\n    int result = a + b;\n    if (result > 100) {\n        result = result - c;\n    }\n    for (int i = 0; i < c; i++) {\n        result += i;\n    }\n    return result;\n}\n\nint main(void) {\n    int vals[8] = {3, 5, 7, 11, 13, 17, 19, 23};\n    int total = 0;\n    for (int i = 0; i < 8; i++) {\n        total = compute(total, vals[i], i + 1);\n    }\n    printf(\"total = %d\\n\", total);\n    return 0;\n}\n"}],"max_tokens":500,"chat_template_kwargs":{"enable_thinking":false},"temperature":0}' \
    | python3 -c "
import json,sys
d=json.load(sys.stdin); t=d['timings']
print('   prompt: %.1f tok/s | decode: %.1f tok/s'%(t['prompt_per_second'], 1000/t['predicted_per_token_ms']))
" | tee -a "$RESULT"
  grep -iE "accept" /tmp/q4_server.log | tail -2 >> "$RESULT" 2>/dev/null || true
  kill "$wdog" 2>/dev/null; kill -9 "$spid" 2>/dev/null; sleep 5
  log "✅ 完成, 剩余内存: $(free -g | awk 'NR==2{print $7"GB"}')"
}

echo "" > "$RESULT"
precheck || exit 1

# 测试1: Q4_K_XL + NVMe-PLE + ngram-mod(0xBakeer 配方)
run_test "Q4_K_XL + PLE-offload + ngram-mod" $PLE_FLAGS --spec-type ngram-mod --parallel 1

# 测试2: Q4_K_XL + NVMe-PLE 无投机(对照)
run_test "Q4_K_XL + PLE-offload 基线(无投机)" $PLE_FLAGS --parallel 1

log ""
log "========== 全部完成 =========="
cat "$RESULT"
