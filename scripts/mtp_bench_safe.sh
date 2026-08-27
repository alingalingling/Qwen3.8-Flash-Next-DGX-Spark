#!/usr/bin/env bash
# MTP/ngram-mod 全方位测速脚本 v2(安全版)
# 变更: ① 删除 cafe fork 组(已两次爆内存死机,永久放弃)
#       ② 新增 ngram-mod 组(零额外内存,0xBakeer 方案)
#       ③ 预检余量≥45GB、组间≥40GB、看门狗15GB/5s、加载期2s轮询内存(可用<15GB立即kill)
# 用法: bash mtp_bench_safe.sh [组名]   (缺省跑全部; 组名: baseline|ngram|mtp|q3ref)
set -u
WATCHDOG="/home/lingyan/smalltask/hf_monitor/watchdog.sh"
STD_SERVER="/home/lingyan/llama.cpp/build/bin/llama-server"
Q3_MODEL="/home/lingyan/models/Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf"
IQ3_MODEL="/home/lingyan/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"
DZ_MTP_DRAFT="/home/lingyan/models/mtp-draft/dzannotti/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf"
PORT=8890
CTX=16384
RESULT="/tmp/mtp_bench_safe_results.txt"
ONLY_GROUP="${1:-all}"
ENV_VARS=""

log() { echo "$(date '+%F %T') $*" | tee -a "$RESULT"; }

precheck() {
  local n=$(ps aux | grep "[l]lama-server" | wc -l)
  if [ "$n" -gt 0 ]; then log "❌ 已有 $n 个 llama-server 在运行,中止"; exit 1; fi
  local avail=$(free -g | awk 'NR==2{print $7+0}')
  if [ "$avail" -lt 45 ]; then log "❌ 可用内存 ${avail}GB < 45GB,中止"; exit 1; fi
  log "✅ 预检通过: 无运行中服务, 可用 ${avail}GB"
}

run_test() {
  # run_test <标签> <组名> <服务器> <模型> [额外参数...]
  local label="$1"; shift
  local gname="$1"; shift
  local server="$1"; shift
  local model="$1"; shift
  [ "$ONLY_GROUP" != "all" ] && [ "$ONLY_GROUP" != "$gname" ] && return 0
  local avail=$(free -g | awk 'NR==2{print $7+0}')
  if [ "$avail" -lt 40 ]; then log "❌ 组间检查: 可用 ${avail}GB < 40GB,跳过后续测试"; return 1; fi
  log ""
  log "========== 测试[$gname]: $label =========="
  setsid nohup env $ENV_VARS "$server" -m "$model" --host 0.0.0.0 --port $PORT --ctx-size $CTX -ngl 999 -t 20 --jinja "$@" > /tmp/bench_safe_server.log 2>&1 < /dev/null &
  local spid=$!
  "$WATCHDOG" "$spid" 15 5 & local wdog=$!
  log "服务 PID=$spid, 看门狗 PID=$wdog(15GB/5s), 等待就绪(加载期2s轮询内存)..."
  local ready=0
  for i in $(seq 1 120); do
    if curl -sf --max-time 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$spid" 2>/dev/null || { log "❌ 服务进程已死(可能被看门狗或OOM)"; break; }
    # 加载期内存守卫: 可用 < 15GB 立即杀(比看门狗更早介入)
    local nowavail=$(free -g | awk 'NR==2{print $7+0}')
    if [ "$nowavail" -lt 15 ]; then
      log "🚨 加载期内存守卫: 可用 ${nowavail}GB < 15GB → 立即 kill"
      kill -9 "$spid" 2>/dev/null
      sleep 3
      break
    fi
    sleep 2
  done
  if [ "$ready" != "1" ]; then
    log "❌ 未就绪(120次轮询内)"; tail -5 /tmp/bench_safe_server.log >> "$RESULT"
    kill "$wdog" 2>/dev/null; kill -9 "$spid" 2>/dev/null; sleep 3
    return 1
  fi
  log "✅ 就绪, 内存: $(free -g | awk 'NR==2{print "已用"$3"G 可用"$7"G"}')"
  # 测试A: 数数 200 tokens
  log "--- A: 数数 200 tokens ---"
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"输出:1,2,3,4,5,6,7,8,9,10, 然后继续数到200,逗号分隔"}],"max_tokens":200,"chat_template_kwargs":{"enable_thinking":false},"temperature":0}' \
    | python3 -c "
import json,sys
d=json.load(sys.stdin)
t=d['timings']
print('   prompt: %.1f tok/s | decode: %.1f tok/s'%(t['prompt_per_second'], 1000/t['predicted_per_token_ms']))
" | tee -a "$RESULT"
  # 测试B: 散文 150 tokens
  log "--- B: 散文 150 tokens ---"
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"写一段关于秋天的散文"}],"max_tokens":150,"chat_template_kwargs":{"enable_thinking":false},"temperature":0.7}' \
    | python3 -c "
import json,sys
d=json.load(sys.stdin)
t=d['timings']
print('   prompt: %.1f tok/s | decode: %.1f tok/s'%(t['prompt_per_second'], 1000/t['predicted_per_token_ms']))
" | tee -a "$RESULT"
  # 测试C: 代码复制修改(copy-heavy, ngram-mod 主战场)
  log "--- C: 代码复制修改 500 tokens ---"
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"请把下面 C 程序中的 `result = a + b` 改成 `result = a * b + c`,然后完整输出整个修改后的程序(不要省略任何行):\n\n#include <stdio.h>\n#include <stdlib.h>\n\nstatic int compute(int a, int b, int c) {\n    int result = a + b;\n    if (result > 100) {\n        result = result - c;\n    }\n    for (int i = 0; i < c; i++) {\n        result += i;\n    }\n    return result;\n}\n\nint main(void) {\n    int vals[8] = {3, 5, 7, 11, 13, 17, 19, 23};\n    int total = 0;\n    for (int i = 0; i < 8; i++) {\n        total = compute(total, vals[i], i + 1);\n    }\n    printf(\"total = %d\\n\", total);\n    return 0;\n}\n"}],"max_tokens":500,"chat_template_kwargs":{"enable_thinking":false},"temperature":0}' \
    | python3 -c "
import json,sys
d=json.load(sys.stdin)
t=d['timings']
print('   prompt: %.1f tok/s | decode: %.1f tok/s'%(t['prompt_per_second'], 1000/t['predicted_per_token_ms']))
" | tee -a "$RESULT"
  # 投机统计
  grep -iE "accept|draft" /tmp/bench_safe_server.log | tail -3 >> "$RESULT" 2>/dev/null || true
  # 收尾
  kill "$wdog" 2>/dev/null; kill -9 "$spid" 2>/dev/null; sleep 5
  log "✅ [$gname] 测试完成, 服务已停, 剩余内存: $(free -g | awk 'NR==2{print $7"GB"}')"
}

echo "" > "$RESULT"
precheck || exit 1

# 组1: IQ3_XXS 基线(标准构建,无投机)
run_test "IQ3_XXS 基线(无投机)" baseline "$STD_SERVER" "$IQ3_MODEL"

# 组2: IQ3_XXS + ngram-mod(0xBakeer 方案,零额外内存)
run_test "IQ3_XXS + ngram-mod" ngram "$STD_SERVER" "$IQ3_MODEL" --spec-type ngram-mod

# 组3: IQ3_XXS + dzannotti MTP 官方路线(补丁构建,dzannotti 推荐参数)
ENV_VARS="LLAMA_ATTN_ROT_DISABLE=1"
run_test "IQ3_XXS + MTP官方路线(dzannotti Q4_K_M)" mtp "$STD_SERVER" "$IQ3_MODEL" \
  -md "$DZ_MTP_DRAFT" -ngld 999 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
  -fa on -ctk q8_0 -ctv q8_0
ENV_VARS=""

# 组4: Q3_K_XL 基线(参照)
run_test "Q3_K_XL 基线(参照)" q3ref "$STD_SERVER" "$Q3_MODEL"

log ""
log "========== 全部测试完成 =========="
