#!/usr/bin/env bash
# Qwen3.8-Flash-Next 生产配方启动脚本(DGX Spark,2026-08-28 终版)
# 配方:UD-Q3_K_XL + NVMe-PLE 卸载 + MTP 头 + ngram-map-k4v 组合投机 + parallel 2
# 用法: ./run_qwen38_q3.sh [start|stop|status|logs]
set -euo pipefail

MODEL_DIR="$HOME/models/Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL"
GGUF="$MODEL_DIR/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf"
MTP_HEAD="$HOME/models/mtp-draft/dzannotti/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf"
# MTP 路线构建(bea3b12d + 补丁);ngram-only 可用 $HOME/llama.cpp/build/bin/llama-server + --spec-type ngram-map-k4v
SERVER="$HOME/llama-mtp-verified/build/bin/llama-server"
PORT=8889
CTX=262144
LOG="$HOME/models/Qwen3.8-Flash-Next-GGUF/llama-server.log"
PIDFILE="$HOME/models/Qwen3.8-Flash-Next-GGUF/llama-server.pid"

start() {
  [ -f "$GGUF" ] || { echo "模型不存在: $GGUF (先完成下载)"; exit 1; }
  [ -f "$MTP_HEAD" ] || { echo "MTP 头不存在: $MTP_HEAD"; exit 1; }
  [ -x "$SERVER" ] || { echo "llama-server 不存在: $SERVER (先构建验证树)"; exit 1; }
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "已在运行 (pid $(cat "$PIDFILE"))"; exit 0
  fi
  # 内存铁律:启动前确认无其他 llama-server 且余量充足
  local n=$(ps aux | grep "[l]lama-server" | wc -l)
  [ "$n" -gt 1 ] && { echo "已有其他 llama-server 在运行,拒绝启动"; exit 1; }
  local avail=$(free -g | awk 'NR==2{print $7+0}')
  [ "$avail" -lt 45 ] && { echo "可用内存 ${avail}GB < 45GB,拒绝启动"; exit 1; }
  setsid nohup env LLAMA_ATTN_ROT_DISABLE=1 "$SERVER" \
    -m "$GGUF" \
    -lm mmap -ot per_layer_token_embd=CPU \
    -md "$MTP_HEAD" -ngld 999 \
    --spec-type draft-mtp,ngram-map-k4v --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
    --host 0.0.0.0 --port $PORT \
    --ctx-size $CTX --parallel 2 \
    -ngl 999 -t 20 \
    -fa on -ctk q8_0 -ctv q8_0 \
    --temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 \
    --repeat-penalty 1.0 --presence-penalty 1.5 \
    --jinja \
    > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  echo "已启动 pid $(cat "$PIDFILE"),端口 $PORT,日志 $LOG"
  echo "就绪后建议预热 PLE 表: GGUF_PY=~/llama.cpp/gguf-py ~/ai-env/bin/python3 ~/smalltask/hf_monitor/warm_table.py $GGUF"
  echo "等待就绪..."
  for i in $(seq 1 150); do
    if curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
      echo "就绪 ✅ $(curl -s http://127.0.0.1:$PORT/health)"; return 0
    fi
    sleep 2
  done
  echo "超时未就绪,看日志: tail -50 $LOG"; return 1
}

stop() {
  if [ -f "$PIDFILE" ]; then
    kill -9 "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
  # PID 文件可能失效(手动 setsid 启动时),用 ps 找真实 PID
  pkill -9 -f "[l]lama-mtp-verified/build/bin/llama-server" 2>/dev/null || true
  pkill -9 -f "[l]lama.cpp/build/bin/llama-server" 2>/dev/null || true
  echo "已停止"
}

status() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "运行中 (pid $(cat "$PIDFILE"))"
    curl -sf http://127.0.0.1:$PORT/health && echo " 健康 ✅"
  else
    echo "未运行"
  fi
  free -g | head -2
}

logs() { tail -50 "$LOG"; }

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  logs) logs ;;
  *) echo "用法: $0 [start|stop|status|logs]"; exit 1 ;;
esac
