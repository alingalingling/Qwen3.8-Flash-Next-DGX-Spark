#!/usr/bin/env bash
# Qwen3.8-Flash-Next UD-Q3_K_XL 启动脚本(DGX Spark)
# 用法: ./run_qwen38_q3.sh [start|stop|status|logs]
set -euo pipefail

MODEL_DIR="$HOME/models/Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL"
GGUF="$MODEL_DIR/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf"
SERVER="$HOME/llama.cpp/build/bin/llama-server"
PORT=8889
CTX=262144
LOG="$HOME/models/Qwen3.8-Flash-Next-GGUF/llama-server.log"
PIDFILE="$HOME/models/Qwen3.8-Flash-Next-GGUF/llama-server.pid"

start() {
  [ -f "$GGUF" ] || { echo "模型不存在: $GGUF (先完成下载)"; exit 1; }
  [ -x "$SERVER" ] || { echo "llama-server 不存在: $SERVER (先构建)"; exit 1; }
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "已在运行 (pid $(cat "$PIDFILE"))"; exit 0
  fi
  nohup "$SERVER" \
    -m "$GGUF" \
    --host 0.0.0.0 --port $PORT \
    --ctx-size $CTX \
    -ngl 999 -t 20 \
    --temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 \
    --repeat-penalty 1.0 --presence-penalty 1.5 \
    --jinja \
    > "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  echo "已启动 pid $(cat "$PIDFILE"),端口 $PORT,日志 $LOG"
  echo "等待就绪..."
  for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
      echo "就绪 ✅ $(curl -s http://127.0.0.1:$PORT/health)"; return 0
    fi
    sleep 2
  done
  echo "超时未就绪,看日志: tail -50 $LOG"; return 1
}

stop() {
  [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null && rm -f "$PIDFILE" && echo "已停止" || echo "未在运行"
}

status() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "运行中 (pid $(cat "$PIDFILE"))"; curl -s http://127.0.0.1:$PORT/health || true
  else
    echo "未运行"
  fi
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  logs) tail -50 "$LOG" ;;
  *) echo "用法: $0 [start|stop|status|logs]"; exit 1 ;;
esac
