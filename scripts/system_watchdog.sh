#!/usr/bin/env bash
# 系统级内存看门狗(重启后自动生效版)——监控任何 llama-server 进程
# 逻辑: 周期检查可用内存,低于阈值 → 杀掉所有 llama-server + 记录日志
# 用法: system_watchdog.sh [阈值GB,默认5] [间隔秒,默认5]
set -u
THRESH="${1:-5}"
INTERVAL="${2:-5}"
LOG="/home/lingyan/smalltask/hf_monitor/watchdog_system.log"

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "系统看门狗启动: 内存阈值=${THRESH}GB, 间隔=${INTERVAL}s"
while true; do
    AVAIL=$(free -g | awk 'NR==2{print $7+0}')
    # 有 llama-server 运行时才严格检查
    N=$(ps aux | grep "[l]lama-server" | wc -l)
    if [ "$N" -gt 0 ] && [ "$AVAIL" -lt "$THRESH" ]; then
        log "🚨 内存告警: 可用 ${AVAIL}GB < ${THRESH}GB, 有 $N 个 llama-server → 全部 kill"
        pkill -9 -f "llama-server" 2>/dev/null
        pkill -9 -f "cafe-llama" 2>/dev/null
        sleep 3
        log "已执行清理, 当前可用: $(free -g | awk 'NR==2{print $7"GB"}')"
    fi
    sleep "$INTERVAL"
done
