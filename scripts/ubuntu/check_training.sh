#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: 远程训练任务实时探查与自动记录。
#
# 功能子命令:
#   status            单次进度快照（默认行为，可直接 bash check_training.sh）
#   watch [秒]        本地循环刷新快照，默认间隔 900 秒（15 分钟）
#   follow            实时跟踪训练日志（仅显示 PPO 进度行与错误行）
#   history [条数]    查看远端自动记录器的历史快照，默认最近 4 条
#   install-recorder  在远程宿主机安装/重启 15 分钟自动记录器
#   stop-recorder     停止远端自动记录器
#
# 依赖: 本机 ~/.ssh/config 中已配置别名 go2-server（免密）。
# 可用环境变量覆盖默认值:
#   GO2_REMOTE    SSH 别名,    默认 go2-server
#   GO2_CONTAINER 容器名,      默认 isaac-sim-kevin
#
# 使用示例:
#   bash scripts/ubuntu/check_training.sh status
#   bash scripts/ubuntu/check_training.sh watch 300
#   bash scripts/ubuntu/check_training.sh history 8
#   bash scripts/ubuntu/check_training.sh install-recorder

set -euo pipefail

REMOTE="${GO2_REMOTE:-go2-server}"
CT="${GO2_CONTAINER:-isaac-sim-kevin}"
CMD="${1:-status}"
ARG="${2:-}"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# ---------------------------------------------------------------------------
# 远端快照脚本（在宿主机执行，通过 docker exec 读取容器内状态）
# ---------------------------------------------------------------------------
cat > "${WORKDIR}/snapshot.sh" <<'SNAP'
CT="${CT:-isaac-sim-kevin}"
LOG=$(docker exec "$CT" bash -c 'ls -t /workspace/go2_rl/logs/train_task1_*.log 2>/dev/null | head -1')
PID=$(docker exec "$CT" bash -c 'pgrep -f "[t]ask1_train.py" | head -1')
RUN=$(docker exec "$CT" bash -c 'ls -td /workspace/go2_rl/logs/task1/go2_task1_skrl_ppo_* 2>/dev/null | head -1')
echo "snapshot_time : $(date "+%F %T")"
echo "log_file      : ${LOG:-none}"
if [ -n "$PID" ]; then
    EL=$(docker exec "$CT" ps -o etime= -p "$PID" 2>/dev/null | tr -d " ")
    echo "train_process : ALIVE pid=$PID elapsed=${EL:-?}"
else
    echo "train_process : NOT_RUNNING"
fi
if [ -n "$LOG" ]; then
    LAST=$(docker exec "$CT" bash -c "tr \"\r\" \"\n\" < \"$LOG\" | grep \"skrl PPO:\" | tail -1")
    if [ -n "$LAST" ]; then
        STEPS=$(printf "%s" "$LAST" | sed -n "s#.*| \([0-9,]*\)/350000000.*#\1#p" | tr -d ",")
        PCT=$(printf "%s" "$LAST" | sed -n "s#.*PPO: *\([0-9]*\)%.*#\1#p")
        SPS=$(printf "%s" "$LAST" | sed -n "s#.*, \([0-9.]*\)steps/s.*#\1#p")
        REW=$(printf "%s" "$LAST" | sed -n "s#.*rew=\([-0-9.]*\).*#\1#p")
        FALL=$(printf "%s" "$LAST" | sed -n "s#.*fall=\([0-9.]*\).*#\1#p")
        echo "progress      : ${STEPS:-?}/350000000 (${PCT:-?}%)  ${SPS:-?} steps/s"
        echo "metrics       : rew=${REW:-?} fall=${FALL:-?}"
        case "${REW:-}" in
            *nan*|*inf*) echo "ANOMALY       : rew=${REW} (NaN/Inf detected!)";;
        esac
    else
        echo "progress      : NO_PPO_LINES_YET (initializing or crashed, check log_file)"
    fi
    ERRS=$(docker exec "$CT" bash -c "grep -cE \"Traceback|RuntimeError|CUDA error|out of memory\" \"$LOG\"" 2>/dev/null || true)
    echo "errors_in_log : ${ERRS:-0}"
fi
docker exec "$CT" nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed "s/^/gpu           : /"
echo "run_dir       : ${RUN:-none}"
if [ -n "$RUN" ]; then
    ITEMS=$(docker exec "$CT" bash -c "ls \"$RUN\" 2>/dev/null | grep -v tfevents | tr \"\n\" \" \"" 2>/dev/null || true)
    echo "run_dir_items : ${ITEMS:-none}"
    FC=$(docker exec "$CT" bash -c "ls \"$RUN/final_checkpoint/go2_task1_model.pt\" 2>/dev/null" || true)
    if [ -n "$FC" ]; then
        echo "final_ckpt    : $FC"
        [ -z "$PID" ] && echo "STATUS        : TRAINING_COMPLETED"
    fi
fi
SNAP

# ---------------------------------------------------------------------------
# 远端记录器循环脚本
# ---------------------------------------------------------------------------
cat > "${WORKDIR}/recorder.sh" <<'REC'
#!/bin/bash
HIST=/root/go2_train1_history.log
while true; do
    echo "" >> "$HIST"
    bash /root/.go2_snapshot.sh >> "$HIST" 2>&1
    sleep 900
done
REC

run_snapshot() {
    { echo "CT=${CT}"; cat "${WORKDIR}/snapshot.sh"; } | ssh "$REMOTE" 'bash -s'
}

case "$CMD" in
    status)
        run_snapshot
        ;;
    watch)
        INTERVAL="${ARG:-900}"
        while true; do
            clear
            date
            run_snapshot || true
            sleep "$INTERVAL"
        done
        ;;
    follow)
        ssh "$REMOTE" "docker exec ${CT} bash -c 'tail -f \$(ls -t /workspace/go2_rl/logs/train_task1_*.log | head -1)' | tr '\r' '\n' | grep --line-buffered -E 'skrl PPO:|Traceback|Error|error|完成|saved|saved'"
        ;;
    history)
        N="${ARG:-4}"
        ssh "$REMOTE" "tail -n \$(( ${N} * 16 )) /root/go2_train1_history.log 2>/dev/null || echo 'no history yet (install-recorder first)'"
        ;;
    install-recorder)
        scp -q "${WORKDIR}/snapshot.sh" "${REMOTE}:/root/.go2_snapshot.sh"
        scp -q "${WORKDIR}/recorder.sh" "${REMOTE}:/root/go2_recorder.sh"
        ssh "$REMOTE" 'bash -s' <<RINST
if [ -f /root/go2_recorder.pid ]; then
    kill "\$(cat /root/go2_recorder.pid)" 2>/dev/null || true
fi
nohup bash /root/go2_recorder.sh >/dev/null 2>&1 &
echo \$! > /root/go2_recorder.pid
sleep 1
kill -0 "\$(cat /root/go2_recorder.pid)" 2>/dev/null && echo "[OK] recorder started, pid=\$(cat /root/go2_recorder.pid), history=/root/go2_train1_history.log" || echo "[FAIL] recorder not running"
RINST
        ;;
    stop-recorder)
        ssh "$REMOTE" 'if [ -f /root/go2_recorder.pid ]; then kill "$(cat /root/go2_recorder.pid)" 2>/dev/null && echo "[OK] recorder stopped" || echo "[WARN] pid not alive"; rm -f /root/go2_recorder.pid; else echo "[WARN] no pid file"; fi'
        ;;
    *)
        echo "unknown command: $CMD"
        echo "usage: bash scripts/ubuntu/check_training.sh {status|watch [sec]|follow|history [n]|install-recorder|stop-recorder}"
        exit 1
        ;;
esac
