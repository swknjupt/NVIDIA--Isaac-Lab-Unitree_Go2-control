#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: 本地仓库 -> 远程宿主机 -> isaac-sim 容器 代码同步管道。
#
# 用途:
#   将本地工作区代码同步到远程服务器 (192.168.14.98) 的宿主机目录,
#   并经由 tar 管道注入 isaac-sim-kevin 容器内的运行目录。
#   容器内 configs/local_paths.yaml 与 logs/ 会被保护, 不会被覆盖或删除。
#
# 使用方式:
#   bash scripts/ubuntu/sync_to_remote.sh
#
# 可用环境变量覆盖默认值:
#   GO2_REMOTE    SSH 别名 (需在 ~/.ssh/config 中配置), 默认 go2-server
#   GO2_CONTAINER 目标容器名,                默认 isaac-sim-kevin
#   GO2_HOST_DIR  宿主机代码目录,            默认 /home/unitree_go2_isaaclab_rl
#   GO2_CT_DIR    容器内代码目录,            默认 /workspace/go2_rl

set -euo pipefail

REMOTE="${GO2_REMOTE:-go2-server}"
CONTAINER="${GO2_CONTAINER:-isaac-sim-kevin}"
HOST_DIR="${GO2_HOST_DIR:-/home/unitree_go2_isaaclab_rl}"
CT_DIR="${GO2_CT_DIR:-/workspace/go2_rl}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EXCLUDES=(
    --exclude .git/
    --exclude logs/
    --exclude backup_*/
    --exclude __pycache__/
    --exclude "*.pyc"
    --exclude configs/local_paths.yaml
)

echo "[1/3] rsync local -> ${REMOTE}:${HOST_DIR}"
rsync -az --delete "${EXCLUDES[@]}" "${ROOT}/" "${REMOTE}:${HOST_DIR}/"

echo "[2/3] tar-pipe ${REMOTE}:${HOST_DIR} -> ${CONTAINER}:${CT_DIR}"
ssh "${REMOTE}" "tar -C ${HOST_DIR} -cf - . | docker exec -i ${CONTAINER} tar -xf - -C ${CT_DIR}"

echo "[3/3] verify (local vs container md5)"
LOCAL_MD5="$(md5sum "${ROOT}/src/go2_rl/tasks/task1/task1_train.py" | awk '{print $1}')"
CT_MD5="$(ssh "${REMOTE}" "docker exec ${CONTAINER} md5sum ${CT_DIR}/src/go2_rl/tasks/task1/task1_train.py" | awk '{print $1}')"
if [ "${LOCAL_MD5}" = "${CT_MD5}" ]; then
    echo "[OK] sync complete (md5 match: ${LOCAL_MD5})"
else
    echo "[FAIL] md5 mismatch local=${LOCAL_MD5} container=${CT_MD5}"
    exit 1
fi
