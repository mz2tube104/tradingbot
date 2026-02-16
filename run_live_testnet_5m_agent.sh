#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/live_testnet_5m_agent.log"
PID_FILE="${LOG_DIR}/live_testnet_5m_agent.pid"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "5m agent already running (PID: $(cat "${PID_FILE}"))."
  exit 0
fi

{
  while true; do
    echo "[$(date -u '+%F %T')] start live_testnet_dry_run_5m"
    python3 -u "${ROOT_DIR}/live_testnet_trader.py" \
      --dry-run \
      --poll-seconds 60 \
      --monitor-interval 30 \
      --initial-capital-usdt 10000 \
      --out-dir "${ROOT_DIR}"
    RC=$?
    echo "[$(date -u '+%F %T')] exited rc=${RC}, restarting in 10s"
    sleep 10
  done
} | tee -a "${LOG_FILE}" &

echo $! > "${PID_FILE}"
echo "Started 5m dry-run agent: PID $!"
echo "Log: ${LOG_FILE}"
