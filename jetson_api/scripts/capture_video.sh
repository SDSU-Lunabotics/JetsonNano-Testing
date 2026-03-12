#!/usr/bin/env bash
set -e

echo "[SCRIPT] capture_video started at $(date)"

DURATION=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)
      DURATION="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

echo "Would record video for $DURATION seconds."

sleep "$DURATION"

echo "[SCRIPT] capture_video finished"