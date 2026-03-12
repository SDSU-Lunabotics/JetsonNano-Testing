#!/usr/bin/env bash
set -e

echo "[SCRIPT] capture_picture started at $(date)"

OUTPUT="/tmp/capture.png"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --path)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

echo "Would capture image to $OUTPUT"

sleep 1

touch "$OUTPUT"

echo "[SCRIPT] capture_picture finished"