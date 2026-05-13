#!/usr/bin/env bash
set -e

echo "[SCRIPT] restart_roborio started at $(date)"

sudo python3 - <<'PY'
import time
import Jetson.GPIO as GPIO

RELAY_PIN = 7  # physical pin 7 = GPIO09

ROBORIO_ON = GPIO.HIGH
ROBORIO_OFF = GPIO.LOW

POWER_OFF_SECONDS = 3

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

try:
    GPIO.setup(RELAY_PIN, GPIO.OUT, initial=ROBORIO_ON)

    print("[RoboRIO Reset] Cutting RoboRIO power...")
    GPIO.output(RELAY_PIN, ROBORIO_OFF)
    time.sleep(POWER_OFF_SECONDS)

    print("[RoboRIO Reset] Restoring RoboRIO power...")
    GPIO.output(RELAY_PIN, ROBORIO_ON)
    time.sleep(0.5)

    print("[RoboRIO Reset] Done.")

finally:
    GPIO.output(RELAY_PIN, ROBORIO_ON)
    GPIO.cleanup()
PY

sleep 2

echo "[SCRIPT] restart_roborio finished"