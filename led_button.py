#!/usr/bin/env python3
"""
Simple button-to-LED bridge for Jetson Orin/Nano using Jetson.GPIO.

Wire the button between the chosen button pin and GND (uses pull-up).
Wire the LED (with a current-limiting resistor) between the LED pin and GND.
Adjust BUTTON_PIN and LED_PIN to match your header wiring (BOARD numbering).
"""

import time

import Jetson.GPIO as GPIO

# Physical pin numbers on the 40-pin header (BOARD mode).
# Using pins that default low at boot to avoid the LED turning on early.
BUTTON_PIN = 16  # button to GND; press pulls low (GPIO23)
LED_PIN = 18     # LED anode; cathode to GND through resistor (GPIO24)


def main() -> None:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    try:
        while True:
            pressed = GPIO.input(BUTTON_PIN) == GPIO.LOW
            GPIO.output(LED_PIN, GPIO.HIGH if pressed else GPIO.LOW)
            time.sleep(0.02)  # debounce-ish
    finally:
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.cleanup()


if __name__ == "__main__":
    main()
