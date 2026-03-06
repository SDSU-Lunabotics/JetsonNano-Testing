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
BUTTON_PIN = 13  # button to GND; press pulls low
LED_PIN = 12     # LED anode; cathode to GND through resistor


def main() -> None:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    print(
        f"Watching button on pin {BUTTON_PIN} -> driving LED on pin {LED_PIN}. "
        "Press Ctrl+C to exit."
    )

    try:
        while True:
            pressed = GPIO.input(BUTTON_PIN) == GPIO.LOW
            GPIO.output(LED_PIN, GPIO.HIGH if pressed else GPIO.LOW)
            time.sleep(0.02)  # debounce-ish
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.cleanup()


if __name__ == "__main__":
    main()
