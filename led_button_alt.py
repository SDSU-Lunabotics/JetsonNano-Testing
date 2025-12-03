#!/usr/bin/env python3
"""
Alternate button-to-LED bridge for Jetson Orin/Nano using Jetson.GPIO.

Uses BOARD numbering with pins away from SPI defaults:
  BUTTON_PIN = 31 (GPIO06), LED_PIN = 33 (GPIO13)
Wire button: pin 31 -> button -> GND (internal pull-up).
Wire LED: pin 33 -> resistor -> LED -> GND.
"""

import time

import Jetson.GPIO as GPIO

BUTTON_PIN = 31  # button to GND; press pulls low
LED_PIN = 33     # LED anode; cathode to GND through resistor


def main() -> None:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    try:
        while True:
            pressed = GPIO.input(BUTTON_PIN) == GPIO.LOW
            GPIO.output(LED_PIN, GPIO.HIGH if pressed else GPIO.LOW)
            time.sleep(0.02)
    finally:
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.cleanup()


if __name__ == "__main__":
    main()
