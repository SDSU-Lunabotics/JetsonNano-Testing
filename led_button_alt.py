#!/usr/bin/env python3
"""LED blinker gated by a button input for Jetson Orin/Nano using Jetson.GPIO.

Default wiring assumes:
  * LED anode on pin 7, cathode (through resistor) to any GND (e.g. pin 6).
  * Button between pin 13 and GND so that pressing shorts to GND.
Set BUTTON_ACTIVE_LOW = False and move the button to 3V3 if you would rather
have the button drive the pin HIGH when pressed.
"""

import time

import Jetson.GPIO as GPIO

LED_PIN = 7        # LED anode on pin 7, cathode to pin 6 GND
BUTTON_PIN = 13    # Button reference pin (BOARD numbering)
BUTTON_ACTIVE_LOW = True  # True if button shorts to GND when pressed
BUTTON_PULL = GPIO.PUD_UP if BUTTON_ACTIVE_LOW else GPIO.PUD_DOWN
BLINK_INTERVAL = 0.5  # seconds
IDLE_POLL_INTERVAL = 0.02


def button_pressed() -> bool:
    """Return True when the configured button input is active."""
    state = GPIO.input(BUTTON_PIN)
    active_level = GPIO.LOW if BUTTON_ACTIVE_LOW else GPIO.HIGH
    return state == active_level


def main() -> None:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=BUTTON_PULL)

    print(
        f"Blinker ready: LED pin {LED_PIN}, button pin {BUTTON_PIN} "
        f"(expects {'LOW' if BUTTON_ACTIVE_LOW else 'HIGH'} while pressed)."
    )
    print(
        f"Button currently reads "
        f"{'PRESSED' if button_pressed() else 'released'}; adjust wiring if needed."
    )

    try:
        while True:
            if button_pressed():
                GPIO.output(LED_PIN, GPIO.HIGH)
                time.sleep(BLINK_INTERVAL)
                GPIO.output(LED_PIN, GPIO.LOW)
                time.sleep(BLINK_INTERVAL)
            else:
                GPIO.output(LED_PIN, GPIO.LOW)
                time.sleep(IDLE_POLL_INTERVAL)
    finally:
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.cleanup()


if __name__ == "__main__":
    main()
