import Jetson.GPIO as GPIO
import time

# Pin definitions
BUTTON_PIN = 31  # Physical pin 31

# Setup
GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("Button Press Detector - Press Ctrl+C to exit")
print("Waiting for button press...")
print()

try:
    while True:
        button_state = GPIO.input(BUTTON_PIN)
        
        if button_state == GPIO.LOW:
            print("BUTTON IS PRESSED!")
            time.sleep(0.1)  # Small delay to avoid spam
        
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    GPIO.cleanup()
    print("Cleaned up.")