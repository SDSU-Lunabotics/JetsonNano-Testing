import Jetson.GPIO as GPIO
import time

# Pin definitions (using BOARD numbering)
LED_PIN = 33  # Physical pin 33
BUTTON_PIN = 31  # Physical pin 31

# Setup
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("LED Button Control - Press Ctrl+C to exit")
print("Press the button to toggle the LED")

led_state = False

try:
    while True:
        # Read button state (LOW when pressed due to pull-up)
        button_state = GPIO.input(BUTTON_PIN)
        
        if button_state == GPIO.LOW:
            # Button is pressed - toggle LED
            led_state = not led_state
            GPIO.output(LED_PIN, GPIO.HIGH if led_state else GPIO.LOW)
            print(f"LED {'ON' if led_state else 'OFF'}")
            
            # Debounce delay
            time.sleep(0.3)
        
        time.sleep(0.01)  # Small delay to reduce CPU usage

except KeyboardInterrupt:
    print("\nCleaning up...")

finally:
    GPIO.cleanup()
    print("GPIO cleaned up. Exiting.")