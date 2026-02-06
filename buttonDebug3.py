import Jetson.GPIO as GPIO
import time

# Pin definitions
BUTTON_PIN = 29  # Physical pin 29 (trying different pin)

# Setup
GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

print("Button Press Detector - Press Ctrl+C to exit")
print("Waiting for button press...")
print()

last_state = GPIO.LOW
button_was_pressed = False

try:
    while True:
        current_state = GPIO.input(BUTTON_PIN)
        
        # Print current state for debugging
        print(f"Pin state: {'HIGH' if current_state else 'LOW'}  ", end='\r')
        
        # Detect rising edge (button press)
        if current_state == GPIO.HIGH and last_state == GPIO.LOW:
            if not button_was_pressed:
                print("\n>>> BUTTON PRESSED! <<<                    ")
                button_was_pressed = True
        
        # Detect falling edge (button release)
        elif current_state == GPIO.LOW and last_state == GPIO.HIGH:
            print("\n>>> Button released <<<                    ")
            button_was_pressed = False
        
        last_state = current_state
        time.sleep(0.05)
        
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    GPIO.cleanup()
    print("Cleaned up.")