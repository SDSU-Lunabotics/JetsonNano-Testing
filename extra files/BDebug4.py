import Jetson.GPIO as GPIO
import time
#Hello 
# Pin definitions
BUTTON_PIN = 11  # Physical pin 7 (commonly pre-configured)

# Setup
GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("Button Press Detector - Press Ctrl+C to exit")
print("Waiting for button press...")
print()

last_state = GPIO.HIGH
button_was_pressed = False

try:
    while True:
        current_state = GPIO.input(BUTTON_PIN)
        
        # Print current state for debugging
        print(f"Pin state: {'HIGH' if current_state else 'LOW'}  ", end='\r')
        
        # Detect falling edge (button press - connects to GND)
        if current_state == GPIO.LOW and last_state == GPIO.HIGH:
            if not button_was_pressed:
                print("\n>>> BUTTON PRESSED! <<<                    ")
                button_was_pressed = True
        
        # Detect rising edge (button release)
        elif current_state == GPIO.HIGH and last_state == GPIO.LOW:
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