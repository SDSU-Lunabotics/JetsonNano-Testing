# Jetson Nano Button-and-LED Examples

Two short Python scripts show how to read a momentary push button and drive an
LED through the 40‑pin expansion header on Jetson Nano/Orin boards using
`Jetson.GPIO`.

## Requirements
- Jetson Nano, Jetson Orin Nano/Orin NX, or another Jetson with the 40‑pin
  header
- Ubuntu for Jetson with Python 3 and the `Jetson.GPIO` package
  (`sudo python3 -m pip install Jetson.GPIO` if it is not already present)
- Breadboard wiring parts: momentary push button, LED, and ~220 Ω resistor

If you are not running the script as `root`, make sure your user is part of the
`gpio` group (`sudo usermod -aG gpio $USER`) and log out/in once so Jetson.GPIO
can access `/dev/gpiochip*`.

## Optional: remote access / headless setup
### Enable SSH on the Jetson
1. Run `sudo systemctl enable --now ssh` once to turn on the OpenSSH server.
2. Discover the Jetson's IP address with `hostname -I` (grab the first address an Ethernet cable or Wi-Fi reports).

You can now connect from any machine with `ssh <user>@<jetson-ip>`. 

### Connect through VS Code Remote‑SSH
1. Install [Visual Studio Code] on your PC/laptop.
2. Install Microsoft's “Remote - SSH” extension inside VS Code.
3. Open the command palette (`Ctrl+Shift+P`), type `Remote-SSH: Add New SSH Host…`,
   and enter `ssh <user>@<jetson-ip>`. Save it to your default `~/.ssh/config`.
4. In the lower-left remote indicator or command palette, choose “Connect to Host…”, pick the entry you just saved, and wait for VS Code to install its server on the Jetson.
5. When the remote window opens, use “File → Open Folder…” and browse to the cloned repo directory on the Jetson (e.g. `/home/<user>/JetsonNano-Testing`).
6. Done! Once you save your code it will update on the jetson nano.

## Wiring cheat sheet
| Script | LED pin (BOARD numbering) | Button pin | Notes |
| ------ | ------------------------ | ---------- | ----- |
| `led_button.py` | Pin 12 anode → resistor → LED → GND | Pin 13 → button → GND | LED mirrors the button state one-to-one |
| `led_button_alt.py` | Pin 7 anode → resistor → LED → pin 6 GND | Pin 13 → button → GND | Button gates a blinking LED; set `BUTTON_ACTIVE_LOW = False` if the button drives 3V3 instead of GND |

Feel free to move the wiring to any other pins—just update the `BUTTON_PIN`,
`LED_PIN`, and related constants at the top of each script so they match your
layout.

## Running the scripts
From the repository directory:

```bash
python3 led_button.py
python3 led_button_alt.py
```

Both scripts will print the configured pin assignments and keep running until
you press `Ctrl+C`. They clean up the GPIO lines on exit so you can rerun them
without rebooting.

## Script behavior
- `led_button.py` configures the button with an internal pull-up resistor and
  simply drives the LED HIGH when the button reads LOW (pressed). The short
  `time.sleep(0.02)` call provides basic debounce.
- `led_button_alt.py` uses the button as a gate for a blinking LED. When the
  button is not pressed it polls at a low duty cycle so the LED stays off.
  Adjust the `BLINK_INTERVAL` to change the blink speed.

These examples are deliberately minimal so you can drop the functions into your
own Jetson projects or expand them with additional sensors, logging, or UI
behavior.


## debug update
https://github.com/jetsonhacks/jetson-orin-gpio-patch

