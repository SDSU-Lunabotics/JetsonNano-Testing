# jetson_record_playback.py
from networktables import NetworkTables
import time
import csv
import threading
import queue

ROBORIO_IP = "10.0.9.2"
NetworkTables.initialize(server=ROBORIO_IP)
sd = NetworkTables.getTable("SmartDashboard")

recording = False
playing = False
path = []
cmd_queue = queue.Queue()

RECORD_DT = 0.02  # still polling fast
DEADBAND = 0.05
CHANGE_THRESHOLD = 0.03
SPEED_SCALE = 1.5
COMMAND_ACK_TIMEOUT = 0.30
AUTOMATION_ENABLE_HEARTBEAT_DT = 0.10
AUTOMATION_FORWARD_SPEED = 1.0
AUTOMATION_TURN_SPEED = 1.0

def input_thread():
    while True:
        cmd = input("Command: ").strip().lower()
        cmd_queue.put(cmd)
        if cmd == "q":
            break

def apply_deadband(value, deadband):
    if abs(value) < deadband:
        return 0.0
    return value


def wait_for_connection(timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if NetworkTables.isConnected():
            return True
        time.sleep(0.05)
    return NetworkTables.isConnected()


def push_automation_state(enabled):
    sd.putBoolean("Jetson/AutomationEnabled", bool(enabled))
    # Explicitly set these because roboRIO scales commands by these values.
    # If left at 0.0, automation may appear to run but produce no movement.
    sd.putNumber("Jetson/Speed", AUTOMATION_FORWARD_SPEED if enabled else 0.0)
    sd.putNumber("Jetson/TurnSpeed", AUTOMATION_TURN_SPEED if enabled else 0.0)

def record_loop():
    last_forward = None
    last_turn = None

    while True:
        if recording:
            t = sd.getNumber("Jetson/DriveTimestamp", time.time())
            forward = sd.getNumber("Jetson/DriveForward", 0.0)
            turn = sd.getNumber("Jetson/DriveTurn", 0.0)

            forward = apply_deadband(forward, DEADBAND)
            turn = apply_deadband(turn, DEADBAND)

            if last_forward is None or \
               abs(forward - last_forward) > CHANGE_THRESHOLD or \
               abs(turn - last_turn) > CHANGE_THRESHOLD:
                path.append([t, forward, turn])
                last_forward = forward
                last_turn = turn

        time.sleep(RECORD_DT)

def save_path(filename="recorded_path.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "forward", "turn"])
        writer.writerows(path)

def load_path(filename="recorded_path.csv"):
    loaded = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loaded.append([float(row["t"]), float(row["forward"]), float(row["turn"])])
    return loaded

def clamp(val, lo=-1.0, hi=1.0):
    return max(lo, min(hi, val))

threading.Thread(target=input_thread, daemon=True).start()
threading.Thread(target=record_loop, daemon=True).start()

print("Commands: record on/off, play, stop, save, load, clear, q")

index = 0
last_enable_push = 0.0

while True:
    try:
        cmd = cmd_queue.get_nowait()
    except queue.Empty:
        cmd = None

    if cmd == "q":
        break
    if cmd == "record on":
        recording = True
        print("Recording...")
    elif cmd == "record off":
        recording = False
        print("Stopped recording.")
    elif cmd == "save":
        save_path()
        print("Saved recorded_path.csv")
    elif cmd == "load":
        path = load_path()
        print(f"Loaded {len(path)} points")
    elif cmd == "clear":
        path = []
        print("Cleared path.")
    elif cmd == "play":
        if not wait_for_connection():
            print("Not connected to roboRIO NetworkTables.")
            continue
        if recording:
            recording = False
            save_path()
            print("Auto-saved recorded_path.csv")
        if len(path) < 2:
            try:
                path = load_path()
                print(f"Loaded {len(path)} points")
            except FileNotFoundError:
                print("No recorded_path.csv found.")
                continue
        push_automation_state(True)
        last_enable_push = time.monotonic()
        sd.putBoolean("Jetson/CommandReady", False)
        playing = True
        index = 0
        print("Playback started. Type 'stop' to end.")
    elif cmd == "stop":
        playing = False
        push_automation_state(False)
        sd.putBoolean("Jetson/CommandReady", False)
        sd.putString("Jetson/Command", "")
        print("Playback stopped.")

    if playing and len(path) > 1:
        # Keep publishing so roboRIO reliably latches automation state.
        if time.monotonic() - last_enable_push >= AUTOMATION_ENABLE_HEARTBEAT_DT:
            push_automation_state(True)
            last_enable_push = time.monotonic()

        if index >= len(path) - 1:
            index = 0  # loop

        t0, f0, r0 = path[index]
        t1, _, _ = path[index + 1]
        duration = max(0.02, t1 - t0)

        scaled_f = clamp(f0 * SPEED_SCALE)
        scaled_t = clamp(r0 * SPEED_SCALE)

        # Avoid overwriting an in-flight command if roboRIO has not consumed it yet.
        ready_wait_deadline = time.monotonic() + COMMAND_ACK_TIMEOUT
        while sd.getBoolean("Jetson/CommandReady", False):
            if time.monotonic() >= ready_wait_deadline:
                print("Warning: stale CommandReady detected; clearing flag.")
                sd.putBoolean("Jetson/CommandReady", False)
                break
            time.sleep(0.005)

        sd.putNumber("Jetson/CommandForward", scaled_f)
        sd.putNumber("Jetson/CommandTurn", scaled_t)
        sd.putNumber("Jetson/CommandDuration", duration)
        sd.putBoolean("Jetson/CommandReady", True)

        ack_deadline = time.monotonic() + COMMAND_ACK_TIMEOUT
        acked = False
        while time.monotonic() < ack_deadline:
            if not sd.getBoolean("Jetson/CommandReady", False):
                acked = True
                break
            time.sleep(0.005)
        if not acked:
            print("Warning: Command ack timeout; roboRIO may not have consumed command yet.")

        index += 1
        sleep_until = time.monotonic() + duration
        while time.monotonic() < sleep_until and playing:
            # Keep automation enabled during command execution window.
            if time.monotonic() - last_enable_push >= AUTOMATION_ENABLE_HEARTBEAT_DT:
                push_automation_state(True)
                last_enable_push = time.monotonic()
            time.sleep(0.02)
    else:
        time.sleep(0.02)
