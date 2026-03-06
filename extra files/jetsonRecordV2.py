# jetson_record_playback.py
from networktables import NetworkTables
import time
import csv
import threading
import queue

ROBORIO_IP = "10.0.8.2"
NetworkTables.initialize(server=ROBORIO_IP)
sd = NetworkTables.getTable("SmartDashboard")

recording = False
playing = False
path = []
cmd_queue = queue.Queue()

RECORD_DT = 0.05  # 20 Hz sampling

def input_thread():
    while True:
        cmd = input("Command: ").strip().lower()
        cmd_queue.put(cmd)
        if cmd == "q":
            break

def record_loop():
    while True:
        if recording:
            t = sd.getNumber("Jetson/DriveTimestamp", time.time())
            forward = sd.getNumber("Jetson/DriveForward", 0.0)
            turn = sd.getNumber("Jetson/DriveTurn", 0.0)
            path.append([t, forward, turn])
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

threading.Thread(target=input_thread, daemon=True).start()
threading.Thread(target=record_loop, daemon=True).start()

print("Commands: record on/off, play, stop, save, load, clear, q")

index = 0

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
        if recording:
            save_path()
            print("Auto-saved recorded_path.csv")
        if len(path) < 2:
            try:
                path = load_path()
                print(f"Loaded {len(path)} points")
            except FileNotFoundError:
                print("No recorded_path.csv found.")
                continue
        sd.putBoolean("Jetson/AutomationEnabled", True)
        playing = True
        index = 0
        print("Playback started. Type 'stop' to end.")
    elif cmd == "stop":
        playing = False
        sd.putBoolean("Jetson/AutomationEnabled", False)
        sd.putString("Jetson/Command", "")
        print("Playback stopped.")

    if playing and len(path) > 1:
        if index >= len(path) - 1:
            index = 0  # loop

        t0, f0, r0 = path[index]
        t1, _, _ = path[index + 1]
        duration = max(0.02, t1 - t0)

        sd.putNumber("Jetson/CommandForward", f0)
        sd.putNumber("Jetson/CommandTurn", r0)
        sd.putNumber("Jetson/CommandDuration", duration)
        sd.putBoolean("Jetson/CommandReady", True)

        index += 1
        time.sleep(duration)
    else:
        time.sleep(0.02)
