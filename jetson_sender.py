import json
import sys
import urllib.request

URL = "http://127.0.0.1:8000/drive/forward"  # replace with computer IP

payload = {
    "action": "drive_forward",
    "duration": 3.0,
    "speed": 0.6,
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    URL,
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        print(resp.status)
        print(resp.read().decode("utf-8"))
except Exception as exc:
    print(f"Request failed: {exc}", file=sys.stderr)
    sys.exit(1)
