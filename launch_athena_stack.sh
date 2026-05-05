#!/bin/bash

# ATHENA Mac launcher
# Double-click this file to open all required Terminal windows

JETSON_USER="creed"
JETSON_IP="10.0.8.100"
JETSON_PASS="Creed123"

JETSON_PROJECT_DIR="jetsonano-test-F25/JetsonNano-Testing"
JETSON_API_DIR="jetsonano-test-F25/JetsonNano-Testing/jetson_api"

LOCAL_BACKEND_DIR="$HOME/Downloads/Lunabotics-UI/backend/src"
LOCAL_FRONTEND_DIR="$HOME/Downloads/Lunabotics-UI/frontend"

SSH_BASE="sshpass -p '${JETSON_PASS}' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${JETSON_USER}@${JETSON_IP}"

open_terminal() {
  TITLE="$1"
  CMD="$2"

  osascript <<EOF
tell application "Terminal"
    activate
    do script "echo '$TITLE'; echo; $CMD; echo; echo 'Process ended or crashed. Terminal staying open.'; exec \$SHELL"
end tell
EOF
}

echo "Launching ATHENA stack on macOS..."

open_terminal "Terminal 1 - Jetson Roborio Bridge" \
"$SSH_BASE 'cd ${JETSON_PROJECT_DIR} && python3 roborio_bridge.py'"

sleep 1

open_terminal "Terminal 2 - Jetson API" \
"$SSH_BASE 'cd ${JETSON_API_DIR} && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload'"

sleep 1

open_terminal "Terminal 3 - Jetson VNC Startup" \
"$SSH_BASE 'mkdir -p ~/.vnc && cat > ~/.vnc/xstartup <<'\''EOF'\''
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4
EOF
chmod +x ~/.vnc/xstartup
tigervncserver -kill :2 || true
tigervncserver :2 -localhost no -xstartup ~/.vnc/xstartup
echo
echo VNC should now be running on ${JETSON_IP}:5902
'"

# Give VNC a little time to start before launching ZEDAuto into it
sleep 5

open_terminal "Terminal 4 - Jetson Camera Startup inside VNC" \
"$SSH_BASE 'cd ${JETSON_PROJECT_DIR} && DISPLAY=:2 ./ZEDAuto/RunAuto.sh'"

sleep 1

open_terminal "Terminal 5 - UI Backend" \
"cd '${LOCAL_BACKEND_DIR}' && uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload"

sleep 1

open_terminal "Terminal 6 - UI Frontend" \
"cd '${LOCAL_FRONTEND_DIR}' && npm run dev"

sleep 1

open_terminal "Terminal 7 - Optional Commands" \
"echo 'Optional command terminal'; echo; echo 'Useful command:'; echo 'curl http://127.0.0.1:8002/status'; echo; exec \$SHELL"

echo "All ATHENA terminals launched."