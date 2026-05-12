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
JETSON_API_CMD="cd ${JETSON_API_DIR} && python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload || { echo; echo Jetson API failed.; echo If the error says No module named uvicorn, run this on the Jetson:; echo cd ${JETSON_API_DIR} \&\& python3 -m pip install -r requirements.txt; }"

open_terminal() {
  TITLE="$1"
  CMD="$2"

  osascript - "$TITLE" "$CMD" <<'EOF'
on run argv
set theTitle to item 1 of argv
set theCmd to item 2 of argv
tell application "Terminal"
    activate
    do script "echo " & quoted form of theTitle & "; echo; " & theCmd & "; echo; echo 'Process ended or crashed. Terminal staying open.'; exec $SHELL"
end tell
end run
EOF
}

echo "Launching ATHENA stack on macOS..."

open_terminal "Terminal 1 - Jetson Roborio Bridge" \
"$SSH_BASE 'cd ${JETSON_PROJECT_DIR} && python3 roborio_bridge.py'"

sleep 1

open_terminal "Terminal 2 - Jetson API" \
"$SSH_BASE '${JETSON_API_CMD}'"

sleep 1


open_terminal "Terminal 3 - UI Backend" \
"cd '${LOCAL_BACKEND_DIR}' && uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload"

sleep 1

open_terminal "Terminal 4 - UI Frontend" \
"cd '${LOCAL_FRONTEND_DIR}' && npm run dev"

sleep 1

open_terminal "Terminal 5 - Optional Commands" \
"echo 'Optional command terminal'; echo; echo 'Useful command:'; echo 'curl http://127.0.0.1:8002/status'; echo; exec \$SHELL"

echo "All ATHENA terminals launched."
