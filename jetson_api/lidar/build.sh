#!/bin/bash
# build.sh — builds lidar_bridge using the already-cloned unilidar_sdk2
# Run from ~/ (your home directory in WSL2)
#
# Usage:
#   cd ~
#   chmod +x build.sh
#   ./build.sh

set -e

SDK_DIR="$HOME/unilidar_sdk2/unitree_lidar_sdk"
BIN_DIR="$SDK_DIR/bin"
LIB_DIR="$SDK_DIR/lib/x86_64"
INC_DIR="$SDK_DIR/include"

echo "=== Checking SDK paths ==="
if [ ! -d "$SDK_DIR" ]; then
    echo "ERROR: SDK not found at $SDK_DIR"
    echo "Run: cp -r /mnt/c/Users/nagar/Downloads/unilidar_sdk2 ~/"
    exit 1
fi

if [ ! -d "$LIB_DIR" ]; then
    echo "ERROR: x86_64 lib not found at $LIB_DIR"
    echo "Available lib dirs:"
    ls "$SDK_DIR/lib/"
    exit 1
fi

# Find the static library
LIB_PATH=$(find "$LIB_DIR" -name "*.a" | head -1)
if [ -z "$LIB_PATH" ]; then
    LIB_PATH=$(find "$LIB_DIR" -name "*.so" | head -1)
fi
if [ -z "$LIB_PATH" ]; then
    echo "ERROR: No library found in $LIB_DIR"
    exit 1
fi
echo "  Library : $LIB_PATH"
echo "  Include : $INC_DIR"
echo "  Output  : $BIN_DIR/lidar_bridge"

echo ""
echo "=== Installing g++ if needed ==="
sudo apt-get install -y g++ 2>/dev/null || true

echo ""
echo "=== Compiling lidar_bridge ==="
mkdir -p "$BIN_DIR"

g++ -std=c++17 -O2 \
    -I"$INC_DIR" \
    "$HOME/lidar_bridge.cpp" \
    "$LIB_PATH" \
    -lpthread \
    -o "$BIN_DIR/lidar_bridge"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Build complete!                                         ║"
echo "║                                                          ║"
echo "║  Terminal 1 — start Python APF first:                   ║"
echo "║    python3 ~/lidar_apf.py                               ║"
echo "║                                                          ║"
echo "║  Terminal 2 — run the bridge:                           ║"
echo "║    sudo chmod a+rw /dev/ttyACM0                         ║"
echo "║    cd ~/unilidar_sdk2/unitree_lidar_sdk/bin              ║"
echo "║    ./lidar_bridge                                        ║"
echo "╚══════════════════════════════════════════════════════════╝"