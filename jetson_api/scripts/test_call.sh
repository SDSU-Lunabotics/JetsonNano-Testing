#!/bin/bash
set -e

echo "status=success"
echo "message=Jetson script call successful"
echo "timestamp=$(date --iso-8601=seconds)"
echo "user=$(whoami)"
echo "pwd=$(pwd)"
