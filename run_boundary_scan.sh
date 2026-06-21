#!/usr/bin/env bash
# Simplified launch script for the SEMBAS boundary scan.
# Starts the server in the background, then runs the client in the foreground.
# Results → src/boundary_scan_results.json

source ~/rl_venv313/bin/activate
export SDL_VIDEODRIVER=dummy
export PYTHONPATH=/mnt/c/Users/lucca/Documents/CARLoS-Agents

cd /mnt/c/Users/lucca/Documents/SEMBAS-RL

python3 src/sembas_server.py > /tmp/server.log 2>&1 &
SERVER_PID=$!
sleep 2

python3 src/carlos_sembas_client.py 2>&1 | tee /tmp/client.log

kill "$SERVER_PID" 2>/dev/null || true
echo ""
echo "=== SERVER LOG ==="
cat /tmp/server.log
