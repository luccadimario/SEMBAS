#!/bin/bash
source ~/rl_venv313/bin/activate
export SDL_VIDEODRIVER=dummy
export PYTHONPATH=/mnt/c/Users/lucca/Documents/CARLoS-Agents:$PYTHONPATH
python3 /mnt/c/Users/lucca/Documents/SEMBAS-RL/src/carlos_sembas_client.py
