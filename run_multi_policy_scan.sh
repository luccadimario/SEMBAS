#!/usr/bin/env bash
# run_multi_policy_scan.sh
# Sequential SEMBAS boundary scans across multiple PPO training checkpoints.
#
# Run from WSL:
#   bash /mnt/c/Users/lucca/Documents/SEMBAS-RL/run_multi_policy_scan.sh

SEMBAS_SRC="/mnt/c/Users/lucca/Documents/SEMBAS-RL/src"
CHECKPOINTS_BASE="/mnt/c/Users/lucca/Documents/CARLoS-Agents/checkpoints"

export SDL_VIDEODRIVER=dummy
export PYTHONPATH="/mnt/c/Users/lucca/Documents/CARLoS-Agents:/mnt/c/Users/lucca/Documents/SEMBAS-RL/src"

echo "Activating ~/rl_venv313..."
# shellcheck disable=SC1090
source ~/rl_venv313/bin/activate

# ── run_scan <checkpoint_wsl_path> <output_json_filename> ─────────────────────
# Starts the server (background), waits for binding, starts client, then waits
# for the server to finish saving results.  Logs and continues on client failure.
run_scan() {
    local checkpoint="$1"
    local output="$2"

    echo ""
    echo "=========================================="
    echo "  Checkpoint : $(basename "$checkpoint")"
    echo "  Output     : $output"
    echo "=========================================="

    # Server binds and waits; must be started before the client.
    python3 "$SEMBAS_SRC/sembas_server.py" --output "$output" &
    local server_pid=$!

    sleep 2  # Give the server time to bind and start listening.

    if python3 "$SEMBAS_SRC/carlos_sembas_client.py" \
            --checkpoint "$checkpoint" \
            --algo ppo; then
        # Client finished cleanly — wait for the server to save results and exit.
        if wait "$server_pid"; then
            echo "DONE: $output"
        else
            echo "WARNING: Server exited non-zero for $output"
        fi
    else
        echo "ERROR: Client failed for $(basename "$checkpoint") — skipping."
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi

    echo "--- Finished: $(basename "$checkpoint") ---"
}

# ── Checkpoints (WSL paths) — executed in training order ──────────────────────

run_scan "$CHECKPOINTS_BASE/ppo_Stage1_Easy_50000_steps.zip"       "boundary_ppo_stage1_easy_50k.json"
run_scan "$CHECKPOINTS_BASE/ppo_Stage2_Medium_131920_steps.zip"    "boundary_ppo_stage2_med_131k.json"
run_scan "$CHECKPOINTS_BASE/ppo_Stage2_Medium_181920_steps.zip"    "boundary_ppo_stage2_med_181k.json"
run_scan "$CHECKPOINTS_BASE/ppo_Stage3_Hard_232272_steps.zip"      "boundary_ppo_stage3_hard_232k.json"
run_scan "$CHECKPOINTS_BASE/ppo_Stage3_Hard_282272_steps.zip"      "boundary_ppo_stage3_hard_282k.json"

echo ""
echo "All scans complete. Results saved to: $SEMBAS_SRC"
