"""
CARLoS SEMBAS Client (ScenarioEvaluator)
=========================================

Connects to a SEMBAS server on 127.0.0.1:2000, receives normalised scenario
vectors (ndim=3, x ∈ [0,1]^3), and returns PASS/FAIL labels produced by a
pre-trained PPO Stage-3-Hard agent via ScenarioEvaluator.

Startup
-------
Use the provided launch script (recommended):

    bash /mnt/c/Users/lucca/Documents/SEMBAS-RL/run_client.sh

Or manually from a WSL shell with rl_venv313 activated:

    export SDL_VIDEODRIVER=dummy
    export PYTHONPATH=/mnt/c/Users/lucca/Documents/CARLoS-Agents:$PYTHONPATH
    python3 /mnt/c/Users/lucca/Documents/SEMBAS-RL/src/carlos_sembas_client.py

Scenario space (ndim=3)
-----------------------
  x[0] → lane_width_ft  = 10 + x[0] * 4    range [10, 14] ft
  x[1] → num_obstacles  = 4  + x[1] * 6    range [4, 10]
  x[2] → initial_speed  = 10 + x[2] * 65   range [10, 75] mph
"""

import os
import struct
import sys

# Suppress pygame display before any CARLoS/gym import touches it.
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Make the CARLoS-Agents rlagent package importable under WSL.
_CARLOS_AGENTS = "/mnt/c/Users/lucca/Documents/CARLoS-Agents"
if _CARLOS_AGENTS not in sys.path:
    sys.path.insert(0, _CARLOS_AGENTS)

from rlagent.sembas_evaluator import ScenarioEvaluator  # noqa: E402

# Ensure sembas_api is importable when running from outside src/.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from sembas_api import receive_request, send_response, setup_socket  # noqa: E402

# ── Best available PPO Stage-3 Hard checkpoint ────────────────────────────────
_CHECKPOINT = os.path.join(
    _CARLOS_AGENTS, "checkpoints", "ppo_Stage2_Medium_181920_steps"
)

NDIM = 3
SEED = 42


def main() -> None:
    print(f"Loading ScenarioEvaluator: {_CHECKPOINT}")
    evaluator = ScenarioEvaluator(_CHECKPOINT, seed=SEED)
    print("Model loaded.")

    print("Connecting to SEMBAS server (127.0.0.1:2000)…")
    client = setup_socket(NDIM)
    print(f"Connected. Listening for scenario vectors (ndim={NDIM}).")

    query_count = 0
    try:
        while True:
            x = receive_request(client, NDIM)
            query_count += 1

            print(
                f"[{query_count:>4}] x={[round(v, 4) for v in x.tolist()]}",
                end="  →  ",
                flush=True,
            )

            passed = evaluator.evaluate_vector(x.numpy())
            print("PASS" if passed else "FAIL")
            send_response(client, passed)

    except (ConnectionResetError, BrokenPipeError, OSError, struct.error) as exc:
        print(f"\nConnection closed by SEMBAS server ({exc}).")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        client.close()
        print(f"Evaluated {query_count} scenario(s). Connection closed.")


if __name__ == "__main__":
    main()
