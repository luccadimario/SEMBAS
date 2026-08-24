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
    python3 /mnt/c/Users/lucca/Documents/SEMBAS-RL/src/carlos_sembas_client.py \
        --checkpoint "C:\\Users\\lucca\\Documents\\CARLoS-Agents\\checkpoints\\ppo_Stage3_Hard_282272_steps.zip" \
        --algo ppo

CLI Arguments
-------------
  --checkpoint  Windows path to the .zip checkpoint (translated to /mnt/c/...
                automatically). Default: ppo_Stage2_Medium_181920_steps.zip
  --algo        RL algorithm class: ppo | ddpg | sac  (default: ppo)

Scenario space (ndim=3)
-----------------------
  x[0] → lane_width_ft  = 10 + x[0] * 4    range [10, 14] ft
  x[1] → num_obstacles  = 4  + x[1] * 6    range [4, 10]
  x[2] → initial_speed  = 10 + x[2] * 65   range [10, 75] mph
"""

import argparse
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

_DEFAULT_CHECKPOINT = (
    r"C:\Users\lucca\Documents\CARLoS-Agents\checkpoints"
    r"\ppo_Stage2_Medium_181920_steps.zip"
)

NDIM = 3
SEED = 42

_ALGO_MAP = {"ppo": "PPO", "ddpg": "DDPG", "sac": "SAC"}


def _win_to_wsl(path: str) -> str:
    """Translate a Windows path (C:\\...) to its WSL /mnt/<drive>/... equivalent."""
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        rest = path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return path.replace("\\", "/")


def _load_algo_class(name: str):
    """Return the stable_baselines3 class for the given algo name."""
    import importlib
    cls_name = _ALGO_MAP.get(name.lower())
    if cls_name is None:
        raise ValueError(f"Unknown algo '{name}'. Choose from: {list(_ALGO_MAP)}")
    return getattr(importlib.import_module("stable_baselines3"), cls_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="CARLoS SEMBAS client")
    parser.add_argument(
        "--checkpoint",
        default=_DEFAULT_CHECKPOINT,
        help=(
            "Path to the .zip checkpoint. Windows paths (C:\\...) are translated "
            "to /mnt/<drive>/... automatically. Default: ppo_Stage2_Medium_181920_steps.zip"
        ),
    )
    parser.add_argument(
        "--algo",
        default="ppo",
        choices=list(_ALGO_MAP),
        help="RL algorithm used to load the checkpoint (default: ppo).",
    )
    parser.add_argument(
        "--speed-aware",
        action="store_true",
        default=False,
        help=(
            "Wrap the env with SpeedAwareEnv (13D obs: 12 sensors + normalised speed). "
            "Required for SAC checkpoints from train_sac_speed_obs.py. "
            "Default: False (plain 12D env for PPO checkpoints)."
        ),
    )
    parser.add_argument(
        "--sps",
        action="store_true",
        default=False,
        help=(
            "Wrap the env with SpeedProportionalSteeringEnv at evaluation time. "
            "Required for checkpoints trained with train_sps_ppo.py or ExpD/ExpE. "
            "Default: False."
        ),
    )
    parser.add_argument(
        "--sensor-length",
        type=float,
        default=None,
        metavar="FT",
        help=(
            "Override sensor range to FT feet and use normalised [0,1] obs with "
            "speed appended (13D).  Required for ExpE checkpoints trained with "
            "train_expe_extended_sensors.py (default sensor_length=300).  "
            "When set, overrides --speed-aware.  Default: None (120 ft, raw obs)."
        ),
    )
    args = parser.parse_args()

    checkpoint_path = _win_to_wsl(args.checkpoint)
    algo_cls = _load_algo_class(args.algo)

    print(
        f"Loading ScenarioEvaluator: {checkpoint_path} "
        f"(algo={args.algo.upper()}, speed_aware={args.speed_aware}, "
        f"sps={args.sps}, sensor_length={args.sensor_length})"
    )
    evaluator = ScenarioEvaluator(
        checkpoint_path, seed=SEED, algo_class=algo_cls,
        speed_aware=args.speed_aware,
        sps=args.sps,
        sensor_length_ft=args.sensor_length,
    )
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
