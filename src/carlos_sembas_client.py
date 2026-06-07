"""
CARLoS SEMBAS Client
====================

This script wires the SEMBAS socket protocol (``sembas_api.py``) to the CARLoS
evaluation loop so that SEMBAS can drive boundary-exploration queries against a
trained CARLoS agent.

Startup sequence
----------------
1. Launch the SEMBAS server first (it binds to 127.0.0.1:2000 and waits for a
   client to connect).

2. From the ``src/`` directory, run this client::

       python carlos_sembas_client.py <model_path> [--ndim N] [--max-steps M] [--seed S]

   Example::

       python carlos_sembas_client.py ./checkpoints/agent_summer_agent_1000.pt.pt

Expected flow
-------------
1. Client connects to SEMBAS and sends ``ndim`` as a big-endian int64.
2. SEMBAS replies ``"OK\\n"``.
3. Loop:
   a. SEMBAS sends ``ndim`` × float64 — a scenario vector x ∈ [0, 1]^ndim.
   b. Client denormalises x to concrete scenario parameters.
   c. Client rebuilds the CARLoS simulation with those parameters and runs one
      episode with a fixed random seed (deterministic, reproducible).
   d. Client sends a 1-byte boolean: ``0x01`` = pass, ``0x00`` = fail.
4. Loop exits when SEMBAS closes the connection.

Scenario parameter mapping (ndim = 3)
--------------------------------------
  x[0] → lane_width_ft     = round(10 + x[0] * 4)   range: 10–14 ft
  x[1] → num_obstacles     = round(4  + x[1] * 6)   range: 4–10
  x[2] → initial_speed_mph = 10 + x[2] * 65          range: 10–75 mph

Pass / fail criterion
---------------------
An episode *passes* when the vehicle remains in-lane **and** in-motion for the
full ``--max-steps`` time-steps.  It *fails* as soon as it leaves the lane or
comes to a stop.

Note on num_obstacles
---------------------
The current CARLoS simulation has no dynamic obstacle-placement API, so
``num_obstacles`` is parsed, logged per query, and marked as a placeholder for
future integration.
"""

import argparse
import math
import os
import random
import struct
import sys

import torch

# Allow running from the project root as well as from within src/
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import carlos_logging
from environment import Environment
from lane import Lane
from point import Point
from sembas_api import receive_request, send_response, setup_socket
from sensor_array import SensorArray
from simulation import Simulation
from summer_agent import SummerAgent
from vehicle import Vehicle

# ── Simulation constants ──────────────────────────────────────────────────────
NUM_SENSORS = 9
SENSOR_LENGTH = 200.0
SENSOR_ANGLE_SPREAD = math.pi
TIME_STEP_SEC = 0.1

# Simple straight lane matching the geometry used in carlos_app_for_tests.py
_LANE_START = Point(50, 150)
_LANE_END = Point(250, 150)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_NDIM = 3
DEFAULT_MAX_STEPS = 500
DEFAULT_SEED = 42


# ── Model loading ─────────────────────────────────────────────────────────────

def load_agent(model_path: str) -> SummerAgent:
    """Load a SummerAgent checkpoint once; reused across all SEMBAS queries."""
    obs_dim = NUM_SENSORS + 2  # speed + heading + sensor readings
    sensor_array = SensorArray(
        num_sensors=NUM_SENSORS,
        sensor_length=SENSOR_LENGTH,
        sensor_angle_spread=SENSOR_ANGLE_SPREAD,
    )
    agent = SummerAgent(sensor_array=sensor_array, obs_dim=obs_dim)
    agent.load(model_path)
    agent.actor.eval()
    agent.critic.eval()
    return agent


# ── Per-query simulation setup ────────────────────────────────────────────────

def build_sim(agent: SummerAgent, lane_width_ft: int, initial_speed_mph: float) -> Simulation:
    """Build a fresh Simulation for one scenario query (agent weights are shared)."""
    lane = Lane(
        control_points=[_LANE_START, _LANE_END],
        lane_width=float(lane_width_ft),
        closed_loop=False,
    )
    env = Environment(lane)
    vehicle = Vehicle()
    sim = Simulation(vehicle=vehicle, environment=env, agent=agent, dt=TIME_STEP_SEC)
    # Place vehicle near the lane start, centred laterally, aligned with lane
    sim.sim_reset(
        longitude=0.02,
        latitude=0.5,
        dir_angle_offset=0.0,
        speed=initial_speed_mph,
    )
    return sim


# ── Episode execution ─────────────────────────────────────────────────────────

def run_episode(sim: Simulation, max_steps: int, seed: int) -> bool:
    """Run one episode and return True (pass) or False (fail).

    Uses a fixed seed so that every call with the same scenario parameters
    produces an identical trajectory.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    done = False
    steps = 0
    with torch.no_grad():
        while not done and steps < max_steps:
            sim.sim_step()
            _, in_lane, in_motion = sim.get_sim_status()
            done = not in_lane or not in_motion
            steps += 1

    # Pass = survived the full episode without leaving the lane or stopping
    return not done


# ── Denormalisation ───────────────────────────────────────────────────────────

def denormalize(x: torch.Tensor) -> tuple[int, int, float]:
    """Map a normalised scenario vector x ∈ [0,1]^3 to concrete parameters."""
    lane_width_ft = round(10 + x[0].item() * 4)      # 10–14 ft
    num_obstacles = round(4 + x[1].item() * 6)        # 4–10  (placeholder)
    initial_speed_mph = 10.0 + x[2].item() * 65.0     # 10–75 mph
    return lane_width_ft, num_obstacles, initial_speed_mph


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CARLoS SEMBAS client — connects to a running SEMBAS server, "
            "receives scenario vectors, runs evaluation episodes, and returns "
            "pass/fail labels."
        )
    )
    parser.add_argument(
        "model_path",
        help="Path to a trained SummerAgent checkpoint (.pt file).",
    )
    parser.add_argument(
        "--ndim",
        type=int,
        default=DEFAULT_NDIM,
        help=f"Scenario-vector dimension expected by SEMBAS (default: {DEFAULT_NDIM}).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum simulation steps per episode (default: {DEFAULT_MAX_STEPS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Fixed random seed for deterministic evaluation (default: {DEFAULT_SEED}).",
    )
    args = parser.parse_args()

    carlos_logging.init_logger("./logs/carlos_sembas_client.log")

    print(f"Loading model: {args.model_path}")
    agent = load_agent(args.model_path)
    print("Model loaded.")

    print("Connecting to SEMBAS server (127.0.0.1:2000)…")
    client = setup_socket(args.ndim)
    print(f"Connected. Listening for scenario vectors (ndim={args.ndim}).")

    query_count = 0
    try:
        while True:
            x = receive_request(client, args.ndim)
            lane_width_ft, num_obstacles, initial_speed_mph = denormalize(x)
            query_count += 1

            print(
                f"[{query_count:>4}] lane_width={lane_width_ft}ft  "
                f"obstacles={num_obstacles}  "
                f"speed={initial_speed_mph:.1f}mph",
                end="  →  ",
                flush=True,
            )

            sim = build_sim(agent, lane_width_ft, initial_speed_mph)
            passed = run_episode(sim, max_steps=args.max_steps, seed=args.seed)

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
