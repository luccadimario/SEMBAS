import os, sys, numpy as np
sys.path.insert(0, '/mnt/c/Users/lucca/Documents/CARLoS-Agents')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlagent.sembas_evaluator import ScenarioEvaluator
from rlagent.carlos_gym_env import CarlosGymEnv
from stable_baselines3 import PPO

CKPT = '/mnt/c/Users/lucca/Documents/CARLoS-Agents/checkpoints/ppo_Stage2_Medium_181920_steps'
model = PPO.load(CKPT)

# Replicate evaluate() logic manually with verbose step output
# Use env_seed=42 (same as our passing diagnostic)
lane_width_ft = 12.0
num_obstacles = 6.0
initial_speed_mph = 30.0

lane_width = int(round(lane_width_ft))
n_obs = int(round(num_obstacles))
SEED = 42
env_seed = (SEED * 31337 + lane_width * 997 + n_obs * 101) % (2**31)

print(f"env_seed used by evaluate(): {env_seed}  (SEED=42, lane={lane_width}, obs={n_obs})")
print(f"Diagnostic used env_seed=42 directly")

env = CarlosGymEnv(seed=env_seed, lane_width_ft=lane_width, num_obstacles=n_obs)
obs, _ = env.reset(seed=env_seed)
env.vehicle.speed_fps = env.vehicle.mph_to_fps(float(np.clip(initial_speed_mph, 0.1, 150.0)))

print(f"Initial speed: {env.vehicle.speed_mph:.1f} mph")
print(f"Initial obs: {obs}")

step = 0
terminated = truncated = False
while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, info = env.step(action)
    step += 1
    if step <= 5 or terminated or truncated:
        print(f"Step {step:>3}: in_lane={info['in_lane']}  collision={info['collision']}  event={info['terminal_event']}  spd={env.vehicle.speed_mph:.1f}mph")

env.close()
result = bool(truncated)
print(f"\nevaluate() result: {'PASS' if result else 'FAIL'}  (steps={step})")
print(f"  terminated={terminated}, truncated={truncated}")
