import os, sys, numpy as np
sys.path.insert(0, '/mnt/c/Users/lucca/Documents/CARLoS-Agents')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlagent.carlos_gym_env import CarlosGymEnv
from stable_baselines3 import PPO

CKPT = '/mnt/c/Users/lucca/Documents/CARLoS-Agents/checkpoints/ppo_Stage2_Medium_181920_steps'
print('Loading model...')
model = PPO.load(CKPT)
print('Model obs_space:', model.observation_space)
print('Model action_space:', model.action_space)

# Easy scenario: 12ft lane, 6 obs, 30 mph
env_seed = 42
env = CarlosGymEnv(seed=env_seed, lane_width_ft=12, num_obstacles=6)
obs, _ = env.reset(seed=env_seed)
env.vehicle.speed_fps = env.vehicle.mph_to_fps(30.0)

print(f'Initial speed: {env.vehicle.speed_mph:.1f} mph')
print(f'Initial obs (12 sensors): {obs}')
print(f'in_lane={env.sim.vehicle_in_lane}, in_motion={env.sim.vehicle_in_motion}')

for step in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(
        f'Step {step+1}: action=[{action[0]:.3f},{action[1]:.3f}]  '
        f'spd={env.vehicle.speed_mph:.1f}mph  '
        f'in_lane={info["in_lane"]}  collision={info["collision"]}  '
        f'event={info["terminal_event"]}  reward={reward:.2f}'
    )
    if terminated or truncated:
        break
env.close()
