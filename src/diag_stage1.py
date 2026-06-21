import os, sys, numpy as np
sys.path.insert(0, '/mnt/c/Users/lucca/Documents/CARLoS-Agents')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlagent.sembas_evaluator import ScenarioEvaluator

CKPT1 = '/mnt/c/Users/lucca/Documents/CARLoS-Agents/checkpoints/ppo_Stage1_Easy_50000_steps'
CKPT2 = '/mnt/c/Users/lucca/Documents/CARLoS-Agents/checkpoints/ppo_Stage2_Medium_181920_steps'

print("=== Stage1_Easy vs Stage2_Medium on a grid ===\n")
test_cases = [
    # (lane, obs, spd)
    (14, 4,  10),
    (14, 4,  30),
    (14, 4,  50),
    (12, 4,  10),
    (12, 4,  30),
    (12, 6,  10),
    (12, 6,  30),
    (12, 8,  10),
    (10, 4,  10),
    (10, 6,  10),
]

ev1 = ScenarioEvaluator(CKPT1, seed=42)
ev2 = ScenarioEvaluator(CKPT2, seed=42)

print(f"{'Scenario':<22}  {'Stage1':>7}  {'Stage2':>7}")
print('-' * 40)
n1 = n2 = 0
for lane, obs, spd in test_cases:
    r1 = ev1.evaluate(lane, obs, spd)
    r2 = ev2.evaluate(lane, obs, spd)
    if r1: n1 += 1
    if r2: n2 += 1
    l1 = "PASS" if r1 else "FAIL"
    l2 = "PASS" if r2 else "FAIL"
    print(f"{lane}ft/{obs}obs/{spd}mph        {l1:>7}  {l2:>7}")

print(f"\nPass rates: Stage1={n1}/{len(test_cases)}  Stage2={n2}/{len(test_cases)}")
