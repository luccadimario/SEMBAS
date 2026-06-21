import os, sys, numpy as np
sys.path.insert(0, '/mnt/c/Users/lucca/Documents/CARLoS-Agents')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlagent.sembas_evaluator import ScenarioEvaluator

CKPT = '/mnt/c/Users/lucca/Documents/CARLoS-Agents/checkpoints/ppo_Stage2_Medium_181920_steps'
evaluator = ScenarioEvaluator(CKPT, seed=42)

# Test all extremes and midpoints
test_cases = [
    (10, 4,  10, "easy_min"),
    (12, 6,  30, "medium"),
    (14, 4,  10, "wide+easy"),
    (14, 10, 75, "hard_max"),
    (10, 10, 75, "extreme_hard"),
    (12, 4,  10, "baseline_easy"),
    (14, 6,  30, "wide_medium"),
]

print(f"{'Scenario':<15} {'lane':>6} {'obs':>4} {'spd':>7} {'result'}")
print('-' * 45)
n_pass = 0
for lane, obs, spd, name in test_cases:
    result = evaluator.evaluate(lane, obs, spd)
    label = "PASS" if result else "FAIL"
    if result:
        n_pass += 1
    print(f"{name:<15} {lane:>5}ft {obs:>4}obs {spd:>7.0f}mph  {label}")

# Also try with master_seed variants
print("\n--- Trying different master seeds ---")
for mseed in [0, 1, 7, 99, 123]:
    ev2 = ScenarioEvaluator(CKPT, seed=mseed)
    result = ev2.evaluate(12, 6, 30)
    label = "PASS" if result else "FAIL"
    print(f"seed={mseed:>3}: 12ft/6obs/30mph → {label}")

print(f"\nPass rate: {n_pass}/{len(test_cases)}")
