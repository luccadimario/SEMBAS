import os, sys, numpy as np
sys.path.insert(0, '/mnt/c/Users/lucca/Documents/CARLoS-Agents')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlagent.sembas_evaluator import ScenarioEvaluator

CKPT = '/mnt/c/Users/lucca/Documents/CARLoS-Agents/checkpoints/ppo_Stage2_Medium_181920_steps'
evaluator = ScenarioEvaluator(CKPT, seed=42)

# The 15 LHS points from the scan (seed=0, ndim=3)
import random
def latin_hypercube(n, d, seed=0):
    rng = random.Random(seed)
    intervals = [[i / n, (i + 1) / n] for i in range(n)]
    cols = []
    for _ in range(d):
        perm = intervals[:]
        rng.shuffle(perm)
        cols.append([rng.uniform(lo, hi) for lo, hi in perm])
    return [[cols[dim][i] for dim in range(d)] for i in range(n)]

lhs_pts = latin_hypercube(15, 3, seed=0)

print(f"{'#':>3}  {'lane':>6}  {'obs':>4}  {'spd':>7}  {'result'}")
print(f"{'---':>3}  {'------':>6}  {'----':>4}  {'-------':>7}  {'------'}")
n_pass = 0
for i, x in enumerate(lhs_pts):
    lw  = 10 + x[0] * 4
    obs = 4  + x[1] * 6
    spd = 10 + x[2] * 65

    # Show what env_seed would be
    lane_i = int(round(lw))
    obs_i  = int(round(obs))
    env_seed = (42 * 31337 + lane_i * 997 + obs_i * 101) % (2**31)

    result = evaluator.evaluate_vector(np.array(x))
    label = "PASS" if result else "FAIL"
    if result:
        n_pass += 1
    print(f"[{i+1:>2}]  {lw:6.1f}ft  {obs_i:>4}obs  {spd:7.1f}mph  {label}  (seed={env_seed})")

print(f"\nPass rate: {n_pass}/{len(lhs_pts)} ({100*n_pass/len(lhs_pts):.0f}%)")
