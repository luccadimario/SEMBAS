import json
import numpy as np
from scipy.spatial.distance import cdist

files = {
    "Stage3_Baseline": "boundary_ppo_stage3_hard_282k.json",
    "ExpA_HighLR_Entropy": "boundary_exp_a_lr_entropy.json",
    "ExpB_Staged": "boundary_exp_b_staged.json",
    "ExpC_RewardShaping": "boundary_exp_c_reward_shaping.json",
}

base_dir = r"C:\Users\lucca\Documents\SEMBAS-RL\src"
results = {}

for name, fname in files.items():
    with open(f"{base_dir}\\{fname}") as f:
        data = json.load(f)
    bp = np.array(data.get("boundary_points_normalized", []))
    pp = np.array(data.get("pass_points_normalized", []))
    fp = np.array(data.get("fail_points_normalized", []))
    total = len(pp) + len(fp)
    pass_rate = len(pp) / total if total > 0 else 0
    speed_ceil = float(pp[:, 2].max() * 65 + 10) if len(pp) > 0 else 0
    speed_mean = float(pp[:, 2].mean() * 65 + 10) if len(pp) > 0 else 0
    if len(bp) > 0 and len(fp) > 0:
        df = cdist(fp, bp)
        near_fail = float((df.min(axis=1) < 0.1).mean())
    else:
        near_fail = None
    results[name] = {
        "total_queries": total,
        "pass": len(pp), "fail": len(fp), "boundary": len(bp),
        "pass_rate": round(pass_rate * 100, 1),
        "speed_ceiling_mph": round(speed_ceil, 1),
        "mean_pass_speed_mph": round(speed_mean, 1),
        "near_bnd_fail_prob": round(near_fail, 3) if near_fail else None,
    }

# Print table
baseline = results["Stage3_Baseline"]
print(f"\n{'Metric':<28} {'Stage3':>10} {'ExpA':>10} {'ExpB':>10} {'ExpC':>10}")
print("-" * 68)
for key in ["pass_rate", "speed_ceiling_mph", "mean_pass_speed_mph", "near_bnd_fail_prob", "boundary"]:
    row = f"{key:<28}"
    for name in ["Stage3_Baseline", "ExpA_HighLR_Entropy", "ExpB_Staged", "ExpC_RewardShaping"]:
        val = results[name][key]
        row += f" {str(val):>10}"
    print(row)

# Hausdorff vs baseline
bp3 = np.array(json.load(open(f"{base_dir}\\boundary_ppo_stage3_hard_282k.json")).get("boundary_points_normalized", []))
print("\nHausdorff distance from Stage3 boundary:")
for name, fname in list(files.items())[1:]:
    with open(f"{base_dir}\\{fname}") as f:
        bpX = np.array(json.load(f).get("boundary_points_normalized", []))
    if len(bp3) > 0 and len(bpX) > 0:
        d = cdist(bpX, bp3)
        h = max(d.min(axis=1).max(), d.min(axis=0).max())
        print(f"  {name}: {h:.4f}")

# Save
import os
os.makedirs(r"C:\Users\lucca\Documents\SEMBAS-RL\outputs", exist_ok=True)
with open(r"C:\Users\lucca\Documents\SEMBAS-RL\outputs\experiment_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to outputs/experiment_comparison.json")
