import json
import numpy as np
from scipy.spatial.distance import cdist

with open(r'/mnt/c/Users/lucca/Documents/SEMBAS-RL/src/boundary_ppo_stage4_speed_ext.json') as f:
    s4 = json.load(f)

with open(r'/mnt/c/Users/lucca/Documents/SEMBAS-RL/src/boundary_ppo_stage3_hard_282k.json') as f:
    s3 = json.load(f)

bp4 = np.array(s4.get('boundary_points_normalized', []))
pp4 = np.array(s4.get('pass_points_normalized', []))
fp4 = np.array(s4.get('fail_points_normalized', []))
bp3 = np.array(s3.get('boundary_points_normalized', []))
pp3 = np.array(s3.get('pass_points_normalized', []))
fp3 = np.array(s3.get('fail_points_normalized', []))

total4 = len(pp4) + len(fp4)
pass_rate4 = len(pp4) / total4 if total4 > 0 else 0
speed_ceil4 = (pp4[:, 2].max() * 65 + 10) if len(pp4) > 0 else 0

total3 = len(pp3) + len(fp3)
pass_rate3 = len(pp3) / total3 if total3 > 0 else 0
speed_ceil3 = (pp3[:, 2].max() * 65 + 10) if len(pp3) > 0 else 0

if len(bp4) > 0 and len(bp3) > 0:
    d = cdist(bp4, bp3)
    hausdorff = float(max(d.min(axis=1).max(), d.min(axis=0).max()))
    chamfer = float(d.min(axis=1).mean() + d.min(axis=0).mean())
else:
    hausdorff = chamfer = None

if len(bp4) > 0 and len(fp4) > 0:
    df = cdist(fp4, bp4)
    near_fail4 = float((df.min(axis=1) < 0.1).mean())
else:
    near_fail4 = None

if len(bp3) > 0 and len(fp3) > 0:
    df3 = cdist(fp3, bp3)
    near_fail3 = float((df3.min(axis=1) < 0.1).mean())
else:
    near_fail3 = None

result = {
    "stage4": {
        "n_queries": int(s4.get("n_queries", total4)),
        "pass_rate": pass_rate4,
        "speed_ceiling_mph": float(speed_ceil4),
        "boundary_points": len(bp4),
        "pass_points": len(pp4),
        "fail_points": len(fp4),
        "near_boundary_fail_prob": near_fail4,
    },
    "stage3": {
        "n_queries": int(s3.get("n_queries", total3)),
        "pass_rate": pass_rate3,
        "speed_ceiling_mph": float(speed_ceil3),
        "boundary_points": len(bp3),
        "pass_points": len(pp3),
        "fail_points": len(fp3),
        "near_boundary_fail_prob": near_fail3,
    },
    "comparison": {
        "hausdorff_boundary_distance": hausdorff,
        "chamfer_boundary_distance": chamfer,
        "speed_ceiling_delta_mph": float(speed_ceil4 - speed_ceil3),
        "pass_rate_delta": float(pass_rate4 - pass_rate3),
    },
}

with open(r'/mnt/c/Users/lucca/Documents/SEMBAS-RL/outputs/stage4_comparison.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"Stage4: pass_rate={pass_rate4:.3f}, speed_ceil={speed_ceil4:.1f}mph, boundary_pts={len(bp4)}")
print(f"Stage3: pass_rate={pass_rate3:.3f}, speed_ceil={speed_ceil3:.1f}mph, boundary_pts={len(bp3)}")
print(f"Hausdorff S4vsS3 boundary: {hausdorff}")
print(f"Chamfer S4vsS3: {chamfer}")
print(f"Near-boundary fail prob S4: {near_fail4}")
print(f"Near-boundary fail prob S3: {near_fail3}")
print(f"Speed ceiling delta: {speed_ceil4 - speed_ceil3:+.1f} mph")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

labels = ['pass_rate', 'speed_ceiling (norm)', 'near_boundary_fail_prob']
s3_vals = [pass_rate3, speed_ceil3 / 75.0, near_fail3 if near_fail3 is not None else 0]
s4_vals = [pass_rate4, speed_ceil4 / 75.0, near_fail4 if near_fail4 is not None else 0]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - width/2, s3_vals, width, label='Stage3 (282k)')
b2 = ax.bar(x + width/2, s4_vals, width, label='Stage4 (312k)')

ax.set_ylabel('Value (speed ceiling normalized to 0-75mph)')
ax.set_title('Stage3 vs Stage4 SEMBAS Boundary Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.bar_label(b1, fmt='%.2f')
ax.bar_label(b2, fmt='%.2f')

plt.tight_layout()
plt.savefig(r'/mnt/c/Users/lucca/Documents/SEMBAS-RL/outputs/stage4_comparison_chart.png', dpi=150)
print("Chart saved.")
