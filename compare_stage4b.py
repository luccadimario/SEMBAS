import json
import numpy as np
from scipy.spatial.distance import cdist

with open('/mnt/c/Users/lucca/Documents/SEMBAS-RL/src/boundary_ppo_stage4b_speed_ext_150k.json') as f:
    s4b = json.load(f)

with open('/mnt/c/Users/lucca/Documents/SEMBAS-RL/src/boundary_ppo_stage3_hard_282k.json') as f:
    s3 = json.load(f)

def analyze(data, name):
    bp = np.array(data.get('boundary_points_normalized', []))
    pp = np.array(data.get('pass_points_normalized', []))
    fp = np.array(data.get('fail_points_normalized', []))
    total = len(pp) + len(fp)
    pass_rate = len(pp) / total if total > 0 else 0
    speed_ceil = float(pp[:, 2].max() * 65 + 10) if len(pp) > 0 else 0
    speed_mean_pass = float(pp[:, 2].mean() * 65 + 10) if len(pp) > 0 else 0

    if len(bp) > 0 and len(fp) > 0:
        df = cdist(fp, bp)
        near_fail = float((df.min(axis=1) < 0.1).mean())
    else:
        near_fail = None

    print(f"\n=== {name} ===")
    print(f"  Total queries: {total}")
    print(f"  Pass: {len(pp)}, Fail: {len(fp)}, Boundary: {len(bp)}")
    print(f"  Pass rate: {pass_rate:.3f} ({pass_rate*100:.1f}%)")
    print(f"  Speed ceiling: {speed_ceil:.1f} mph")
    print(f"  Mean pass speed: {speed_mean_pass:.1f} mph")
    print(f"  Near-boundary fail prob: {near_fail:.3f}" if near_fail is not None else "  Near-boundary fail prob: N/A")
    return bp, pp, fp, pass_rate, speed_ceil, near_fail

bp3, pp3, fp3, pr3, sc3, nbf3 = analyze(s3, "Stage3_Hard_282k (BASELINE)")
bp4b, pp4b, fp4b, pr4b, sc4b, nbf4b = analyze(s4b, "Stage4b_150k (NEW)")

hausdorff = None
chamfer = None
if len(bp3) > 0 and len(bp4b) > 0:
    d = cdist(bp4b, bp3)
    hausdorff = float(max(d.min(axis=1).max(), d.min(axis=0).max()))
    chamfer = float(d.min(axis=1).mean() + d.min(axis=0).mean())
    print(f"\n=== Boundary Shift (Stage4b vs Stage3) ===")
    print(f"  Hausdorff distance: {hausdorff:.4f}")
    print(f"  Chamfer distance: {chamfer:.4f}")
    print(f"  Pass rate change: {pr4b - pr3:+.3f} ({(pr4b-pr3)*100:+.1f}pp)")
    print(f"  Speed ceiling change: {sc4b - sc3:+.1f} mph")

result = {
    "stage3_baseline": {"pass_rate": pr3, "speed_ceiling_mph": sc3, "near_bnd_fail_prob": nbf3, "boundary_pts": len(bp3)},
    "stage4b_new": {"pass_rate": pr4b, "speed_ceiling_mph": sc4b, "near_bnd_fail_prob": nbf4b, "boundary_pts": len(bp4b)},
    "delta": {"pass_rate": pr4b - pr3, "speed_ceiling_mph": sc4b - sc3},
    "hausdorff": hausdorff,
    "chamfer": chamfer,
}
with open('/mnt/c/Users/lucca/Documents/SEMBAS-RL/outputs/stage4b_comparison.json', 'w') as f:
    json.dump(result, f, indent=2)
print("\nSaved to outputs/stage4b_comparison.json")
