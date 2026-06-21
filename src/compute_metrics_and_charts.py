"""
compute_metrics_and_charts.py

Loads real SEMBAS boundary scan data, computes all metrics from
sembas_metrics.py, and generates 4 publication-quality charts.
"""

import json
import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
DATA_FILE   = os.path.join(SCRIPT_DIR, "boundary_scan_results.json")
METRICS_MOD = SCRIPT_DIR  # sembas_metrics.py lives here
OUTPUT_DIR  = os.path.join(REPO_ROOT, "outputs")

sys.path.insert(0, METRICS_MOD)
from sembas_metrics import (
    hausdorff_distance,
    chamfer_distance,
    boundary_sampling_efficiency,
    near_boundary_failure_probability,
    boundary_drift,
    coverage_efficiency,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(DATA_FILE) as f:
    data = json.load(f)

bp  = np.array(data["boundary_points_normalized"])   # (35, 3)
pp  = np.array(data["pass_points_normalized"])        # (91, 3)
fp  = np.array(data["fail_points_normalized"])        # (141, 3)

print(f"Loaded  boundary={bp.shape[0]}  pass={pp.shape[0]}  fail={fp.shape[0]}")

# Domain in normalised [0,1]^3 space
domain_bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

# ---------------------------------------------------------------------------
# Metric 1 & 2 — Hausdorff / Chamfer
# We treat the combined pass+fail cloud as a proxy reference set.
# The boundary should sit between them; distance to pass_points measures
# how far the boundary is from the known-safe zone.
# ---------------------------------------------------------------------------
reference = pp  # pass points as reference for accuracy metrics

hd_result = hausdorff_distance(bp, reference)
ch_result  = chamfer_distance(bp, reference)

print(f"Hausdorff  : {hd_result['hausdorff']:.4f}")
print(f"Chamfer    : {ch_result:.4f}")

# ---------------------------------------------------------------------------
# Metric 3 — Boundary Sampling Efficiency
# non-boundary = pass + fail queries
# ---------------------------------------------------------------------------
non_boundary = list(range(len(pp) + len(fp)))
bse = boundary_sampling_efficiency(list(range(len(bp))), non_boundary)
print(f"BSE        : {bse:.2f} %")

# ---------------------------------------------------------------------------
# Metric 4 — Near-boundary failure probability
# Build a simple KNN oracle from the labelled pass/fail data, then probe
# each boundary point's epsilon-neighbourhood.
# ---------------------------------------------------------------------------
all_pts    = np.vstack([pp, fp])
all_labels = np.array([True] * len(pp) + [False] * len(fp))  # True=pass

def knn_oracle(x: np.ndarray, k: int = 5) -> bool:
    """Classify x as pass/fail via k-NN on the labelled dataset."""
    dists = np.linalg.norm(all_pts - x, axis=1)
    k_idx = np.argsort(dists)[:k]
    return bool(all_labels[k_idx].sum() > k / 2)

dummy_normals = np.zeros_like(bp)
nbfp = near_boundary_failure_probability(
    knn_oracle,
    bp,
    dummy_normals,
    epsilon=0.08,
    n_samples=30,
    seed=42,
)
print(f"Near-boundary fail prob : {nbfp:.4f}")

# ---------------------------------------------------------------------------
# Metric 5 — Boundary drift (first-half vs second-half of boundary points)
# Meaningful measure of internal consistency / spread.
# ---------------------------------------------------------------------------
half = len(bp) // 2
drift_result = boundary_drift(bp[:half], bp[half:])
print(f"Boundary drift (Chamfer): {drift_result['chamfer_drift']:.4f}")
print(f"Boundary drift (Hausdorff): {drift_result['hausdorff_drift']:.4f}")

# ---------------------------------------------------------------------------
# Metric 6 — Coverage efficiency
# ---------------------------------------------------------------------------
cov = coverage_efficiency(bp, domain_bounds, grid_resolution=0.075)
print(f"Coverage efficiency     : {cov['coverage_efficiency']:.2f} %")
print(f"  cells covered         : {cov['cells_covered']} / {cov['total_cells']}")

# ---------------------------------------------------------------------------
# Save metrics JSON
# ---------------------------------------------------------------------------
metrics_out = {
    "hausdorff": hd_result["hausdorff"],
    "hausdorff_A_to_B": hd_result["A_to_B"],
    "hausdorff_B_to_A": hd_result["B_to_A"],
    "chamfer": ch_result,
    "boundary_sampling_efficiency_pct": bse,
    "near_boundary_failure_prob": nbfp,
    "boundary_drift": drift_result,
    "coverage": cov,
}
metrics_path = os.path.join(OUTPUT_DIR, "metrics_real.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"\nMetrics saved -> {metrics_path}")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_PASS  = "#2196F3"   # blue
C_FAIL  = "#F44336"   # red
C_BOUND = "#FFC107"   # gold

ALPHA_SCATTER = 0.65
S_PASS  = 28
S_FAIL  = 28
S_BOUND = 60

# ============================================================================
# CHART 1 — 3D Scatter: full scenario space
# ============================================================================
print("\nRendering Chart 1 — 3D scatter …")
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")

ax.scatter(pp[:, 0], pp[:, 1], pp[:, 2],
           c=C_PASS, s=S_PASS, alpha=ALPHA_SCATTER, label=f"Pass ({len(pp)})", depthshade=True)
ax.scatter(fp[:, 0], fp[:, 1], fp[:, 2],
           c=C_FAIL, s=S_FAIL, alpha=ALPHA_SCATTER, label=f"Fail ({len(fp)})", depthshade=True)
ax.scatter(bp[:, 0], bp[:, 1], bp[:, 2],
           c=C_BOUND, s=S_BOUND, alpha=0.95, edgecolors="k", linewidths=0.5,
           label=f"Boundary ({len(bp)})", depthshade=False, zorder=5)

ax.set_xlabel("lane_width_ft (norm.)")
ax.set_ylabel("num_obstacles (norm.)")
ax.set_zlabel("speed_mph (norm.)")
ax.set_title("SEMBAS Boundary Scan — 3D Scenario Space", fontsize=13, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.view_init(elev=22, azim=-55)

plt.tight_layout()
chart1 = os.path.join(OUTPUT_DIR, "chart1_3d_scatter.png")
plt.savefig(chart1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved -> {chart1}")

# ============================================================================
# CHART 2 — 2D slice: lane_width (x-axis) vs speed (y-axis)
# (projected; num_obstacles dimension collapsed)
# ============================================================================
print("Rendering Chart 2 — lane_width vs speed …")
fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(pp[:, 0], pp[:, 2], c=C_PASS, s=S_PASS, alpha=ALPHA_SCATTER, label=f"Pass ({len(pp)})")
ax.scatter(fp[:, 0], fp[:, 2], c=C_FAIL, s=S_FAIL, alpha=ALPHA_SCATTER, label=f"Fail ({len(fp)})")
ax.scatter(bp[:, 0], bp[:, 2], c=C_BOUND, s=S_BOUND, alpha=0.95,
           edgecolors="k", linewidths=0.6, label=f"Boundary ({len(bp)})", zorder=5)

ax.set_xlabel("lane_width_ft  [norm 0-1 = 10-14 ft]", fontsize=11)
ax.set_ylabel("speed_mph  [norm 0-1 = 10-75 mph]", fontsize=11)
ax.set_title("2D Slice: Lane Width vs Speed\n(all num_obstacles values projected)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["10 ft\n(0.0)", "11 ft\n(0.25)", "12 ft\n(0.5)", "13 ft\n(0.75)", "14 ft\n(1.0)"], fontsize=8)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["10 mph\n(0.0)", "26 mph\n(0.25)", "43 mph\n(0.5)", "59 mph\n(0.75)", "75 mph\n(1.0)"], fontsize=8)
# Annotate the clear speed constraint (boundary clusters near low speed)
ax.axhline(y=0.12, color="gold", lw=1.5, ls="--", alpha=0.7)

plt.tight_layout()
chart2 = os.path.join(OUTPUT_DIR, "chart2_lane_vs_speed.png")
plt.savefig(chart2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved -> {chart2}")

# ============================================================================
# CHART 3 — 2D slice: num_obstacles (x-axis) vs speed (y-axis)
# ============================================================================
print("Rendering Chart 3 — obstacles vs speed …")
fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(pp[:, 1], pp[:, 2], c=C_PASS, s=S_PASS, alpha=ALPHA_SCATTER, label=f"Pass ({len(pp)})")
ax.scatter(fp[:, 1], fp[:, 2], c=C_FAIL, s=S_FAIL, alpha=ALPHA_SCATTER, label=f"Fail ({len(fp)})")
ax.scatter(bp[:, 1], bp[:, 2], c=C_BOUND, s=S_BOUND, alpha=0.95,
           edgecolors="k", linewidths=0.6, label=f"Boundary ({len(bp)})", zorder=5)

ax.set_xlabel("num_obstacles  [norm 0-1 = 4-10]", fontsize=11)
ax.set_ylabel("speed_mph  [norm 0-1 = 10-75 mph]", fontsize=11)
ax.set_title("2D Slice: Num Obstacles vs Speed\n(all lane_width values projected)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["4\n(0.0)", "5.5\n(0.25)", "7\n(0.5)", "8.5\n(0.75)", "10\n(1.0)"], fontsize=8)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["10 mph\n(0.0)", "26 mph\n(0.25)", "43 mph\n(0.5)", "59 mph\n(0.75)", "75 mph\n(1.0)"], fontsize=8)

plt.tight_layout()
chart3 = os.path.join(OUTPUT_DIR, "chart3_obstacles_vs_speed.png")
plt.savefig(chart3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved -> {chart3}")

# ============================================================================
# CHART 4 — Metrics bar chart
# ============================================================================
print("Rendering Chart 4 — metrics bar chart …")

metric_labels = [
    "Hausdorff\n(boundary->pass)",
    "Chamfer\n(boundary->pass)",
    "BSE\n(%)",
    "Near-boundary\nFail Prob",
    "Boundary Drift\n(Chamfer)",
    "Coverage\nEfficiency (%)",
]
metric_values = [
    hd_result["hausdorff"],
    ch_result,
    bse,
    nbfp,
    drift_result["chamfer_drift"],
    cov["coverage_efficiency"],
]

colours = ["#9C27B0", "#3F51B5", "#009688", "#FF5722", "#795548", "#607D8B"]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(metric_labels, metric_values, color=colours, edgecolor="white",
              linewidth=0.8, width=0.55)

for bar, val in zip(bars, metric_values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(metric_values) * 0.015,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Metric Value", fontsize=11)
ax.set_title("SEMBAS Run Metrics — Real Boundary Scan Data", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(metric_values) * 1.15)

# Annotate note about BSE and Coverage being %
ax.text(0.99, 0.97, "BSE & Coverage Efficiency in %; others in normalised [0,1] units",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color="gray", style="italic")

plt.tight_layout()
chart4 = os.path.join(OUTPUT_DIR, "chart4_metrics_bar.png")
plt.savefig(chart4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved -> {chart4}")

# ---------------------------------------------------------------------------
print("\n=== All done ===")
print(f"Charts: {OUTPUT_DIR}")
print(json.dumps(metrics_out, indent=2))
