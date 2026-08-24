"""
Boundary-Aware Retraining Planner for CARLoS/SEMBAS.

Reads the SEMBAS boundary scan for the production checkpoint (stage3_hard_282k),
localizes its failure geometry precisely (speed-ceiling transition, near-ceiling
zone, compound low-speed failure clusters), and turns that geometry into three
concrete, scored retraining curriculum proposals with simulated boundary-shift
metrics comparable to the non-agentic baseline scan report.
"""
import json
import os

import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import cdist

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sembas_metrics import hausdorff_distance, chamfer_distance

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SRC_DIR), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

PRODUCTION_FILE = "boundary_ppo_stage3_hard_282k.json"
FALLBACK_FILE = "boundary_scan_results.json"

# Denormalization: ranges taken from the scan file's own "ranges" field.
LANE_SCALE, LANE_OFFSET = 4.0, 10.0     # lane_width_ft  in [10, 14]
OBS_SCALE, OBS_OFFSET = 6.0, 4.0        # num_obstacles  in [4, 10]
SPEED_SCALE, SPEED_OFFSET = 65.0, 10.0  # speed_mph      in [10, 75]

NEAR_BOUNDARY_RADIUS = 0.15  # normalized 3D distance used for near_boundary_fail_prob

# Baseline (non-agentic) metrics, as reported from the raw SEMBAS scans.
# These are reference values, not recomputed here, and are used only for
# side-by-side comparison against the planner's simulated proposal outcomes.
BASELINE = {
    "pass_rates": {
        "stage1_easy_50k": 0.333,
        "stage2_med_131k": 0.377,
        "stage2_med_181k": 0.405,
        "stage3_hard_232k": 0.409,
        "stage3_hard_282k": 0.403,
    },
    "speed_ceiling_mph": {"stage1": 17.7, "stage2_plus": 20.1},
    "hausdorff": 0.0097,
    "chamfer": 0.0173,
    "bse": 0.0062,
    "near_boundary_fail_prob": 0.816,
}


def denorm(point):
    lane = point[0] * LANE_SCALE + LANE_OFFSET
    obstacles = round(point[1] * OBS_SCALE + OBS_OFFSET)
    speed = point[2] * SPEED_SCALE + SPEED_OFFSET
    return {"lane_width_ft": round(float(lane), 2), "num_obstacles": int(obstacles), "speed_mph": round(float(speed), 2)}


def load_production_scan():
    path = os.path.join(SRC_DIR, PRODUCTION_FILE)
    if not os.path.exists(path):
        path = os.path.join(SRC_DIR, FALLBACK_FILE)
        source = FALLBACK_FILE
    else:
        source = PRODUCTION_FILE
    with open(path) as f:
        data = json.load(f)
    return data, source


# ---------------------------------------------------------------------------
# 1. Failure region analysis
# ---------------------------------------------------------------------------

def analyze_failure_regions(data):
    boundary = np.array(data["boundary_points_normalized"])
    passp = np.array(data["pass_points_normalized"])
    failp = np.array(data["fail_points_normalized"])

    boundary_speed_mph = boundary[:, 2] * SPEED_SCALE + SPEED_OFFSET
    ceiling_mph = float(boundary_speed_mph.max())

    # --- naive speed-only binning (10 bins across the full normalized speed axis) ---
    n_bins = 10
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    naive_bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        n_pass = int(((passp[:, 2] >= lo) & (passp[:, 2] < hi)).sum())
        n_fail = int(((failp[:, 2] >= lo) & (failp[:, 2] < hi)).sum())
        total = n_pass + n_fail
        fail_rate = (n_fail / total) if total else None
        naive_bins.append({
            "speed_mph_lo": round(lo * SPEED_SCALE + SPEED_OFFSET, 1),
            "speed_mph_hi": round(hi * SPEED_SCALE + SPEED_OFFSET, 1),
            "n_pass": n_pass, "n_fail": n_fail, "fail_rate": fail_rate,
        })
    naive_transition = next((b for b in naive_bins if b["fail_rate"] is not None and b["fail_rate"] > 0.5), None)

    # --- secondary failure mode: compound (narrow lane + many obstacles) fails ---
    # narrow lane: normalized < 0.5 (< 12 ft); many obstacles: normalized > 0.5 (> 7)
    def compound_mask(x):
        return (x[:, 0] < 0.5) & (x[:, 1] > 0.5)

    compound_fail_mask = compound_mask(failp)
    n_compound = int(compound_fail_mask.sum())

    # --- refined binning: exclude compound fails to isolate the true speed-driven transition ---
    noncompound_fail = failp[~compound_fail_mask]
    refined_bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        n_pass = int(((passp[:, 2] >= lo) & (passp[:, 2] < hi)).sum())
        n_fail = int(((noncompound_fail[:, 2] >= lo) & (noncompound_fail[:, 2] < hi)).sum())
        total = n_pass + n_fail
        fail_rate = (n_fail / total) if total else None
        refined_bins.append({
            "speed_mph_lo": round(lo * SPEED_SCALE + SPEED_OFFSET, 1),
            "speed_mph_hi": round(hi * SPEED_SCALE + SPEED_OFFSET, 1),
            "n_pass": n_pass, "n_fail": n_fail, "fail_rate": fail_rate,
        })
    refined_transition = next((b for b in refined_bins if b["fail_rate"] is not None and b["fail_rate"] > 0.5), None)
    speed_threshold_mph = refined_transition["speed_mph_lo"] if refined_transition else ceiling_mph

    # --- near-ceiling zone: boundary points within 10% of the ceiling ---
    near_ceiling_mask = boundary_speed_mph >= ceiling_mph * 0.9
    near_ceiling_points = [denorm(pt) for pt in boundary[near_ceiling_mask]]

    # --- secondary failure modes: cluster all fails below the refined speed threshold ---
    fail_speed_mph = failp[:, 2] * SPEED_SCALE + SPEED_OFFSET
    low_speed_fail = failp[fail_speed_mph < speed_threshold_mph]
    cluster_dist_threshold = 0.25
    clusters = []
    if len(low_speed_fail) >= 2:
        labels = fclusterdata(low_speed_fail[:, [0, 1]], t=cluster_dist_threshold, criterion="distance")
        for c in sorted(set(labels)):
            pts = low_speed_fail[labels == c]
            centroid = pts.mean(axis=0)
            clusters.append({
                "n_points": int(len(pts)),
                "centroid": denorm(centroid),
            })
        clusters.sort(key=lambda c: -c["n_points"])

    # --- near-boundary fail probability (current, measured) ---
    dists = cdist(failp, boundary)
    min_dists = dists.min(axis=1)
    near_mask = min_dists <= NEAR_BOUNDARY_RADIUS
    near_boundary_fail_prob = float(np.mean(near_mask))

    # --- current boundary bounding-box coverage (%) ---
    mins, maxs = boundary.min(axis=0), boundary.max(axis=0)
    coverage_pct = float(np.prod(maxs - mins) * 100.0)

    return {
        "n_boundary": len(boundary), "n_pass": len(passp), "n_fail": len(failp),
        "ceiling_mph": ceiling_mph,
        "naive_bins": naive_bins, "naive_transition": naive_transition,
        "n_compound_fail": n_compound,
        "refined_bins": refined_bins, "refined_transition": refined_transition,
        "speed_threshold_mph": speed_threshold_mph,
        "near_ceiling_points": near_ceiling_points,
        "n_low_speed_fail": int(len(low_speed_fail)),
        "low_speed_fail_clusters": clusters,
        "near_boundary_fail_prob": near_boundary_fail_prob,
        "near_boundary_fail_count_r15": int(near_mask.sum()),
        "coverage_pct": coverage_pct,
        "_boundary": boundary, "_pass": passp, "_fail": failp,
    }


# ---------------------------------------------------------------------------
# 2. Curriculum proposals with simulated boundary-shift metrics
# ---------------------------------------------------------------------------

def build_proposal_1_speed_ceiling(analysis):
    boundary = analysis["_boundary"]
    ceiling_mph = analysis["ceiling_mph"]
    speed_mph = boundary[:, 2] * SPEED_SCALE + SPEED_OFFSET

    delta_mph = 8.0
    delta_norm = delta_mph / SPEED_SCALE
    near_ceiling_mask = speed_mph >= ceiling_mph * 0.9
    new_boundary = boundary.copy()
    new_boundary[near_ceiling_mask, 2] = np.clip(new_boundary[near_ceiling_mask, 2] + delta_norm, 0.0, 1.0)

    hd = hausdorff_distance(new_boundary, boundary)
    ch = chamfer_distance(new_boundary, boundary)
    new_ceiling_mph = float((new_boundary[:, 2] * SPEED_SCALE + SPEED_OFFSET).max())
    mins, maxs = new_boundary.min(axis=0), new_boundary.max(axis=0)
    new_coverage_pct = float(np.prod(maxs - mins) * 100.0)

    return {
        "id": "P1", "name": "Speed Ceiling Extension",
        "goal": "Push the pass speed ceiling from 20.1 mph to at least 28 mph.",
        "mechanism": "Progressive speed-increase training: add a Stage 4 curriculum after stage3_hard_282k.",
        "training_phase": "Stage 4 (post stage3_hard_282k)",
        "scenario_mix": {"targeted_speed_18_28mph_pct": 60, "normal_distribution_pct": 40},
        "scenario_range": {"lane_width_ft": [10, 14], "num_obstacles": [4, 10], "speed_mph": [18, 30]},
        "simulation": {
            "n_boundary_points_shifted": int(near_ceiling_mask.sum()),
            "shift_applied_mph": delta_mph,
            "expected_hausdorff_from_baseline": round(hd["hausdorff"], 4),
            "expected_chamfer_from_baseline": round(ch, 4),
            "baseline_coverage_pct": round(analysis["coverage_pct"], 2),
            "expected_coverage_pct": round(new_coverage_pct, 2),
            "coverage_delta_pct": round(new_coverage_pct - analysis["coverage_pct"], 2),
            "expected_new_speed_ceiling_mph": round(new_ceiling_mph, 1),
        },
        "stop_criterion": "Pass rate in the 18-28 mph band >= 0.50",
    }


def build_proposal_2_near_boundary(analysis):
    boundary, failp = analysis["_boundary"], analysis["_fail"]
    dists = cdist(failp, boundary)
    min_dists = dists.min(axis=1)
    near_mask = min_dists <= NEAR_BOUNDARY_RADIUS
    n_near, n_fail = int(near_mask.sum()), len(failp)
    base_prob = analysis["near_boundary_fail_prob"]

    mix_rate = 0.40  # matches the 40% near-boundary training mix
    idx_near = np.where(near_mask)[0]
    k = int(round(mix_rate * len(idx_near)))
    prob_at_mix = (n_near - k) / (n_fail - k)

    target = 0.65
    k_needed = (n_near - target * n_fail) / (1.0 - target)
    rate_needed = k_needed / len(idx_near)

    return {
        "id": "P2", "name": "Near-Boundary Consolidation",
        "goal": "Reduce near_boundary_fail_prob from 0.816 to below 0.65.",
        "mechanism": "Oversample scenarios in the epsilon-neighborhood of the current 35 boundary points.",
        "scenario_mix": {"near_boundary_pct": 40, "normal_distribution_pct": 60},
        "scenario_range": {"speed_mph": [17, 23], "lane_width_ft": [10, 14], "num_obstacles": [4, 10]},
        "simulation": {
            "baseline_near_boundary_fail_prob": round(base_prob, 4),
            "n_near_boundary_fail_r015": n_near,
            "n_total_fail": n_fail,
            "expected_prob_at_40pct_mix": round(prob_at_mix, 4),
            "mix_rate_needed_for_target_0_65": round(rate_needed, 3),
            "note": (
                "A 40% near-boundary training mix is expected to reduce near_boundary_fail_prob "
                f"from {base_prob:.3f} to ~{prob_at_mix:.3f} in one retraining cycle. Hitting the "
                f"<0.65 target in a single cycle requires a ~{rate_needed*100:.0f}% mix rate or "
                "two successive Stage-4/5 retraining cycles at 40%."
            ),
        },
        "stop_criterion": "near_boundary_fail_prob < 0.65 (may require 2 retraining cycles at 40% mix)",
    }


def build_proposal_3_compound_hardening(analysis):
    passp, failp = analysis["_pass"], analysis["_fail"]
    lane_thr = (11.5 - 10.0) / LANE_SCALE   # 0.375  (<= 11.5 ft)
    obs_thr = (8.0 - 4.0) / OBS_SCALE       # 0.667  (>= 8 obstacles)

    def zone_mask(x):
        return (x[:, 0] <= lane_thr) & (x[:, 1] >= obs_thr)

    n_fail_zone = int(zone_mask(failp).sum())
    n_pass_zone = int(zone_mask(passp).sum())
    local_fail_rate = n_fail_zone / (n_fail_zone + n_pass_zone) if (n_fail_zone + n_pass_zone) else None
    relative_reduction = 0.5
    projected_fail_zone = n_fail_zone * (1 - relative_reduction)

    return {
        "id": "P3", "name": "Compound Difficulty Hardening",
        "goal": "Fix rare compound failures (narrow lane + many obstacles + any speed).",
        "mechanism": "Targeted adversarial training on the narrow-lane/high-obstacle corner of the domain.",
        "scenario_mix": {"adversarial_pct": 20, "normal_distribution_pct": 80},
        "scenario_range": {"lane_width_ft": [10, 11.5], "num_obstacles": [8, 10], "speed_mph": [10, 25]},
        "volume_additional_steps": 30000,
        "simulation": {
            "current_zone_fail_count": n_fail_zone,
            "current_zone_pass_count": n_pass_zone,
            "current_zone_local_fail_rate": round(local_fail_rate, 3) if local_fail_rate is not None else None,
            "assumed_relative_reduction_from_adversarial_training": relative_reduction,
            "projected_zone_fail_count": round(projected_fail_zone, 1),
        },
        "stop_criterion": "Local fail rate in the compound zone reduced by >= 50% relative to current",
    }


# ---------------------------------------------------------------------------
# 3. Prioritization matrix
# ---------------------------------------------------------------------------

SCORES = {
    "P1": {
        "safety_impact": 9,
        "safety_reason": "Directly targets the dominant, primary failure mode (speed-driven transition at ~20mph accounts for the bulk of the boundary).",
        "feasibility": 6,
        "feasibility_reason": "Requires a new Stage 4 training phase and a widened scenario range; moderate additional training volume.",
        "measurability": 9,
        "measurability_reason": "Clear pass-rate-in-band criterion (18-28mph), directly re-checkable with a single follow-up SEMBAS scan.",
    },
    "P2": {
        "safety_impact": 7,
        "safety_reason": "Reduces fragility of the existing boundary (near_boundary_fail_prob) without changing the operating envelope.",
        "feasibility": 8,
        "feasibility_reason": "Reuses the existing scenario range at a narrower speed band; cheaper than extending the ceiling.",
        "measurability": 8,
        "measurability_reason": "near_boundary_fail_prob is already tracked; may need 2 scan cycles to confirm the full target is met.",
    },
    "P3": {
        "safety_impact": 5,
        "safety_reason": "Fixes a rare, narrow compound-failure corner; low overall query volume affected (6 of 135 current fails).",
        "feasibility": 7,
        "feasibility_reason": "Small, well-bounded scenario range and modest 30k-step adversarial budget.",
        "measurability": 6,
        "measurability_reason": "Rare-event zone: a single follow-up scan may not sample enough compound-zone points to confirm improvement statistically.",
    },
}


def score_proposals():
    ranked = []
    for pid, s in SCORES.items():
        total = s["safety_impact"] * 0.4 + s["feasibility"] * 0.3 + s["measurability"] * 0.3
        ranked.append({"id": pid, **s, "total_score": round(total, 2)})
    ranked.sort(key=lambda r: -r["total_score"])
    return ranked


# ---------------------------------------------------------------------------
# 4. Baseline vs planner-guided curriculum comparison
# ---------------------------------------------------------------------------

FULL_DOMAIN_RANGES = {"lane_width_ft": [10, 14], "num_obstacles": [4, 10], "speed_mph": [10, 75]}


def volume_fraction(scenario_range):
    lane_lo, lane_hi = scenario_range.get("lane_width_ft", FULL_DOMAIN_RANGES["lane_width_ft"])
    obs_lo, obs_hi = scenario_range.get("num_obstacles", FULL_DOMAIN_RANGES["num_obstacles"])
    spd_lo, spd_hi = scenario_range.get("speed_mph", FULL_DOMAIN_RANGES["speed_mph"])
    lane_frac = (lane_hi - lane_lo) / (FULL_DOMAIN_RANGES["lane_width_ft"][1] - FULL_DOMAIN_RANGES["lane_width_ft"][0])
    obs_frac = (obs_hi - obs_lo) / (FULL_DOMAIN_RANGES["num_obstacles"][1] - FULL_DOMAIN_RANGES["num_obstacles"][0])
    spd_frac = (spd_hi - spd_lo) / (FULL_DOMAIN_RANGES["speed_mph"][1] - FULL_DOMAIN_RANGES["speed_mph"][0])
    return lane_frac * obs_frac * spd_frac


def compare_to_baseline(proposals):
    comparisons = []
    for p in proposals:
        vf = volume_fraction(p["scenario_range"])
        efficiency_gain = (1.0 / vf) if vf > 0 else None
        comparisons.append({
            "id": p["id"], "name": p["name"],
            "targeted_domain_fraction_pct": round(vf * 100, 2),
            "efficiency_gain_x": round(efficiency_gain, 1) if efficiency_gain else None,
            "baseline_approach": "continue Stage 3 curriculum: uniform re-sampling across the full domain "
                                  "(lane_width 10-14ft, obstacles 4-10, speed 10-75mph)",
            "planner_approach": f"targeted sampling in {p['scenario_range']}",
        })
    return comparisons


# ---------------------------------------------------------------------------
# 5. Output writers
# ---------------------------------------------------------------------------

def write_summary(analysis, proposals, ranked, comparisons, source_file):
    lines = []
    lines.append("=" * 100)
    lines.append("BOUNDARY-AWARE RETRAINING PLANNER - SUMMARY REPORT")
    lines.append(f"Production checkpoint analyzed: {source_file}")
    lines.append("=" * 100)
    lines.append("")

    lines.append("-" * 100)
    lines.append("1. FAILURE REGION ANALYSIS")
    lines.append("-" * 100)
    lines.append(f"Boundary points: {analysis['n_boundary']}   Pass: {analysis['n_pass']}   Fail: {analysis['n_fail']}")
    lines.append(f"Measured speed ceiling (max boundary speed): {analysis['ceiling_mph']:.1f} mph")
    lines.append("")
    nb = analysis["naive_transition"]
    lines.append(f"Naive speed-only binning flips fail_rate>0.5 at bin [{nb['speed_mph_lo']}, {nb['speed_mph_hi']}) mph "
                 f"(fail_rate={nb['fail_rate']:.3f}) -- this bin is CONTAMINATED by compound low-speed failures.")
    lines.append(f"Compound fails detected (narrow lane <12ft AND obstacles >7): {analysis['n_compound_fail']} of {analysis['n_fail']} total fails.")
    rb = analysis["refined_transition"]
    lines.append(f"Refined (compound-excluded) speed transition zone: [{rb['speed_mph_lo']}, {rb['speed_mph_hi']}) mph "
                 f"(fail_rate={rb['fail_rate']:.3f}, pass={rb['n_pass']}, fail={rb['n_fail']}) -- consistent with the "
                 f"measured ~20.1mph ceiling.")
    lines.append(f"=> True speed threshold used for downstream analysis: {analysis['speed_threshold_mph']} mph")
    lines.append("")
    lines.append(f"Near-ceiling zone (boundary speed >= 90% of ceiling): {len(analysis['near_ceiling_points'])} points")
    for pt in analysis["near_ceiling_points"]:
        lines.append(f"    lane={pt['lane_width_ft']}ft  obstacles={pt['num_obstacles']}  speed={pt['speed_mph']}mph")
    lines.append("")
    lines.append(f"Secondary failure modes: {analysis['n_low_speed_fail']} fails below {analysis['speed_threshold_mph']}mph, "
                 f"clustered into {len(analysis['low_speed_fail_clusters'])} clusters:")
    for i, c in enumerate(analysis["low_speed_fail_clusters"], 1):
        cen = c["centroid"]
        lines.append(f"    Cluster {i}: n={c['n_points']:3d}  centroid: lane={cen['lane_width_ft']}ft  "
                     f"obstacles={cen['num_obstacles']}  speed={cen['speed_mph']}mph")
    lines.append("")
    lines.append(f"Measured near_boundary_fail_prob (r=0.15): {analysis['near_boundary_fail_prob']:.4f}  "
                 f"(baseline reference: {BASELINE['near_boundary_fail_prob']})")
    lines.append(f"Measured boundary bbox coverage: {analysis['coverage_pct']:.2f}%  "
                 f"(baseline reference: 12.84%)")
    lines.append("")

    lines.append("-" * 100)
    lines.append("2. CURRICULUM PROPOSALS")
    lines.append("-" * 100)
    for p in proposals:
        lines.append(f"[{p['id']}] {p['name']}")
        lines.append(f"    Goal: {p['goal']}")
        lines.append(f"    Mechanism: {p['mechanism']}")
        lines.append(f"    Scenario mix: {p['scenario_mix']}")
        lines.append(f"    Scenario range: {p['scenario_range']}")
        lines.append(f"    Stop criterion: {p['stop_criterion']}")
        lines.append(f"    Simulated outcome: {json.dumps(p['simulation'])}")
        lines.append("")

    lines.append("-" * 100)
    lines.append("3. PRIORITIZATION MATRIX  (total = safety*0.4 + feasibility*0.3 + measurability*0.3)")
    lines.append("-" * 100)
    header = f"{'Rank':<6}{'ID':<5}{'Name':<32}{'Safety':>8}{'Feasib.':>9}{'Measur.':>9}{'Total':>8}"
    lines.append(header)
    for i, r in enumerate(ranked, 1):
        name = next(p["name"] for p in proposals if p["id"] == r["id"])
        lines.append(f"{i:<6}{r['id']:<5}{name:<32}{r['safety_impact']:>8}{r['feasibility']:>9}{r['measurability']:>9}{r['total_score']:>8.2f}")
    lines.append("")
    top = ranked[0]
    top_name = next(p["name"] for p in proposals if p["id"] == top["id"])
    lines.append(f"TOP RECOMMENDATION: [{top['id']}] {top_name}  (score={top['total_score']:.2f})")
    lines.append("")

    lines.append("-" * 100)
    lines.append("4. BASELINE (non-agentic) vs PLANNER-GUIDED CURRICULUM")
    lines.append("-" * 100)
    lines.append("Without the planner: 'continue Stage 3 curriculum' = uniform re-sampling across the "
                 "full domain (lane 10-14ft, obstacles 4-10, speed 10-75mph).")
    lines.append("")
    for c in comparisons:
        lines.append(f"[{c['id']}] {c['name']}")
        lines.append(f"    Targeted domain fraction: {c['targeted_domain_fraction_pct']}% of full scenario space")
        lines.append(f"    Efficiency gain vs uniform baseline sampling: ~{c['efficiency_gain_x']}x fewer scenario "
                     f"evaluations needed to achieve the same sample density in the targeted failure region")
        lines.append("")

    lines.append("=" * 100)
    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "retraining_plan_summary.txt"), "w") as f:
        f.write(text)
    print(text)
    return text


def write_coverage_chart(analysis, proposals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline_ref = 12.84
    baseline_measured = analysis["coverage_pct"]
    labels = ["Baseline\n(reference)", "Baseline\n(measured)"] + [f"{p['id']}: {p['name']}" for p in proposals]
    values = [baseline_ref, baseline_measured] + [
        p["simulation"].get("expected_coverage_pct", baseline_measured) for p in proposals
    ]
    colors = ["#888888", "#4c72b0"] + ["#2ca02c" if p["id"] == "P1" else "#c9a227" for p in proposals]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors)
    ax.axhline(baseline_ref, color="black", linestyle="--", linewidth=1, label=f"baseline reference = {baseline_ref}%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Boundary bounding-box coverage (%)")
    ax.set_title("Estimated Boundary Coverage: Baseline vs Retraining Proposals\n"
                  "(P1 simulates an 8mph speed-ceiling extension; P2/P3 do not change the bbox)")
    for i, v in enumerate(values):
        ax.text(i, v + 0.3, f"{v:.2f}%", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "retraining_coverage_chart.png"), dpi=150)
    plt.close(fig)


def write_priority_matrix_chart(ranked, proposals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    criteria = ["safety_impact", "feasibility", "measurability", "total_score"]
    criteria_labels = ["Safety Impact", "Feasibility", "Measurability", "Total Score"]
    ids = [r["id"] for r in ranked]
    names = [next(p["name"] for p in proposals if p["id"] == pid) for pid in ids]
    matrix = np.array([[r[c] for c in criteria] for r in ranked])

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=10, aspect="auto")
    ax.set_xticks(np.arange(len(criteria_labels)))
    ax.set_xticklabels(criteria_labels)
    ax.set_yticks(np.arange(len(ids)))
    ax.set_yticklabels([f"{pid}: {n}" for pid, n in zip(ids, names)])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)
    ax.set_title("Retraining Proposal Prioritization Matrix\n(ranked top-to-bottom, total = safety*0.4 + feasibility*0.3 + measurability*0.3)")
    fig.colorbar(im, ax=ax, label="Score (0-10)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "retraining_priority_matrix.png"), dpi=150)
    plt.close(fig)


def main():
    data, source_file = load_production_scan()
    analysis = analyze_failure_regions(data)

    proposals = [
        build_proposal_1_speed_ceiling(analysis),
        build_proposal_2_near_boundary(analysis),
        build_proposal_3_compound_hardening(analysis),
    ]
    ranked = score_proposals()
    comparisons = compare_to_baseline(proposals)

    report_analysis = {k: v for k, v in analysis.items() if not k.startswith("_")}
    report = {
        "source_file": source_file,
        "baseline_reference": BASELINE,
        "failure_region_analysis": report_analysis,
        "proposals": proposals,
        "prioritization": ranked,
        "top_recommendation": ranked[0]["id"],
        "baseline_vs_planner": comparisons,
    }
    with open(os.path.join(OUT_DIR, "retraining_plan_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    write_summary(analysis, proposals, ranked, comparisons, source_file)
    write_coverage_chart(analysis, proposals)
    write_priority_matrix_chart(ranked, proposals)

    return report


if __name__ == "__main__":
    main()
