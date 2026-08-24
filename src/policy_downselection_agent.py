"""
Policy Down-Selection Agent for CARLoS/SEMBAS.

Applies formal safety gates to boundary-scan results from 5 PPO checkpoints
and produces a structured ADVANCE / CONDITIONAL / ELIMINATE verdict for each,
rather than naively picking the checkpoint with the highest pass rate.
"""
import json
import os
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cdist

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SRC_DIR), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CHECKPOINTS = [
    ("stage1_easy_50k", "boundary_ppo_stage1_easy_50k.json"),
    ("stage2_med_131k", "boundary_ppo_stage2_med_131k.json"),
    ("stage2_med_181k", "boundary_ppo_stage2_med_181k.json"),
    ("stage3_hard_232k", "boundary_ppo_stage3_hard_232k.json"),
    ("stage3_hard_282k", "boundary_ppo_stage3_hard_282k.json"),
]

# Baseline (non-agentic) metrics reported from the raw scans, for comparison only.
BASELINE = {
    "stage1_easy_50k":  {"pass_rate": 0.333, "max_speed": 17.7, "mean_speed": 13.1},
    "stage2_med_131k":  {"pass_rate": 0.379, "max_speed": 20.1, "mean_speed": 14.7},
    "stage2_med_181k":  {"pass_rate": 0.394, "max_speed": 20.1, "mean_speed": 14.3},
    "stage3_hard_232k": {"pass_rate": 0.398, "max_speed": 20.1, "mean_speed": 14.3},
    "stage3_hard_282k": {"pass_rate": 0.403, "max_speed": 20.1, "mean_speed": 14.4},
}

SPEED_DIM = 2          # index of speed_mph in dimensions
SPEED_SCALE = 65.0      # ranges['speed_mph'] = [10, 75] -> norm * 65 + 10
SPEED_OFFSET = 10.0
NEAR_BOUNDARY_RADIUS = 0.15  # normalized-distance threshold for "close to boundary"

# ---------------------------------------------------------------------------
# Safety gates. Thresholds are deliberately conservative and justified below;
# they are independent of which checkpoint happens to score highest.
# ---------------------------------------------------------------------------
GATES = {
    "G1_speed_ceiling": {
        "desc": "Boundary speed ceiling >= 18.0 mph",
        "justification": (
            "18 mph is the minimum operational speed CARLoS must sustain up to "
            "the boundary of its safe envelope to be useful on the target course; "
            "a policy that only holds together below this speed is not deployable "
            "regardless of how well it scores on pass rate."
        ),
        "threshold": 18.0,
        "compare": "gte",
        "metric": "boundary_speed_ceiling",
    },
    "G2_pass_rate": {
        "desc": "Pass rate >= 35% of sampled test space",
        "justification": (
            "35% is the pass rate of the weakest checkpoint that has historically "
            "been considered viable in this project; it is a floor, not a target, "
            "so it screens out policies that regressed rather than rewarding the "
            "single highest scorer."
        ),
        "threshold": 0.35,
        "compare": "gte",
        "metric": "pass_rate",
    },
    "G3_boundary_stability": {
        "desc": "Boundary drift (std of boundary speed) <= 4.0 mph",
        "justification": (
            "A tight, consistent boundary speed (low std-dev across boundary points) "
            "means the failure envelope is predictable; a policy whose boundary speed "
            "swings widely is unsafe to certify at any single speed limit even if its "
            "mean/max look good."
        ),
        "threshold": 4.0,
        "compare": "lte",
        "metric": "boundary_drift",
    },
    "G4_near_boundary_safety": {
        "desc": "Fraction of fail points within 0.15 (normalized) of the boundary <= 0.85",
        "justification": (
            "If almost all failures cluster immediately adjacent to the boundary, the "
            "safe margin is razor-thin and small perturbations (sensor noise, actuation "
            "delay) push the vehicle from pass to fail; capping this fraction ensures "
            "some fail points are safely far from the edge, i.e. the envelope has real "
            "margin rather than being a knife-edge."
        ),
        "threshold": 0.85,
        "compare": "lte",
        "metric": "near_boundary_fail_prob",
    },
    "G5_query_efficiency": {
        "desc": "Query efficiency (boundary points / n_queries) >= 0.12",
        "justification": (
            "This checks that the boundary search actually found and resolved a "
            "reasonably dense boundary given its query budget; low efficiency means "
            "the scan under-sampled the boundary and the reported metrics for that "
            "checkpoint are statistically less trustworthy."
        ),
        "threshold": 0.12,
        "compare": "gte",
        "metric": "query_efficiency",
    },
}


def load_scan(filename):
    path = os.path.join(SRC_DIR, filename)
    with open(path) as f:
        return json.load(f)


def compute_metrics(data):
    boundary = np.array(data["boundary_points_normalized"])
    passp = np.array(data["pass_points_normalized"])
    failp = np.array(data["fail_points_normalized"])
    n_queries = data["n_queries"]

    n_pass, n_fail = len(passp), len(failp)
    pass_rate = n_pass / (n_pass + n_fail) if (n_pass + n_fail) > 0 else 0.0

    boundary_speed_norm = boundary[:, SPEED_DIM]
    boundary_speed_mph = boundary_speed_norm * SPEED_SCALE + SPEED_OFFSET

    boundary_speed_ceiling = float(np.max(boundary_speed_mph))
    boundary_speed_mean = float(np.mean(boundary_speed_mph))
    boundary_drift = float(np.std(boundary_speed_norm) * SPEED_SCALE)

    if len(failp) > 0 and len(boundary) > 0:
        dists = cdist(failp, boundary)
        min_dists = dists.min(axis=1)
        near_boundary_fail_prob = float(np.mean(min_dists <= NEAR_BOUNDARY_RADIUS))
    else:
        near_boundary_fail_prob = 0.0

    if len(boundary) > 0:
        mins = boundary.min(axis=0)
        maxs = boundary.max(axis=0)
        boundary_coverage = float(np.prod(maxs - mins))
    else:
        boundary_coverage = 0.0

    query_efficiency = len(boundary) / n_queries if n_queries else 0.0

    return {
        "n_queries": n_queries,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_boundary": len(boundary),
        "pass_rate": pass_rate,
        "boundary_speed_ceiling": boundary_speed_ceiling,
        "boundary_speed_mean": boundary_speed_mean,
        "boundary_drift": boundary_drift,
        "near_boundary_fail_prob": near_boundary_fail_prob,
        "boundary_coverage": boundary_coverage,
        "query_efficiency": query_efficiency,
    }


def evaluate_gates(metrics):
    results = {}
    for gate_id, gate in GATES.items():
        value = metrics[gate["metric"]]
        threshold = gate["threshold"]
        passed = value >= threshold if gate["compare"] == "gte" else value <= threshold
        results[gate_id] = {
            "passed": bool(passed),
            "value": value,
            "threshold": threshold,
            "compare": gate["compare"],
        }
    return results


def verdict_from_gates(gate_results):
    n_passed = sum(1 for g in gate_results.values() if g["passed"])
    n_total = len(gate_results)
    if n_passed == n_total:
        verdict = "ADVANCE"
    elif n_passed == n_total - 1:
        verdict = "CONDITIONAL"
    else:
        verdict = "ELIMINATE"
    return verdict, n_passed, n_total


def main():
    report = {"gates": {gid: {k: v for k, v in g.items() if k != "metric"} | {"metric": g["metric"]}
                        for gid, g in GATES.items()},
              "checkpoints": {}}

    rows = []  # for chart + summary table
    for name, filename in CHECKPOINTS:
        data = load_scan(filename)
        metrics = compute_metrics(data)
        gate_results = evaluate_gates(metrics)
        verdict, n_passed, n_total = verdict_from_gates(gate_results)

        report["checkpoints"][name] = {
            "metrics": metrics,
            "gates": gate_results,
            "gates_passed": n_passed,
            "gates_total": n_total,
            "verdict": verdict,
        }
        rows.append((name, metrics, gate_results, verdict, n_passed, n_total))

    # naive baseline verdict: highest pass_rate wins, always
    naive_pick = max(rows, key=lambda r: r[1]["pass_rate"])[0]
    agent_advance = [r[0] for r in rows if r[3] == "ADVANCE"]
    agent_conditional = [r[0] for r in rows if r[3] == "CONDITIONAL"]
    agent_eliminate = [r[0] for r in rows if r[3] == "ELIMINATE"]

    report["naive_baseline_pick"] = naive_pick
    report["agent_advance"] = agent_advance
    report["agent_conditional"] = agent_conditional
    report["agent_eliminate"] = agent_eliminate
    report["agent_vs_naive_agree"] = (
        naive_pick in agent_advance if agent_advance else False
    )

    with open(os.path.join(OUT_DIR, "policy_downselection_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    write_summary(rows, naive_pick)
    write_chart(rows)

    return rows, naive_pick


def write_summary(rows, naive_pick):
    lines = []
    lines.append("=" * 100)
    lines.append("POLICY DOWN-SELECTION AGENT - SUMMARY REPORT")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Safety gates applied (independent of raw pass-rate ranking):")
    for gid, gate in GATES.items():
        op = ">=" if gate["compare"] == "gte" else "<="
        lines.append(f"  {gid}: {gate['desc']}  ({gate['metric']} {op} {gate['threshold']})")
        lines.append(f"      Why: {gate['justification']}")
    lines.append("")
    lines.append("-" * 100)
    lines.append("PER-CHECKPOINT METRICS")
    lines.append("-" * 100)
    header = f"{'Checkpoint':<20}{'PassRate':>10}{'SpeedCeil':>11}{'SpeedMean':>11}{'Drift':>8}{'NearBFail':>11}{'QueryEff':>10}"
    lines.append(header)
    for name, m, gr, verdict, n_passed, n_total in rows:
        lines.append(
            f"{name:<20}{m['pass_rate']*100:>9.1f}%{m['boundary_speed_ceiling']:>10.1f}m"
            f"{m['boundary_speed_mean']:>10.1f}m{m['boundary_drift']:>7.1f}m"
            f"{m['near_boundary_fail_prob']*100:>10.1f}%{m['query_efficiency']:>10.3f}"
        )
    lines.append("")
    lines.append("-" * 100)
    lines.append("GATE PASS/FAIL TABLE  (P = pass, F = FAIL)")
    lines.append("-" * 100)
    gate_ids = list(GATES.keys())
    header2 = f"{'Checkpoint':<20}" + "".join(f"{gid.split('_')[0]:>6}" for gid in gate_ids) + f"{'Score':>8}{'Verdict':>14}"
    lines.append(header2)
    for name, m, gr, verdict, n_passed, n_total in rows:
        row = f"{name:<20}"
        for gid in gate_ids:
            row += f"{'P' if gr[gid]['passed'] else 'F':>6}"
        score_str = f"{n_passed}/{n_total}"
        row += f"{score_str:>8}{verdict:>14}"
        lines.append(row)
    lines.append("")
    lines.append("-" * 100)
    lines.append("BASELINE (non-agentic) vs AGENT VERDICT COMPARISON")
    lines.append("-" * 100)
    lines.append(f"{'Checkpoint':<20}{'Baseline PassRate':>20}{'Baseline MaxSpd':>18}{'Agent Verdict':>16}")
    for name, m, gr, verdict, n_passed, n_total in rows:
        b = BASELINE[name]
        lines.append(f"{name:<20}{b['pass_rate']*100:>19.1f}%{b['max_speed']:>17.1f}m{verdict:>16}")
    lines.append("")
    lines.append(f"Naive 'pick highest pass rate' would select: {naive_pick}")
    advance = [r[0] for r in rows if r[3] == "ADVANCE"]
    conditional = [r[0] for r in rows if r[3] == "CONDITIONAL"]
    eliminate = [r[0] for r in rows if r[3] == "ELIMINATE"]
    lines.append(f"Agent ADVANCE list:    {advance if advance else '(none)'}")
    lines.append(f"Agent CONDITIONAL list: {conditional if conditional else '(none)'}")
    lines.append(f"Agent ELIMINATE list:  {eliminate if eliminate else '(none)'}")
    lines.append("")
    if naive_pick in advance:
        lines.append("=> Agent AGREES with naive pass-rate ranking on this checkpoint set.")
    elif naive_pick in eliminate:
        lines.append(
            f"=> Agent DISAGREES: naive ranking would deploy '{naive_pick}', "
            "but the gate analysis ELIMINATES it for failing safety gates the raw "
            "pass-rate metric does not capture."
        )
    else:
        lines.append(
            f"=> Agent flags the naive pick '{naive_pick}' as CONDITIONAL, not a clean ADVANCE."
        )
    lines.append("=" * 100)

    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "policy_downselection_summary.txt"), "w") as f:
        f.write(text)
    print(text)


def write_chart(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_specs = [
        ("pass_rate", "Pass Rate", GATES["G2_pass_rate"]["threshold"], "gte", lambda v: v * 100, "%"),
        ("boundary_speed_ceiling", "Speed Ceiling (mph)", GATES["G1_speed_ceiling"]["threshold"], "gte", lambda v: v, "mph"),
        ("boundary_drift", "Boundary Drift (mph)", GATES["G3_boundary_stability"]["threshold"], "lte", lambda v: v, "mph"),
        ("near_boundary_fail_prob", "Near-Boundary Fail Prob", GATES["G4_near_boundary_safety"]["threshold"], "lte", lambda v: v * 100, "%"),
        ("query_efficiency", "Query Efficiency", GATES["G5_query_efficiency"]["threshold"], "gte", lambda v: v, ""),
    ]

    names = [r[0] for r in rows]
    verdicts = [r[3] for r in rows]
    verdict_color = {"ADVANCE": "#2ca02c", "CONDITIONAL": "#ff9800", "ELIMINATE": "#d62728"}

    fig, axes = plt.subplots(1, len(metric_specs), figsize=(24, 5))
    x = np.arange(len(names))

    for ax, (key, label, threshold, compare, scale, unit) in zip(axes, metric_specs):
        raw_vals = [r[1][key] for r in rows]
        vals = [scale(v) for v in raw_vals]
        thr_scaled = scale(threshold) if key != "query_efficiency" else threshold
        colors = []
        for v in raw_vals:
            ok = v >= threshold if compare == "gte" else v <= threshold
            colors.append("#4c72b0" if ok else "#c44e52")
        ax.bar(x, vals, color=colors)
        ax.axhline(thr_scaled, color="black", linestyle="--", linewidth=1.2, label=f"gate = {thr_scaled:.2f}{unit}")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Policy Down-Selection: Metrics vs Safety Gate Thresholds\n"
                 "(blue = gate passed, red = gate failed; verdicts: "
                 + ", ".join(f"{n}={v}" for n, v in zip(names, verdicts)) + ")",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(OUT_DIR, "policy_downselection_chart.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
