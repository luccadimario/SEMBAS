"""
sembas_metrics.py — Accuracy and efficiency metrics for SEMBAS boundary exploration.

Phase 1 of the research evaluation framework: rigorous quantitative assessment of
how well a boundary exploration run has found the true pass/fail boundary.

Metric categories
-----------------
Accuracy (requires a ground-truth boundary):
    hausdorff_distance   — worst-case point-to-boundary error
    chamfer_distance     — average-case point-to-boundary error

Efficiency (derived from the exploration run log):
    boundary_sampling_efficiency     — BSE as defined in the SEMBAS paper (Eq. 12)
    near_boundary_failure_probability — fragility of the found boundary region
    coverage_efficiency              — spatial coverage of the boundary sample

Drift (requires two exploration runs):
    boundary_drift — how much the boundary moved between two runs

All functions operate on plain numpy arrays; no scipy dependency.
Nearest-neighbour search is implemented via numpy broadcasting (squared-distance
identity), which is efficient for the point counts typical of SEMBAS runs
(hundreds to low thousands of boundary points per dimension).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nearest_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """For each point in A, return the Euclidean distance to the nearest point in B.

    Uses the squared-distance identity
    ``||a - b||^2 = ||a||^2 + ||b||^2 - 2 * dot(a, b)``
    to avoid an explicit O(N * M * d) subtraction loop.

    Parameters
    ----------
    A : np.ndarray, shape (N, d)
    B : np.ndarray, shape (M, d)

    Returns
    -------
    np.ndarray, shape (N,)
        ``result[i]`` = min over j of ``||A[i] - B[j]||``.
    """
    A_sq = np.sum(A ** 2, axis=1, keepdims=True)   # (N, 1)
    B_sq = np.sum(B ** 2, axis=1, keepdims=True)   # (M, 1)
    dist_sq = A_sq + B_sq.T - 2.0 * (A @ B.T)     # (N, M)
    np.clip(dist_sq, 0.0, None, out=dist_sq)        # guard floating-point negatives
    return np.sqrt(dist_sq.min(axis=1))             # (N,)


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def hausdorff_distance(
    estimated_boundary: np.ndarray,
    true_boundary: np.ndarray,
) -> dict[str, float]:
    """Hausdorff distance between an estimated and a ground-truth boundary.

    The Hausdorff distance is the "worst-case" error: it equals the maximum
    distance that any point in one set is from the nearest point in the other
    set.  A single outlier drives this metric up, making it sensitive to rare
    but large errors.

    The directed Hausdorff A→B is the maximum over points in A of the distance
    to the nearest point in B.  The symmetric Hausdorff is the maximum of both
    directed versions.

    Parameters
    ----------
    estimated_boundary : np.ndarray, shape (N, ndim)
        Boundary points produced by SEMBAS.
    true_boundary : np.ndarray, shape (M, ndim)
        Ground-truth boundary points (e.g., analytically sampled from a known
        geometry such as a sphere or hypercube surface).

    Returns
    -------
    dict with keys:
        ``hausdorff`` — symmetric Hausdorff distance (max of both directions).
        ``A_to_B``    — directed Hausdorff from estimated → true.
        ``B_to_A``    — directed Hausdorff from true → estimated.
    """
    d_A_to_B = float(_nearest_distances(estimated_boundary, true_boundary).max())
    d_B_to_A = float(_nearest_distances(true_boundary, estimated_boundary).max())
    return {
        "hausdorff": max(d_A_to_B, d_B_to_A),
        "A_to_B": d_A_to_B,
        "B_to_A": d_B_to_A,
    }


def chamfer_distance(
    estimated_boundary: np.ndarray,
    true_boundary: np.ndarray,
) -> float:
    """Chamfer distance between an estimated and a ground-truth boundary.

    The Chamfer distance averages the nearest-neighbour distances in both
    directions, making it far less sensitive to outliers than the Hausdorff
    distance.  It is the primary "average-case" accuracy metric.

    Chamfer = mean_A(min_B dist) + mean_B(min_A dist)

    Parameters
    ----------
    estimated_boundary : np.ndarray, shape (N, ndim)
        Boundary points produced by SEMBAS.
    true_boundary : np.ndarray, shape (M, ndim)
        Ground-truth boundary points.

    Returns
    -------
    float
        Sum of the mean nearest-neighbour distances in both directions.
    """
    mean_A_to_B = _nearest_distances(estimated_boundary, true_boundary).mean()
    mean_B_to_A = _nearest_distances(true_boundary, estimated_boundary).mean()
    return float(mean_A_to_B + mean_B_to_A)


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------

def boundary_sampling_efficiency(
    boundary_points: list,
    non_boundary_points: list,
) -> float:
    """Boundary Sampling Efficiency (BSE) as defined in the SEMBAS paper (Eq. 12).

    BSE measures what fraction of oracle calls produced boundary (informative)
    samples rather than interior/exterior ones.  Higher is better: 100 % would
    mean every oracle call landed exactly on the boundary.

    BSE = |boundary_points| / |non_boundary_points| × 100

    Parameters
    ----------
    boundary_points : list
        Points identified as lying on (or very near) the pass/fail boundary.
        Any sized container works — only ``len()`` is called.
    non_boundary_points : list
        Interior or exterior points evaluated but not classified as boundary.

    Returns
    -------
    float
        BSE in percent.  Values > 100 are possible (more boundary samples than
        non-boundary ones) but unusual.

    Raises
    ------
    ValueError
        If ``non_boundary_points`` is empty (division by zero).
    """
    if len(non_boundary_points) == 0:
        raise ValueError(
            "non_boundary_points must be non-empty (denominator would be zero)."
        )
    return float(len(boundary_points) / len(non_boundary_points) * 100.0)


def near_boundary_failure_probability(
    evaluator_fn: Callable[[np.ndarray], bool],
    boundary_points: np.ndarray,
    normal_vectors: np.ndarray,
    epsilon: float,
    n_samples: int = 50,
    seed: int | None = None,
) -> float:
    """Estimate the probability of failure in the epsilon-neighbourhood of the boundary.

    For each boundary point ``b_i``, this function draws *n_samples* points
    uniformly from the epsilon-ball centred at ``b_i`` and calls *evaluator_fn*
    on each one.  The returned value is the global fraction that fail (evaluator
    returns False).

    A high failure probability means that a large portion of the neighbourhood
    is inside the failure region — consistent with a correctly placed boundary.
    Very low values suggest the found boundary points are actually in the safe
    interior far from the true boundary.

    Sampling uses a uniform-in-ball distribution: directions are drawn from
    the unit sphere (normalised Gaussian), radii are scaled by ``u^(1/d)``
    to achieve uniform volume coverage.

    Parameters
    ----------
    evaluator_fn : callable (np.ndarray) -> bool
        Oracle function.  Given a single scenario point ``x`` of shape
        ``(ndim,)``, returns True (pass) or False (fail).
    boundary_points : np.ndarray, shape (N, ndim)
        Estimated boundary points to probe.
    normal_vectors : np.ndarray, shape (N, ndim)
        OSVs at each boundary point.  Reserved for future directional-sampling
        variants; currently unused in the sampling step.
    epsilon : float
        Radius of the neighbourhood ball around each boundary point.
    n_samples : int, optional
        Number of random probe points per boundary point.  Default 50.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    float
        Fraction of probed points in [0, 1] that returned False (failure).
    """
    rng = np.random.default_rng(seed)
    n_points, ndim = boundary_points.shape
    total_calls = 0
    total_failures = 0

    for i in range(n_points):
        # Sample uniformly inside an ndim-ball: normalise Gaussian directions,
        # then scale radii by u^(1/d) for uniform volume distribution.
        directions = rng.standard_normal((n_samples, ndim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        radii = epsilon * rng.uniform(0.0, 1.0, n_samples) ** (1.0 / ndim)
        samples = boundary_points[i] + radii[:, None] * directions  # (n_samples, ndim)

        for x in samples:
            if not evaluator_fn(x):
                total_failures += 1
            total_calls += 1

    return float(total_failures / total_calls) if total_calls > 0 else 0.0


def coverage_efficiency(
    boundary_points: np.ndarray,
    domain_bounds: np.ndarray,
    grid_resolution: float = 0.075,
) -> dict[str, float | int]:
    """Measure how uniformly the boundary samples cover the scenario space.

    The domain is discretised into a regular grid of hypercubes with side
    *grid_resolution*.  Each boundary point is assigned to a cell; the function
    counts how many distinct cells contain at least one point.

    Parameters
    ----------
    boundary_points : np.ndarray, shape (N, ndim)
        Estimated boundary points.
    domain_bounds : np.ndarray, shape (ndim, 2)
        Axis-aligned bounds of the scenario domain.  ``domain_bounds[:, 0]``
        is the lower bound and ``domain_bounds[:, 1]`` the upper bound for
        each dimension.
    grid_resolution : float, optional
        Cell width applied uniformly across all dimensions.  Default 0.075.

    Returns
    -------
    dict with keys:
        ``coverage_pct``        — cells_covered / total_cells × 100.
        ``cells_covered``       — number of grid cells containing ≥ 1 point.
        ``total_cells``         — total cells in the domain grid.
        ``samples_used``        — total boundary points (N).
        ``coverage_efficiency`` — cells_covered / samples_used × 100.
                                  Range (0, 100]: 100 means every sample
                                  landed in a unique cell.
    """
    N, ndim = boundary_points.shape
    lower = domain_bounds[:, 0]   # (ndim,)
    upper = domain_bounds[:, 1]   # (ndim,)

    n_cells_per_dim = np.maximum(
        np.ceil((upper - lower) / grid_resolution).astype(int), 1
    )
    total_cells = int(np.prod(n_cells_per_dim))

    cell_indices = np.floor(
        (boundary_points - lower) / grid_resolution
    ).astype(int)
    # Clip handles points that sit exactly on the upper boundary.
    cell_indices = np.clip(cell_indices, 0, n_cells_per_dim - 1)

    cells_covered = len(set(map(tuple, cell_indices.tolist())))

    return {
        "coverage_pct": float(cells_covered / total_cells * 100.0),
        "cells_covered": cells_covered,
        "total_cells": total_cells,
        "samples_used": N,
        "coverage_efficiency": float(cells_covered / N * 100.0) if N > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Drift metrics
# ---------------------------------------------------------------------------

def boundary_drift(
    boundary_before: np.ndarray,
    boundary_after: np.ndarray,
) -> dict[str, float]:
    """Quantify how much the estimated boundary shifted between two SEMBAS runs.

    Useful for tracking boundary evolution during RL training: how far did the
    safety boundary move after a policy update?  Chamfer drift is the primary
    metric because it is stable under outliers.  Hausdorff drift captures the
    worst-case shift.  Mean displacement gives a one-sided "from-before" view.

    Parameters
    ----------
    boundary_before : np.ndarray, shape (N, ndim)
        Boundary sample from the earlier run (e.g., before retraining).
    boundary_after : np.ndarray, shape (M, ndim)
        Boundary sample from the later run (e.g., after retraining).

    Returns
    -------
    dict with keys:
        ``chamfer_drift``     — Chamfer distance between the two boundaries.
        ``hausdorff_drift``   — Hausdorff distance between the two boundaries.
        ``mean_displacement`` — Mean distance from each point in
                                *boundary_before* to its nearest point in
                                *boundary_after*.  Indicates how far individual
                                anchor points "moved" on average (one-sided).
    """
    ch = chamfer_distance(boundary_before, boundary_after)
    hd = hausdorff_distance(boundary_before, boundary_after)
    mean_disp = float(_nearest_distances(boundary_before, boundary_after).mean())

    return {
        "chamfer_drift": ch,
        "hausdorff_drift": hd["hausdorff"],
        "mean_displacement": mean_disp,
    }


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class BoundaryMetricsReport:
    """Container for all boundary-quality metrics from a single SEMBAS run.

    Fields default to None; populate only the metrics that are available for a
    given run (accuracy metrics require a ground-truth boundary).

    Fields
    ------
    hausdorff : float or None
        Symmetric Hausdorff distance to the ground-truth boundary.
    hausdorff_A_to_B : float or None
        Directed Hausdorff: estimated → true.
    hausdorff_B_to_A : float or None
        Directed Hausdorff: true → estimated.
    chamfer : float or None
        Chamfer distance to the ground-truth boundary.
    bse : float or None
        Boundary Sampling Efficiency in percent.
    near_boundary_failure_prob : float or None
        Fraction of epsilon-neighbourhood probes that fail the oracle.
    drift : dict or None
        Output of :func:`boundary_drift`; present when comparing two runs.
    coverage : dict or None
        Output of :func:`coverage_efficiency`.
    """

    hausdorff: float | None = None
    hausdorff_A_to_B: float | None = None
    hausdorff_B_to_A: float | None = None
    chamfer: float | None = None
    bse: float | None = None
    near_boundary_failure_prob: float | None = None
    drift: dict[str, float] | None = None
    coverage: dict[str, float | int] | None = None

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a formatted table of all non-None metrics to stdout."""
        label_w = 40
        sep = "=" * (label_w + 18)

        def row(label: str, value: float | int | None) -> str:
            if value is None:
                return ""
            if isinstance(value, float):
                return f"  {label:<{label_w}}  {value:.6f}"
            return f"  {label:<{label_w}}  {value}"

        print(sep)
        print("  SEMBAS Boundary Metrics Report")
        print(sep)

        lines = [
            row("Hausdorff distance",              self.hausdorff),
            row("  Hausdorff (A→B, est.→true)",   self.hausdorff_A_to_B),
            row("  Hausdorff (B→A, true→est.)",   self.hausdorff_B_to_A),
            row("Chamfer distance",                self.chamfer),
            row("Boundary Sampling Eff. (%)",     self.bse),
            row("Near-boundary failure prob",      self.near_boundary_failure_prob),
        ]

        if self.drift is not None:
            lines += [
                "",
                row("Drift — Chamfer",           self.drift.get("chamfer_drift")),
                row("Drift — Hausdorff",         self.drift.get("hausdorff_drift")),
                row("Drift — Mean displacement", self.drift.get("mean_displacement")),
            ]

        if self.coverage is not None:
            lines += [
                "",
                row("Coverage (%)",                     self.coverage.get("coverage_pct")),
                row("Cells covered",                    self.coverage.get("cells_covered")),
                row("Total cells",                      self.coverage.get("total_cells")),
                row("Coverage efficiency (cells/samp%)", self.coverage.get("coverage_efficiency")),
            ]

        for line in lines:
            if line:
                print(line)

        print(sep)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        All fields are included; None values are preserved so the schema is
        stable across runs with varying metric availability.

        Returns
        -------
        dict
            Flat dictionary suitable for ``json.dump``.
        """
        return {
            "hausdorff": self.hausdorff,
            "hausdorff_A_to_B": self.hausdorff_A_to_B,
            "hausdorff_B_to_A": self.hausdorff_B_to_A,
            "chamfer": self.chamfer,
            "bse": self.bse,
            "near_boundary_failure_prob": self.near_boundary_failure_prob,
            "drift": self.drift,
            "coverage": self.coverage,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Generating synthetic 3-D sphere boundary...")
    print("  Ground truth : sphere  r=0.4  centre=[0.5, 0.5, 0.5]  (M=800 pts)")
    print("  Estimated    : noisy subset of true pts  (N=400, σ=0.02)")
    print()

    rng = np.random.default_rng(0)
    centre = np.array([0.5, 0.5, 0.5])
    radius = 0.4

    # Ground-truth boundary: 800 uniformly distributed points on the sphere.
    M = 800
    v_true = rng.standard_normal((M, 3))
    v_true /= np.linalg.norm(v_true, axis=1, keepdims=True)
    true_boundary = centre + radius * v_true  # (800, 3)

    # Estimated boundary: random 400-point subset with Gaussian noise.
    N = 400
    idx = rng.choice(M, size=N, replace=False)
    estimated_boundary = true_boundary[idx] + rng.normal(0.0, 0.02, (N, 3))

    # Outward surface normals at each estimated boundary point.
    normals = estimated_boundary - centre
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # 1. Hausdorff distance
    # ------------------------------------------------------------------
    print("Computing Hausdorff distance...")
    hd = hausdorff_distance(estimated_boundary, true_boundary)

    # ------------------------------------------------------------------
    # 2. Chamfer distance
    # ------------------------------------------------------------------
    print("Computing Chamfer distance...")
    ch = chamfer_distance(estimated_boundary, true_boundary)

    # ------------------------------------------------------------------
    # 3. Boundary Sampling Efficiency
    #    Simulate: 80 non-boundary evaluations were also made during the run.
    # ------------------------------------------------------------------
    bse = boundary_sampling_efficiency(
        list(range(N)),
        list(range(80)),
    )

    # ------------------------------------------------------------------
    # 4. Near-boundary failure probability
    #    Oracle: a point passes iff it is outside the sphere.
    #    Probe only 20 boundary points for speed in this demo.
    # ------------------------------------------------------------------
    print("Probing near-boundary failure probability (20 pts × 50 samples)...")

    def oracle(x: np.ndarray) -> bool:
        return bool(np.linalg.norm(x - centre) >= radius)

    nbfp = near_boundary_failure_probability(
        oracle,
        estimated_boundary[:20],
        normals[:20],
        epsilon=0.04,
        n_samples=50,
        seed=1,
    )

    # ------------------------------------------------------------------
    # 5. Coverage efficiency  (domain [0,1]^3, 7.5 cm grid cells)
    # ------------------------------------------------------------------
    domain_bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    cov = coverage_efficiency(estimated_boundary, domain_bounds, grid_resolution=0.075)

    # ------------------------------------------------------------------
    # 6. Boundary drift: compare run-1 vs a second noisier run.
    # ------------------------------------------------------------------
    print("Computing boundary drift against a second run (σ=0.03)...")
    idx2 = rng.choice(M, size=N, replace=False)
    estimated_v2 = true_boundary[idx2] + rng.normal(0.0, 0.03, (N, 3))
    drift = boundary_drift(estimated_boundary, estimated_v2)

    # ------------------------------------------------------------------
    # Assemble and display report
    # ------------------------------------------------------------------
    report = BoundaryMetricsReport(
        hausdorff=hd["hausdorff"],
        hausdorff_A_to_B=hd["A_to_B"],
        hausdorff_B_to_A=hd["B_to_A"],
        chamfer=ch,
        bse=bse,
        near_boundary_failure_prob=nbfp,
        drift=drift,
        coverage=cov,
    )

    print()
    report.summary()

    print("\nJSON export preview:")
    print(json.dumps(report.to_dict(), indent=2))

    sys.exit(0)
