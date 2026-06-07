"""
boundary_envelope.py — Universal data schema connecting SEMBAS boundary exploration
results to RL training, metrics, and cross-method comparison.

A BoundaryEnvelope captures everything known about a single discovered boundary
point: the scenario vector that was queried, whether it passed or failed, where
the estimated boundary surface lies, and geometric properties of that surface
(normal direction, curvature, local reliability).  A BoundaryEnvelopeCollection
aggregates many such records for persistence, analysis, and DataFrame export.

Scenario space conventions
---------------------------
CARLoS (4-D):  [longitude, latitude, dir_angle_offset, speed_mph]
CARLoS-Agents gym (3-D):  [lane_width_ft, num_obstacles, initial_speed_mph]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------

@dataclass
class BoundaryEnvelope:
    """A single boundary point record produced by a SEMBAS-style exploration run.

    Each instance corresponds to one oracle query (simulator call) whose result
    placed a point near the pass/fail boundary of a scenario space.  The fields
    capture both the raw query result and the geometric properties SEMBAS
    estimates about the local boundary surface.

    Fields
    ------
    scenario_id : str
        Unique identifier for this scenario query.  Typically a UUID or a
        deterministic hash of `x`, so that records from independent runs that
        queried the same point can be matched.

    x : np.ndarray, shape (ndim,)
        The scenario parameter vector that was evaluated.  Dimensions follow
        the space convention documented at module level (e.g. for CARLoS-Agents:
        [lane_width_ft, num_obstacles, initial_speed_mph]).

    outcome : bool
        Oracle (simulator) label for `x`.  True means the scenario passed the
        safety criterion; False means it failed.

    threshold : float
        The numerical pass/fail criterion value used to derive `outcome`.  For
        lane-keeping tasks this is typically lane_width / 2 (metres or feet
        depending on the space).  Storing this here makes the records
        self-describing when the threshold changes across experiments.

    boundary_point : np.ndarray, shape (ndim,)
        SEMBAS's best estimate of the point on the pass/fail boundary that is
        closest to (or consistent with) `x`.  Computed by SEMBAS after binary
        search or interpolation along the normal direction.

    normal_vector : np.ndarray, shape (ndim,)
        The Oriented Surface Vector (OSV) — a unit vector in scenario space that
        points outward from the failure region at `boundary_point`.  Used by
        SEMBAS to guide the next exploration sample and by downstream RL reward
        shaping as a margin direction.

    local_error_estimate : float
        Estimated Euclidean distance (in scenario space units) from
        `boundary_point` to the true boundary.  Lower values indicate a tighter,
        more confident boundary localisation.  Derived from SEMBAS's internal
        convergence criterion.

    curvature_proxy : float
        A scalar proxy for the local curvature of the boundary surface near
        `boundary_point`.  Higher values suggest a strongly curved (non-planar)
        boundary, which implies that linear OSV extrapolation is less reliable.
        Units are scenario-space-dependent (1 / distance).

    label_stability : float
        Fraction of repeated or nearby oracle queries around `boundary_point`
        that returned the same label as `outcome`.  Range [0, 1].  Values near
        1.0 indicate a deterministic, stable boundary; values near 0.5 indicate
        a noisy or stochastic boundary region that SEMBAS should re-sample.

    simulator_calls_used : int
        Total number of oracle (simulator) evaluations consumed to localise this
        boundary point, counting from the initial sample to convergence.  Used
        to compare the sample efficiency of different methods.

    source_method : str
        Tag identifying the exploration algorithm that produced this record.
        One of: 'SEMBAS', 'GenBo', 'random', 'GD' (gradient-descent baseline).
        Used to partition BoundaryEnvelopeCollections for cross-method comparison.

    policy_id : str
        Checkpoint tag of the RL policy that was being evaluated when this
        boundary point was found.  Allows tracking how the boundary shifts as
        the policy improves during training.

    training_iteration : int
        The RL training episode or iteration index at which this boundary point
        was collected.  Together with `policy_id` this gives a full temporal
        position in the training curriculum.
    """

    scenario_id: str
    x: np.ndarray
    outcome: bool
    threshold: float
    boundary_point: np.ndarray
    normal_vector: np.ndarray
    local_error_estimate: float
    curvature_proxy: float
    label_stability: float
    simulator_calls_used: int
    source_method: str
    policy_id: str
    training_iteration: int

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        numpy arrays are converted to plain Python lists so the result can be
        passed directly to ``json.dump``.

        Returns
        -------
        dict
            All fields with numpy arrays replaced by lists.
        """
        return {
            "scenario_id": self.scenario_id,
            "x": self.x.tolist(),
            "outcome": self.outcome,
            "threshold": self.threshold,
            "boundary_point": self.boundary_point.tolist(),
            "normal_vector": self.normal_vector.tolist(),
            "local_error_estimate": self.local_error_estimate,
            "curvature_proxy": self.curvature_proxy,
            "label_stability": self.label_stability,
            "simulator_calls_used": self.simulator_calls_used,
            "source_method": self.source_method,
            "policy_id": self.policy_id,
            "training_iteration": self.training_iteration,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BoundaryEnvelope":
        """Deserialise from a dictionary produced by ``to_dict``.

        Lists are converted back to numpy float64 arrays.

        Parameters
        ----------
        d : dict
            Dictionary as returned by ``to_dict`` or parsed from JSON.

        Returns
        -------
        BoundaryEnvelope
        """
        return cls(
            scenario_id=d["scenario_id"],
            x=np.array(d["x"], dtype=np.float64),
            outcome=bool(d["outcome"]),
            threshold=float(d["threshold"]),
            boundary_point=np.array(d["boundary_point"], dtype=np.float64),
            normal_vector=np.array(d["normal_vector"], dtype=np.float64),
            local_error_estimate=float(d["local_error_estimate"]),
            curvature_proxy=float(d["curvature_proxy"]),
            label_stability=float(d["label_stability"]),
            simulator_calls_used=int(d["simulator_calls_used"]),
            source_method=d["source_method"],
            policy_id=d["policy_id"],
            training_iteration=int(d["training_iteration"]),
        )

    def to_numpy_row(self) -> np.ndarray:
        """Flatten all numeric fields into a single 1-D float64 array.

        Layout (for an ndim-dimensional scenario space)::

            [ x (ndim)
            | outcome (1)
            | threshold (1)
            | boundary_point (ndim)
            | normal_vector (ndim)
            | local_error_estimate (1)
            | curvature_proxy (1)
            | label_stability (1)
            | simulator_calls_used (1)
            | training_iteration (1) ]

        String fields (``scenario_id``, ``source_method``, ``policy_id``) are
        excluded.  Total length = 3*ndim + 7.

        Returns
        -------
        np.ndarray, shape (3*ndim + 7,)
        """
        return np.concatenate([
            self.x,
            [float(self.outcome)],
            [self.threshold],
            self.boundary_point,
            self.normal_vector,
            [self.local_error_estimate],
            [self.curvature_proxy],
            [self.label_stability],
            [float(self.simulator_calls_used)],
            [float(self.training_iteration)],
        ]).astype(np.float64)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

class BoundaryEnvelopeCollection:
    """An ordered collection of :class:`BoundaryEnvelope` records.

    Designed as the primary container for storing and analysing results from a
    SEMBAS exploration run or a set of runs.  Provides persistence (JSON
    save/load), optional pandas DataFrame export, method-based filtering, and
    summary statistics.

    Parameters
    ----------
    records : list of BoundaryEnvelope, optional
        Initial records.  Defaults to an empty list.
    """

    def __init__(self, records: list[BoundaryEnvelope] | None = None) -> None:
        self.records: list[BoundaryEnvelope] = records if records is not None else []

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def append(self, envelope: BoundaryEnvelope) -> None:
        """Add a single BoundaryEnvelope to the collection."""
        self.records.append(envelope)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the collection to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination file path.  Parent directories must exist.
        """
        payload = [r.to_dict() for r in self.records]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BoundaryEnvelopeCollection":
        """Load a collection from a JSON file produced by :meth:`save`.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        BoundaryEnvelopeCollection
        """
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls([BoundaryEnvelope.from_dict(d) for d in payload])

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def filter_by_method(self, source_method: str) -> "BoundaryEnvelopeCollection":
        """Return a new collection containing only records from *source_method*.

        Parameters
        ----------
        source_method : str
            One of 'SEMBAS', 'GenBo', 'random', 'GD', or any custom tag.

        Returns
        -------
        BoundaryEnvelopeCollection
            Filtered copy; the original is unchanged.
        """
        return BoundaryEnvelopeCollection(
            [r for r in self.records if r.source_method == source_method]
        )

    def summary_stats(self) -> dict[str, Any]:
        """Compute aggregate statistics over the collection.

        Returns
        -------
        dict with keys:
            ``count`` — total number of records.
            ``mean_local_error`` — mean of ``local_error_estimate`` across all records.
            ``std_local_error`` — standard deviation of ``local_error_estimate``.
            ``mean_simulator_calls`` — mean ``simulator_calls_used``.
            ``pass_rate`` — fraction of records where ``outcome`` is True.
            ``methods`` — list of distinct ``source_method`` values present.
        """
        if not self.records:
            return {
                "count": 0,
                "mean_local_error": None,
                "std_local_error": None,
                "mean_simulator_calls": None,
                "pass_rate": None,
                "methods": [],
            }

        errors = np.array([r.local_error_estimate for r in self.records])
        calls = np.array([r.simulator_calls_used for r in self.records], dtype=float)
        outcomes = np.array([float(r.outcome) for r in self.records])
        methods = sorted({r.source_method for r in self.records})

        return {
            "count": len(self.records),
            "mean_local_error": float(errors.mean()),
            "std_local_error": float(errors.std()),
            "mean_simulator_calls": float(calls.mean()),
            "pass_rate": float(outcomes.mean()),
            "methods": methods,
        }

    def as_dataframe(self):
        """Convert the collection to a pandas DataFrame.

        Each row corresponds to one BoundaryEnvelope.  numpy array fields
        (``x``, ``boundary_point``, ``normal_vector``) are stored as object
        columns containing the original arrays.

        Returns
        -------
        pandas.DataFrame
            One row per record.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for as_dataframe(); install it with: pip install pandas"
            ) from exc

        return pd.DataFrame([r.to_dict() for r in self.records])
