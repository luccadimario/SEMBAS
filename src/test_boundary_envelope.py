"""
test_boundary_envelope.py — Round-trip and integrity tests for BoundaryEnvelope.

Run with:
    python src/test_boundary_envelope.py
or via pytest from the project root:
    pytest src/test_boundary_envelope.py
"""

import json
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from boundary_envelope import BoundaryEnvelope, BoundaryEnvelopeCollection


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

def _make_envelope(
    ndim: int = 3,
    outcome: bool = True,
    source_method: str = "SEMBAS",
    training_iteration: int = 0,
) -> BoundaryEnvelope:
    """Return a BoundaryEnvelope with deterministic dummy values."""
    rng = np.random.default_rng(seed=ndim + int(outcome))
    x = rng.uniform(0, 1, ndim)
    bp = x + rng.uniform(-0.05, 0.05, ndim)
    nv = rng.standard_normal(ndim)
    nv /= np.linalg.norm(nv)

    return BoundaryEnvelope(
        scenario_id=str(uuid.uuid4()),
        x=x,
        outcome=outcome,
        threshold=0.5,
        boundary_point=bp,
        normal_vector=nv,
        local_error_estimate=float(rng.uniform(0.01, 0.1)),
        curvature_proxy=float(rng.uniform(0.0, 0.5)),
        label_stability=float(rng.uniform(0.7, 1.0)),
        simulator_calls_used=int(rng.integers(5, 30)),
        source_method=source_method,
        policy_id="checkpoint_ep100",
        training_iteration=training_iteration,
    )


def _make_collection(n: int = 5, methods=("SEMBAS", "random")) -> BoundaryEnvelopeCollection:
    col = BoundaryEnvelopeCollection()
    for i in range(n):
        method = methods[i % len(methods)]
        col.append(_make_envelope(ndim=3, outcome=(i % 2 == 0), source_method=method, training_iteration=i))
    return col


# ---------------------------------------------------------------------------
# BoundaryEnvelope unit tests
# ---------------------------------------------------------------------------

class TestBoundaryEnvelopeRoundTrip:
    def test_to_dict_contains_all_keys(self):
        env = _make_envelope()
        d = env.to_dict()
        expected_keys = {
            "scenario_id", "x", "outcome", "threshold", "boundary_point",
            "normal_vector", "local_error_estimate", "curvature_proxy",
            "label_stability", "simulator_calls_used", "source_method",
            "policy_id", "training_iteration",
        }
        assert set(d.keys()) == expected_keys

    def test_dict_arrays_are_lists(self):
        env = _make_envelope()
        d = env.to_dict()
        assert isinstance(d["x"], list)
        assert isinstance(d["boundary_point"], list)
        assert isinstance(d["normal_vector"], list)

    def test_from_dict_restores_arrays(self):
        env = _make_envelope(ndim=4)
        restored = BoundaryEnvelope.from_dict(env.to_dict())
        np.testing.assert_array_almost_equal(env.x, restored.x)
        np.testing.assert_array_almost_equal(env.boundary_point, restored.boundary_point)
        np.testing.assert_array_almost_equal(env.normal_vector, restored.normal_vector)

    def test_from_dict_restores_scalars(self):
        env = _make_envelope()
        restored = BoundaryEnvelope.from_dict(env.to_dict())
        assert restored.scenario_id == env.scenario_id
        assert restored.outcome == env.outcome
        assert restored.threshold == pytest.approx(env.threshold)
        assert restored.local_error_estimate == pytest.approx(env.local_error_estimate)
        assert restored.curvature_proxy == pytest.approx(env.curvature_proxy)
        assert restored.label_stability == pytest.approx(env.label_stability)
        assert restored.simulator_calls_used == env.simulator_calls_used
        assert restored.source_method == env.source_method
        assert restored.policy_id == env.policy_id
        assert restored.training_iteration == env.training_iteration

    def test_json_serialisable(self):
        env = _make_envelope()
        raw = json.dumps(env.to_dict())
        assert isinstance(raw, str)
        assert env.scenario_id in raw

    def test_to_numpy_row_shape(self):
        ndim = 3
        env = _make_envelope(ndim=ndim)
        row = env.to_numpy_row()
        expected_len = 3 * ndim + 7
        assert row.shape == (expected_len,)

    def test_to_numpy_row_dtype(self):
        env = _make_envelope()
        assert env.to_numpy_row().dtype == np.float64

    def test_to_numpy_row_outcome_encoded(self):
        env_pass = _make_envelope(outcome=True)
        env_fail = _make_envelope(outcome=False)
        ndim = 3
        # outcome is at index ndim in the flat row
        assert env_pass.to_numpy_row()[ndim] == 1.0
        assert env_fail.to_numpy_row()[ndim] == 0.0

    def test_4d_scenario_space(self):
        """CARLoS 4-D space: [longitude, latitude, dir_angle_offset, speed_mph]."""
        env = _make_envelope(ndim=4)
        assert env.x.shape == (4,)
        assert env.normal_vector.shape == (4,)
        row = env.to_numpy_row()
        assert row.shape == (3 * 4 + 7,)


# ---------------------------------------------------------------------------
# BoundaryEnvelopeCollection tests
# ---------------------------------------------------------------------------

class TestBoundaryEnvelopeCollection:
    def test_append_and_len(self):
        col = BoundaryEnvelopeCollection()
        assert len(col) == 0
        col.append(_make_envelope())
        assert len(col) == 1

    def test_save_and_load_round_trip(self, tmp_path):
        col = _make_collection(n=4)
        save_path = tmp_path / "test_collection.json"
        col.save(save_path)

        loaded = BoundaryEnvelopeCollection.load(save_path)
        assert len(loaded) == len(col)

        for orig, rest in zip(col, loaded):
            assert orig.scenario_id == rest.scenario_id
            np.testing.assert_array_almost_equal(orig.x, rest.x)
            np.testing.assert_array_almost_equal(orig.boundary_point, rest.boundary_point)
            np.testing.assert_array_almost_equal(orig.normal_vector, rest.normal_vector)
            assert orig.outcome == rest.outcome
            assert orig.simulator_calls_used == rest.simulator_calls_used

    def test_save_produces_valid_json(self, tmp_path):
        col = _make_collection(n=3)
        save_path = tmp_path / "out.json"
        col.save(save_path)
        with open(save_path) as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) == 3

    def test_filter_by_method(self):
        col = _make_collection(n=6, methods=("SEMBAS", "random", "GenBo"))
        sembas = col.filter_by_method("SEMBAS")
        assert all(r.source_method == "SEMBAS" for r in sembas)
        assert len(sembas) + len(col.filter_by_method("random")) + len(col.filter_by_method("GenBo")) == len(col)

    def test_filter_returns_new_collection(self):
        col = _make_collection(n=4)
        filtered = col.filter_by_method("SEMBAS")
        assert filtered is not col

    def test_summary_stats_empty(self):
        col = BoundaryEnvelopeCollection()
        stats = col.summary_stats()
        assert stats["count"] == 0
        assert stats["mean_local_error"] is None

    def test_summary_stats_keys(self):
        col = _make_collection(n=5)
        stats = col.summary_stats()
        for key in ("count", "mean_local_error", "std_local_error", "mean_simulator_calls", "pass_rate", "methods"):
            assert key in stats

    def test_summary_stats_pass_rate_range(self):
        col = _make_collection(n=10)
        stats = col.summary_stats()
        assert 0.0 <= stats["pass_rate"] <= 1.0

    def test_summary_stats_count(self):
        col = _make_collection(n=7)
        assert col.summary_stats()["count"] == 7

    def test_summary_stats_methods_list(self):
        col = _make_collection(n=4, methods=("SEMBAS", "random"))
        methods = col.summary_stats()["methods"]
        assert set(methods) == {"SEMBAS", "random"}

    def test_iteration(self):
        col = _make_collection(n=3)
        ids = [r.scenario_id for r in col]
        assert len(ids) == 3
        assert len(set(ids)) == 3  # all unique


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

def _run_all_tests():
    """Execute all test classes manually without pytest."""
    import traceback

    test_classes = [TestBoundaryEnvelopeRoundTrip, TestBoundaryEnvelopeCollection]
    passed = failed = 0

    for cls in test_classes:
        instance = cls()
        for name in dir(instance):
            if not name.startswith("test_"):
                continue
            method = getattr(instance, name)
            # inject tmp_path for collection tests that need a temp directory
            import inspect
            sig = inspect.signature(method)
            kwargs = {}
            if "tmp_path" in sig.parameters:
                tmp_dir = Path(tempfile.mkdtemp())
                kwargs["tmp_path"] = tmp_dir
            try:
                method(**kwargs)
                print(f"  PASS  {cls.__name__}::{name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {cls.__name__}::{name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    return failed == 0


if __name__ == "__main__":
    import sys
    ok = _run_all_tests()
    sys.exit(0 if ok else 1)
