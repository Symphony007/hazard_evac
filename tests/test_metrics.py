"""Unit and integration tests for metrics, statistical CIs, and experiment runner."""

from __future__ import annotations

import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import scipy.stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig, HazardConfig, ScenarioConfig, StudyAreaConfig, load_config
from src.experiment.runner import run_single
from src.graph.loader import get_or_build_graph
from src.metrics.metrics import per_agent_frame, run_metrics
from src.metrics.stats import mean_ci, paired_diff_ci
from src.scenario import exit_spread_report
from src.sim.single_agent import AgentResult


# ---------------------------------------------------------------------------
# Tests 1-2: per_agent_frame & run_metrics
# ---------------------------------------------------------------------------

def test_per_agent_frame_and_run_metrics():
    """Verify metrics calculation on synthetic AgentResult samples."""
    results = [
        AgentResult(
            agent_id=0,
            router_name="static",
            safe=True,
            trapped_reason=None,
            depart_time_s=0.0,
            finish_time_s=600.0,
            evac_time_s=600.0,  # 10 min
            path_nodes=[1, 2, 3],
            path_length_m=1000.0,
            n_replans=0,
            compute_ms=5.0,
            trap_time_s=None,
        ),
        AgentResult(
            agent_id=1,
            router_name="static",
            safe=True,
            trapped_reason=None,
            depart_time_s=0.0,
            finish_time_s=1200.0,
            evac_time_s=1200.0,  # 20 min
            path_nodes=[2, 4],
            path_length_m=2000.0,
            n_replans=0,
            compute_ms=7.0,
            trap_time_s=None,
        ),
        AgentResult(
            agent_id=2,
            router_name="static",
            safe=False,
            trapped_reason="no_route",
            depart_time_s=0.0,
            finish_time_s=None,
            evac_time_s=None,
            path_nodes=[3],
            path_length_m=0.0,
            n_replans=0,
            compute_ms=2.0,
            trap_time_s=0.0,
        ),
        AgentResult(
            agent_id=3,
            router_name="static",
            safe=False,
            trapped_reason="edge_closed_en_route",
            depart_time_s=0.0,
            finish_time_s=None,
            evac_time_s=None,
            path_nodes=[4, 5],
            path_length_m=500.0,
            n_replans=0,
            compute_ms=4.0,
            trap_time_s=300.0,
        ),
    ]

    baseline_lengths = {1: 1000.0, 2: 1000.0, 3: 500.0, 4: 500.0}

    df = per_agent_frame(results, baseline_lengths)

    # Frame assertions
    assert len(df) == 4
    assert "max_congestion" in df.columns
    assert np.isnan(df["max_congestion"]).all()
    assert df.loc[0, "path_overhead"] == pytest.approx(1.0)
    assert df.loc[1, "path_overhead"] == pytest.approx(2.0)
    assert np.isnan(df.loc[2, "path_overhead"])
    assert np.isnan(df.loc[3, "path_overhead"])

    # Aggregated run metrics
    m = run_metrics(df)
    assert m["pct_safe"] == pytest.approx(50.0)
    assert m["mean_evac_min"] == pytest.approx(15.0)
    assert m["last_evac_min"] == pytest.approx(20.0)
    assert m["p90_evac_min"] == pytest.approx(19.0)
    assert m["mean_path_overhead"] == pytest.approx(1.5)
    assert m["mean_compute_ms"] == pytest.approx(4.5)
    assert m["n_trapped_no_route"] == 1
    assert m["n_trapped_en_route"] == 1
    assert m["n_agents"] == 4


def test_run_metrics_zero_safe_does_not_crash():
    """Verify that 0 safe agents results in NaNs for safe-only metrics without raising."""
    results = [
        AgentResult(
            agent_id=i,
            router_name="static",
            safe=False,
            trapped_reason="no_route",
            depart_time_s=0.0,
            finish_time_s=None,
            evac_time_s=None,
            path_nodes=[i],
            path_length_m=0.0,
            n_replans=0,
            compute_ms=1.5,
        )
        for i in range(5)
    ]
    df = per_agent_frame(results, {i: 100.0 for i in range(5)})
    m = run_metrics(df)

    assert m["pct_safe"] == pytest.approx(0.0)
    assert np.isnan(m["mean_evac_min"])
    assert np.isnan(m["p90_evac_min"])
    assert np.isnan(m["last_evac_min"])
    assert np.isnan(m["mean_path_overhead"])
    assert m["mean_compute_ms"] == pytest.approx(1.5)
    assert m["n_trapped_no_route"] == 5
    assert m["n_trapped_en_route"] == 0


# ---------------------------------------------------------------------------
# Tests 3-4: Statistical helpers (mean_ci & paired_diff_ci)
# ---------------------------------------------------------------------------

def test_mean_ci_against_scipy():
    """Verify Student-t CI matches scipy.stats exact interval."""
    data = [10.0, 12.0, 14.0, 16.0, 18.0]
    m, low, high = mean_ci(data, level=0.95)

    expected_low, expected_high = scipy.stats.t.interval(
        0.95, df=len(data) - 1, loc=np.mean(data), scale=scipy.stats.sem(data)
    )

    assert m == pytest.approx(14.0)
    assert low == pytest.approx(expected_low)
    assert high == pytest.approx(expected_high)

    # Fewer than 2 values returns NaN CI bounds
    m1, low1, high1 = mean_ci([42.0])
    assert m1 == 42.0
    assert np.isnan(low1) and np.isnan(high1)

    # With NaNs
    m_nan, low_nan, high_nan = mean_ci([np.nan, 42.0])
    assert m_nan == 42.0
    assert np.isnan(low_nan) and np.isnan(high_nan)


def test_paired_diff_ci():
    """Verify paired difference CI sign (a - b) and metrics."""
    a = [10.0, 20.0, 30.0]
    b = [8.0, 15.0, 25.0]
    # diffs = [2.0, 5.0, 5.0], mean = 4.0

    res = paired_diff_ci(a, b, level=0.95)
    assert res.mean_diff == pytest.approx(4.0)
    assert res.ci_low < 4.0 < res.ci_high
    assert res.n == 3
    assert 0.0 <= res.p_ttest <= 1.0
    assert 0.0 <= res.p_wilcoxon <= 1.0


# ---------------------------------------------------------------------------
# Test 5: Exit spread diagnostics on synthetic circle
# ---------------------------------------------------------------------------

def test_exit_spread_report():
    """Verify exit_spread_report on synthetic geometric node rings."""
    G = nx.MultiDiGraph()
    center_lat, center_lon = 22.5, 88.35

    # 1. 8 evenly spaced nodes on a circle (every 45 degrees)
    r = 0.01  # degrees
    for i in range(8):
        angle_rad = math.radians(i * 45.0)
        G.add_node(
            i,
            y=center_lat + r * math.sin(angle_rad),
            x=center_lon + r * math.cos(angle_rad),
        )

    rep_even = exit_spread_report(G, exits=list(range(8)), center_lat=center_lat, center_lon=center_lon)
    assert rep_even["n_distinct_sectors"] == 8
    assert rep_even["largest_gap_deg"] == pytest.approx(45.0, abs=0.5)
    assert rep_even["angular_coverage_deg"] == pytest.approx(315.0, abs=0.5)

    # 2. 3 tightly clustered nodes in sector 0
    G_clustered = nx.MultiDiGraph()
    for i, deg in enumerate([10.0, 12.0, 14.0], start=100):
        rad = math.radians(deg)
        G_clustered.add_node(
            i,
            y=center_lat + r * math.sin(rad),
            x=center_lon + r * math.cos(rad),
        )

    rep_cluster = exit_spread_report(G_clustered, exits=[100, 101, 102], center_lat=center_lat, center_lon=center_lon)
    assert rep_cluster["n_distinct_sectors"] == 1
    assert rep_cluster["largest_gap_deg"] == pytest.approx(356.0, abs=0.5)
    assert rep_cluster["angular_coverage_deg"] == pytest.approx(4.0, abs=0.5)


# ---------------------------------------------------------------------------
# Tests 6-7: Experiment runner determinism and resume capability
# ---------------------------------------------------------------------------

def test_runner_determinism_and_resume():
    """Verify run_single determinism and resume skipping behavior."""
    graph_cfg = load_config(PROJECT_ROOT / "configs" / "study_area.yaml", StudyAreaConfig)
    hazard_cfg = load_config(PROJECT_ROOT / "configs" / "hazard.yaml", HazardConfig)
    scenario_cfg = replace(
        load_config(PROJECT_ROOT / "configs" / "scenario.yaml", ScenarioConfig),
        n_evacuees=25,  # Small count for test execution
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        exp_cfg = ExperimentConfig(
            n_seeds=1,
            output_dir=tmpdir,
        )

        G = get_or_build_graph(graph_cfg)

        # First execution
        rec1 = run_single(
            graph_cfg=graph_cfg,
            hazard_cfg=hazard_cfg,
            scenario_cfg=scenario_cfg,
            speed=30.0,
            seed_index=0,
            exp_cfg=exp_cfg,
            G=G,
            force=True,
        )

        raw_csv = Path(tmpdir) / "raw" / "agents_speed30_seed0.csv"
        raw_json = Path(tmpdir) / "raw" / "run_speed30_seed0.json"
        assert raw_csv.exists()
        assert raw_json.exists()
        mtime_initial = raw_csv.stat().st_mtime_ns

        # Second execution with force=False (resume should trigger)
        rec2 = run_single(
            graph_cfg=graph_cfg,
            hazard_cfg=hazard_cfg,
            scenario_cfg=scenario_cfg,
            speed=30.0,
            seed_index=0,
            exp_cfg=exp_cfg,
            G=G,
            force=False,
        )

        # File was not modified (resumed from cache)
        assert raw_csv.stat().st_mtime_ns == mtime_initial

        # Determinism: metrics match exactly
        for router in ("static", "reactive", "time_dependent"):
            m1 = rec1["metrics"][router]
            m2 = rec2["metrics"][router]
            for key in ("pct_safe", "mean_evac_min", "mean_compute_ms", "n_trapped_en_route"):
                val1 = m1[key]
                val2 = m2[key]
                if val1 is None or (isinstance(val1, float) and np.isnan(val1)):
                    assert val2 is None or (isinstance(val2, float) and np.isnan(val2))
                else:
                    assert val1 == pytest.approx(val2), f"Mismatch in {router}.{key}"

