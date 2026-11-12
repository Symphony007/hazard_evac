"""Unit and smoke tests for the Task 7b optimality gap runner."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.config import (
    CoordinatedConfig,
    ExactConfig,
    GapConfig,
    HazardConfig,
    ScenarioConfig,
    StudyAreaConfig,
    TrafficConfig,
    load_config,
)
from src.routing.base import make_routing_params
from scripts.run_gap_experiment import run_single_gap_instance

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Test a: Unit test - RoutingParams equivalence across runners
# ---------------------------------------------------------------------------

def test_routing_params_equivalence():
    """Verify make_routing_params matches default scenario config parameters across runners."""
    scenario_cfg = load_config(PROJECT_ROOT / "configs" / "scenario.yaml", ScenarioConfig)
    params = make_routing_params(scenario_cfg)

    assert params.vehicle_speed_scale == 0.5
    assert params.sf_floor == 0.05
    assert params.vehicle_speed_scale == scenario_cfg.vehicle_speed_scale
    assert params.sf_floor == scenario_cfg.sf_floor


# ---------------------------------------------------------------------------
# Test b: Slow test - Gap runner time_dependent matches Task 6
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_gap_runner_reused_heuristic_matches_task6():
    """For demand 1500, seed 0, time_dependent n_safe matches outputs/task6/raw."""
    task6_csv = PROJECT_ROOT / "outputs" / "task6" / "raw" / "agents_demand1500_seed0.csv"
    if not task6_csv.exists():
        pytest.skip(f"Reference file {task6_csv} not found")

    df_task6 = pd.read_csv(task6_csv)
    expected_safe = int(df_task6[(df_task6["router_name"] == "time_dependent") & (df_task6["safe"] == True)]["safe"].count())

    graph_cfg = load_config(PROJECT_ROOT / "configs" / "study_area.yaml", StudyAreaConfig)
    hazard_cfg = load_config(PROJECT_ROOT / "configs" / "hazard.yaml", HazardConfig)
    scenario_cfg = load_config(PROJECT_ROOT / "configs" / "scenario.yaml", ScenarioConfig)
    traffic_cfg = load_config(PROJECT_ROOT / "configs" / "traffic.yaml", TrafficConfig)
    coord_cfg = load_config(PROJECT_ROOT / "configs" / "coordinated.yaml", CoordinatedConfig)
    exact_cfg = load_config(PROJECT_ROOT / "configs" / "exact_gap.yaml", ExactConfig)
    gap_cfg = load_config(PROJECT_ROOT / "configs" / "gap.yaml", GapConfig)

    # Temporary output directory for test
    tmp_out = PROJECT_ROOT / "outputs" / "test_gap_tmp"
    gap_cfg_tmp = replace(gap_cfg, output_dir=str(tmp_out.relative_to(PROJECT_ROOT)))

    record = run_single_gap_instance(
        graph_cfg=graph_cfg,
        hazard_cfg=hazard_cfg,
        scenario_cfg=scenario_cfg,
        traffic_cfg=traffic_cfg,
        coord_cfg=coord_cfg,
        exact_cfg=exact_cfg,
        gap_cfg=gap_cfg_tmp,
        demand=1500,
        seed_index=0,
        force=False,
        task6_raw_dir=PROJECT_ROOT / "outputs" / "task6" / "raw",
    )

    td_safe = record["heuristic_metrics"]["time_dependent"]["n_safe"]
    assert td_safe == expected_safe


# ---------------------------------------------------------------------------
# Test c: Slow smoke test - Full instance pipeline on demand 100, seed 0
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_gap_runner_instance_smoke_demand100():
    """Per-instance function runs end-to-end for demand 100, seed 0 and satisfies bounds."""
    graph_cfg = load_config(PROJECT_ROOT / "configs" / "study_area.yaml", StudyAreaConfig)
    hazard_cfg = load_config(PROJECT_ROOT / "configs" / "hazard.yaml", HazardConfig)
    scenario_cfg = load_config(PROJECT_ROOT / "configs" / "scenario.yaml", ScenarioConfig)
    traffic_cfg = load_config(PROJECT_ROOT / "configs" / "traffic.yaml", TrafficConfig)
    coord_cfg = load_config(PROJECT_ROOT / "configs" / "coordinated.yaml", CoordinatedConfig)
    exact_cfg = load_config(PROJECT_ROOT / "configs" / "exact_gap.yaml", ExactConfig)
    gap_cfg = load_config(PROJECT_ROOT / "configs" / "gap.yaml", GapConfig)

    tmp_out = PROJECT_ROOT / "outputs" / "test_gap_smoke_tmp"
    gap_cfg_tmp = replace(gap_cfg, output_dir=str(tmp_out.relative_to(PROJECT_ROOT)))

    record = run_single_gap_instance(
        graph_cfg=graph_cfg,
        hazard_cfg=hazard_cfg,
        scenario_cfg=scenario_cfg,
        traffic_cfg=traffic_cfg,
        coord_cfg=coord_cfg,
        exact_cfg=exact_cfg,
        gap_cfg=gap_cfg_tmp,
        demand=100,
        seed_index=0,
        force=False,
        task6_raw_dir=PROJECT_ROOT / "outputs" / "task6" / "raw",
    )

    # Check required fields
    required_fields = [
        "demand",
        "seed_index",
        "n_agents",
        "ceiling_n",
        "ceiling_pct",
        "n_safe_ceil_model",
        "pct_safe_ceil_model",
        "n_safe_floor_model",
        "pct_safe_floor_model",
        "ub_n",
        "ub_pct",
        "exact_plan_realized_n",
        "exact_plan_realized_pct",
        "build_time_ceil_s",
        "solve_time_ceil_s",
        "n_nodes_ceil",
        "n_arcs_ceil",
        "build_time_floor_s",
        "solve_time_floor_s",
        "n_nodes_floor",
        "n_arcs_floor",
        "heuristic_metrics",
        "horizon_warnings",
    ]
    for field in required_fields:
        assert field in record, f"Missing field {field} in record"

    assert record["demand"] == 100
    assert record["seed_index"] == 0
    assert record["n_agents"] == 100

    # Bounds check
    assert record["exact_plan_realized_n"] <= record["ceiling_n"]
    assert record["ub_n"] >= record["exact_plan_realized_n"] - 1

    print(f"\n[Smoke demand 100 seed 0 outcomes]")
    print(f"  Ceiling n:      {record['ceiling_n']}")
    print(f"  UB n:           {record['ub_n']}")
    print(f"  Floor model n:  {record['n_safe_floor_model']}")
    print(f"  Ceil model n:   {record['n_safe_ceil_model']}")
    print(f"  Realized n:     {record['exact_plan_realized_n']}")
