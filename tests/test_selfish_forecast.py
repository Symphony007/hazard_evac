"""Unit tests for SelfishForecastRouter, common-safe metrics, and router-aware resume."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CongestionConfig,
    CoordinatedConfig,
    HazardConfig,
    ScenarioConfig,
    StudyAreaConfig,
    TrafficConfig,
    load_config,
)
from src.experiment.ladder_runner import run_single_ladder
from src.graph.loader import get_or_build_graph
from src.hazard.flood import FloodHazard
from src.routing.base import Evacuee, RoutingParams
from src.routing.planners import (
    SelfishForecastRouter,
    SelfishRouter,
    StaticRouter,
    TimeDependentRouter,
)
from src.routing.search import earliest_arrival_search
from src.scenario import build_exits, generate_agents
from src.sim.multi_agent import simulate_population
from tests.test_multiagent import StubMultiAgentHazard


# ---------------------------------------------------------------------------
# Test 1: Avoids a flooding edge like time_dependent
# ---------------------------------------------------------------------------

def test_selfish_forecast_avoids_flooding_edge():
    """Verify that SelfishForecastRouter avoids an edge closing en-route."""
    G = nx.MultiDiGraph()
    # Diamond graph: 0 (A), 1 (B), 2 (C), 3 (D)
    # Route 1: 0 -> 1 -> 3 (30s + 30s = 60s, but 1->3 closes at 50s)
    # Route 2: 0 -> 2 -> 3 (40s + 40s = 80s, never closes)
    for n in range(4):
        G.add_node(n, y=0.0, x=float(n))
    G.add_edge(0, 1, key=0, free_flow_time=30.0, flow_capacity=12.0, length=1000.0, capacity=10)
    G.add_edge(1, 3, key=0, free_flow_time=30.0, flow_capacity=12.0, length=1000.0, capacity=10)
    G.add_edge(0, 2, key=0, free_flow_time=40.0, flow_capacity=12.0, length=1200.0, capacity=10)
    G.add_edge(2, 3, key=0, free_flow_time=40.0, flow_capacity=12.0, length=1200.0, capacity=10)

    # Edge (1, 3, 0) floods at 50s
    hazard = StubMultiAgentHazard(edge_arrivals={(1, 3, 0): 50.0})
    router = SelfishForecastRouter(replan_interval_s=60.0)
    agent = Evacuee(id=1, origin=0, depart_time_s=0.0)
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)

    pop_res = simulate_population(G, hazard, router, [agent], exits={3}, params=params)
    res = pop_res.agent_results[0]

    assert res.safe is True
    # Successfully detoured via node 2
    assert 2 in res.path_nodes
    assert 1 not in res.path_nodes
    assert res.evac_time_s == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Test 2: Uses live congestion to divert traffic
# ---------------------------------------------------------------------------

def test_selfish_forecast_uses_live_congestion():
    """Verify that SelfishForecastRouter diverts agents under congestion."""
    G = nx.MultiDiGraph()
    # Route A: 0 -> 1 -> 3 (tau = 10 + 10 = 20s)
    # Route B: 0 -> 2 -> 3 (tau = 16 + 16 = 32s)
    # Headway h = 10s on all edges (flow_capacity = 6.0 veh/min)
    for n in range(4):
        G.add_node(n, y=0.0, x=float(n))
    G.add_edge(0, 1, key=0, free_flow_time=10.0, flow_capacity=6.0, capacity=10, length=100.0)
    G.add_edge(1, 3, key=0, free_flow_time=10.0, flow_capacity=6.0, capacity=10, length=100.0)
    G.add_edge(0, 2, key=0, free_flow_time=16.0, flow_capacity=6.0, capacity=10, length=160.0)
    G.add_edge(2, 3, key=0, free_flow_time=16.0, flow_capacity=6.0, capacity=10, length=160.0)

    hazard = StubMultiAgentHazard()
    agents = [Evacuee(id=i, origin=0, depart_time_s=0.0) for i in range(1, 7)]
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)

    # TimeDependent sends everyone via route A (node 1)
    pop_td = simulate_population(G, hazard, TimeDependentRouter(), agents, exits={3}, params=params)
    b_users_td = sum(1 for r in pop_td.agent_results if 2 in r.path_nodes)
    assert b_users_td == 0

    # SelfishForecast diverts to route B (node 2) when route A becomes congested
    pop_sf = simulate_population(G, hazard, SelfishForecastRouter(replan_interval_s=60.0), agents, exits={3}, params=params)
    b_users_sf = sum(1 for r in pop_sf.agent_results if 2 in r.path_nodes)
    assert b_users_sf >= 1


# ---------------------------------------------------------------------------
# Test 3: Infinite capacity regression on cached graph
# ---------------------------------------------------------------------------

def test_infinite_capacity_regression():
    """Verify SelfishForecastRouter matches TimeDependentRouter when capacity is infinite."""
    cfg = load_config(PROJECT_ROOT / "configs" / "study_area.yaml", StudyAreaConfig)
    G = get_or_build_graph(cfg).copy()

    for _, _, d in G.edges(data=True):
        d["flow_capacity"] = 1e9

    hazard_cfg = load_config(PROJECT_ROOT / "configs" / "hazard.yaml", HazardConfig)
    h_cfg = HazardConfig(
        spread_speed_m_per_min=60.0,
        seed=3000,
        terrain_seed=hazard_cfg.terrain_seed,
        horizon_min=hazard_cfg.horizon_min,
    )
    hazard = FloodHazard(G, h_cfg)
    scenario_cfg = load_config(PROJECT_ROOT / "configs" / "scenario.yaml", ScenarioConfig)
    exits, _ = build_exits(
        G=G,
        hazard=hazard,
        scenario_cfg=scenario_cfg,
        center_lat=cfg.center_lat,
        center_lon=cfg.center_lon,
        horizon_min=h_cfg.horizon_min,
    )

    rng = np.random.default_rng(4000)
    agents = generate_agents(
        G=G,
        hazard=hazard,
        exits=exits,
        scenario_cfg=ScenarioConfig(n_evacuees=50),
        rng=rng,
    )
    params = RoutingParams(
        vehicle_speed_scale=scenario_cfg.vehicle_speed_scale,
        sf_floor=scenario_cfg.sf_floor,
    )

    pop_td = simulate_population(G, hazard, TimeDependentRouter(), agents, exits, params)
    pop_sf = simulate_population(G, hazard, SelfishForecastRouter(replan_interval_s=60.0), agents, exits, params)

    for r_td, r_sf in zip(pop_td.agent_results, pop_sf.agent_results):
        assert r_td.safe == r_sf.safe
        if r_td.safe:
            assert r_sf.evac_time_s == pytest.approx(r_td.evac_time_s, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 4: extra_delay=None leaves time_dependent results unchanged
# ---------------------------------------------------------------------------

def test_extra_delay_none_leaves_time_dependent_unchanged():
    """Verify that extra_delay=None in earliest_arrival_search exactly matches baseline."""
    G = nx.MultiDiGraph()
    G.add_node(0, y=0.0, x=0.0)
    G.add_node(1, y=0.0, x=1.0)
    G.add_node(2, y=0.0, x=2.0)
    G.add_edge(0, 1, key=0, free_flow_time=15.0, flow_capacity=12.0, length=150.0, capacity=10)
    G.add_edge(1, 2, key=0, free_flow_time=25.0, flow_capacity=12.0, length=250.0, capacity=10)

    hazard = StubMultiAgentHazard()
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)

    r_baseline = earliest_arrival_search(G, hazard, 0, 0.0, {2}, params, mode="time_dependent")
    r_explicit = earliest_arrival_search(G, hazard, 0, 0.0, {2}, params, mode="time_dependent", extra_delay=None)

    assert r_baseline is not None and r_explicit is not None
    assert r_baseline.nodes == r_explicit.nodes
    assert r_baseline.planned_arrival_s == pytest.approx(r_explicit.planned_arrival_s)


# ---------------------------------------------------------------------------
# Test 5: Determinism across runs
# ---------------------------------------------------------------------------

def test_determinism():
    """Verify identical runs of SelfishForecastRouter produce identical outcomes."""
    G = nx.MultiDiGraph()
    G.add_node(0, y=0.0, x=0.0)
    G.add_node(1, y=0.0, x=1.0)
    G.add_node(2, y=0.0, x=2.0)
    G.add_edge(0, 1, key=0, free_flow_time=10.0, flow_capacity=6.0, length=100.0, capacity=10)
    G.add_edge(1, 2, key=0, free_flow_time=10.0, flow_capacity=6.0, length=100.0, capacity=10)

    hazard = StubMultiAgentHazard()
    agents = [Evacuee(id=i, origin=0, depart_time_s=0.0) for i in range(5)]
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)
    router = SelfishForecastRouter(replan_interval_s=60.0)

    pop1 = simulate_population(G, hazard, router, agents, exits={2}, params=params)
    pop2 = simulate_population(G, hazard, router, agents, exits={2}, params=params)

    times1 = [r.finish_time_s for r in pop1.agent_results]
    times2 = [r.finish_time_s for r in pop2.agent_results]
    assert times1 == times2


# ---------------------------------------------------------------------------
# Test 6: mean_evac_min_common_safe on a hand-built example
# ---------------------------------------------------------------------------

def test_mean_evac_min_common_safe_calculation():
    """Verify common-safe evacuation time correctly filters to mutual survivors."""
    df_a = pd.DataFrame([
        {"agent_id": 1, "safe": True, "evac_time_s": 100.0},
        {"agent_id": 2, "safe": True, "evac_time_s": 200.0},
        {"agent_id": 3, "safe": True, "evac_time_s": 300.0},
        {"agent_id": 4, "safe": False, "evac_time_s": np.nan},
    ])

    df_b = pd.DataFrame([
        {"agent_id": 1, "safe": False, "evac_time_s": np.nan},
        {"agent_id": 2, "safe": True, "evac_time_s": 220.0},
        {"agent_id": 3, "safe": True, "evac_time_s": 340.0},
        {"agent_id": 4, "safe": True, "evac_time_s": 400.0},
    ])

    safe_a = set(df_a[df_a["safe"]]["agent_id"])
    safe_b = set(df_b[df_b["safe"]]["agent_id"])
    common_ids = safe_a & safe_b

    # Common safe set must be exactly {2, 3}
    assert common_ids == {2, 3}

    evac_a = df_a[df_a["agent_id"].isin(common_ids)]["evac_time_s"].values / 60.0
    evac_b = df_b[df_b["agent_id"].isin(common_ids)]["evac_time_s"].values / 60.0

    mean_a = float(np.mean(evac_a))  # (200 + 300) / 2 / 60 = 250 / 60 = 4.1667 min
    mean_b = float(np.mean(evac_b))  # (220 + 340) / 2 / 60 = 280 / 60 = 4.6667 min

    assert mean_a == pytest.approx(250.0 / 60.0)
    assert mean_b == pytest.approx(280.0 / 60.0)
    diff = mean_a - mean_b  # -30 / 60 = -0.5 min
    assert diff == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Test 7: Merge test (Router-aware resume)
# ---------------------------------------------------------------------------

def test_router_aware_resume_merge():
    """Verify that merging a missing router matches a from-scratch run exactly."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="test_resume_"))
    try:
        cfg = load_config(PROJECT_ROOT / "configs" / "study_area.yaml", StudyAreaConfig)
        G = get_or_build_graph(cfg)
        h_cfg = load_config(PROJECT_ROOT / "configs" / "hazard.yaml", HazardConfig)
        sc_cfg = load_config(PROJECT_ROOT / "configs" / "scenario.yaml", ScenarioConfig)
        tr_cfg = TrafficConfig(replan_interval_s=60.0)
        co_cfg = CoordinatedConfig()

        # Step 1: Run with only static and time_dependent on 30 agents
        cg_cfg1 = CongestionConfig(
            flood_speed_m_per_min=60.0,
            demand_levels=[30],
            n_seeds=1,
            base_hazard_seed=3000,
            base_scenario_seed=4000,
            routers=["static", "time_dependent"],
            output_dir=str(tmp_dir),
        )

        res1 = run_single_ladder(
            graph_cfg=cfg,
            hazard_cfg=h_cfg,
            scenario_cfg=sc_cfg,
            traffic_cfg=tr_cfg,
            cong_cfg=cg_cfg1,
            coord_cfg=co_cfg,
            demand=30,
            seed_index=0,
            G=G,
            force=True,
        )
        assert set(res1["metrics"].keys()) == {"static", "time_dependent"}

        # Step 2: Resume with static, time_dependent, AND selfish_forecast
        # Router-aware resume must simulate ONLY selfish_forecast and merge
        cg_cfg2 = CongestionConfig(
            flood_speed_m_per_min=60.0,
            demand_levels=[30],
            n_seeds=1,
            base_hazard_seed=3000,
            base_scenario_seed=4000,
            routers=["static", "time_dependent", "selfish_forecast"],
            output_dir=str(tmp_dir),
        )

        res2 = run_single_ladder(
            graph_cfg=cfg,
            hazard_cfg=h_cfg,
            scenario_cfg=sc_cfg,
            traffic_cfg=tr_cfg,
            cong_cfg=cg_cfg2,
            coord_cfg=co_cfg,
            demand=30,
            seed_index=0,
            G=G,
            force=False,
        )
        assert set(res2["metrics"].keys()) == {"static", "time_dependent", "selfish_forecast"}

        # Step 3: Run selfish_forecast from scratch in a fresh directory
        tmp_fresh = Path(tempfile.mkdtemp(prefix="test_fresh_"))
        try:
            cg_cfg_fresh = CongestionConfig(
                flood_speed_m_per_min=60.0,
                demand_levels=[30],
                n_seeds=1,
                base_hazard_seed=3000,
                base_scenario_seed=4000,
                routers=["selfish_forecast"],
                output_dir=str(tmp_fresh),
            )
            res_fresh = run_single_ladder(
                graph_cfg=cfg,
                hazard_cfg=h_cfg,
                scenario_cfg=sc_cfg,
                traffic_cfg=tr_cfg,
                cong_cfg=cg_cfg_fresh,
                coord_cfg=co_cfg,
                demand=30,
                seed_index=0,
                G=G,
                force=True,
            )

            # Compare merged selfish_forecast outcomes against fresh run
            merged_sf = res2["metrics"]["selfish_forecast"]
            fresh_sf = res_fresh["metrics"]["selfish_forecast"]

            assert merged_sf["pct_safe"] == fresh_sf["pct_safe"]
            assert merged_sf["mean_evac_min"] == pytest.approx(fresh_sf["mean_evac_min"], abs=1e-5)
            assert merged_sf["n_trapped_en_route"] == fresh_sf["n_trapped_en_route"]
            assert merged_sf["n_trapped_no_route"] == fresh_sf["n_trapped_no_route"]
        finally:
            shutil.rmtree(tmp_fresh, ignore_errors=True)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

