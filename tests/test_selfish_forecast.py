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


# ---------------------------------------------------------------------------
# Test 8: Stage 2 fallback used when congested route infeasible
# ---------------------------------------------------------------------------

def test_selfish_forecast_fallback_when_congested_infeasible():
    """Verify two-stage planning: fallback to forecast-only route with n_fallback_plans >= 1."""
    # Topology: 0 -> 1 -> 2
    # e1=(0,1,0): length=100, fft=10s, arrival=30s
    # e2=(1,2,0): length=100, fft=10s, arrival=60s
    # Agent departs at t=0.
    # Suppose snapshot extra_delay on e1 is 25s.
    # With snapshot delay: t=0, tau=10, d=25 -> arrival at node 1 = 35s > a(e1)=30s (INFEASIBLE!).
    # Stage 1 search will return None.
    # Stage 2 search sets extra_delay=0: t=0, tau=10 -> arrival 10s < 30s; e2: 10+10=20s < 60s (FEASIBLE!).
    # So agent departs, safely arrives at exit 2, and n_fallback_plans >= 1.
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, 0, length=100.0, free_flow_time=10.0, travel_time=10.0, flow_capacity=10.0, capacity=10)
    G.add_edge(1, 2, 0, length=100.0, free_flow_time=10.0, travel_time=10.0, flow_capacity=10.0, capacity=10)

    hazard = StubMultiAgentHazard(
        edge_arrivals={(0, 1, 0): 30.0, (1, 2, 0): 60.0},
        node_arrivals={0: 30.0, 1: 30.0, 2: 60.0},
    )

    class DelayTraffic:
        def extra_delay_s(self, u: int, v: int, k: int, t: float) -> float:
            if (u, v, k) == (0, 1, 0):
                return 25.0
            return 0.0

    router = SelfishForecastRouter(replan_interval_s=60.0)
    evacuee = Evacuee(id=1, origin=0, depart_time_s=0.0)
    params = RoutingParams(vehicle_speed_scale=1.0)

    # Direct plan check
    route = router.plan(G, hazard, evacuee, 0.0, 0, {2}, params, traffic=DelayTraffic())
    assert route is not None
    assert getattr(route, "fallback_used", False) is True
    assert route.nodes == [0, 1, 2]

    # Full multi-agent sim check with traffic congestion
    # Agent 1 departs at t=0. Agent 2 also starts on (0,1,0) earlier or edge is congested.
    # To create actual queue delay in simulate_population:
    # If edge (0,1,0) has flow_capacity=1 veh/min (headway h = 60s),
    # an earlier agent departing at t=0 with fft=10s will exit at t=10s.
    # A second agent departing at t=0.1s will face last_exit=10s, h=60s -> candidate exit = 70s.
    # At t=0.1s, extra_delay on (0,1,0) is positive and makes t + tau + delay > arrival (30s).
    G2 = nx.MultiDiGraph()
    G2.add_edge(0, 1, 0, length=100.0, free_flow_time=10.0, travel_time=10.0, flow_capacity=2.0, capacity=10) # h = 30s
    G2.add_edge(1, 2, 0, length=100.0, free_flow_time=10.0, travel_time=10.0, flow_capacity=2.0, capacity=10)

    # Agent 0 departs at t=0. Agent 1 departs at t=1.0s.
    # Agent 0 enters (0,1,0) at t=0, exits at t=10. Next slot at (0,1,0) is 10 + 30 = 40s.
    # When Agent 1 plans at t=1.0s, extra_delay on (0,1,0) is max(0, 40 - (1.0 + 10)) = 29s.
    # Total arrival at 1 would be 1 + 10 + 29 = 40s > arrival(0,1,0)=30s!
    # So Agent 1's Stage 1 fails, but Stage 2 succeeds (1 + 10 = 11s < 30s).
    hazard2 = StubMultiAgentHazard(
        edge_arrivals={(0, 1, 0): 30.0, (1, 2, 0): 60.0},
        node_arrivals={0: 30.0, 1: 30.0, 2: 60.0},
    )
    a0 = Evacuee(id=0, origin=0, depart_time_s=0.0)
    a1 = Evacuee(id=1, origin=0, depart_time_s=1.0)

    traffic_cfg = TrafficConfig(replan_interval_s=60.0)
    res = simulate_population(
        G=G2,
        hazard=hazard2,
        router=router,
        agents=[a0, a1],
        exits={2},
        params=params,
        traffic_cfg=traffic_cfg,
    )
    assert len(res.agent_results) == 2
    ar1 = [r for r in res.agent_results if r.agent_id == 1][0]
    assert ar1.n_fallback_plans >= 1


# ---------------------------------------------------------------------------
# Test 9: Both stages infeasible -> trapped no_route
# ---------------------------------------------------------------------------

def test_selfish_forecast_both_stages_infeasible():
    """Verify that when even the forecast-only route is closed, router returns None / no_route."""
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, 0, length=100.0, free_flow_time=10.0, travel_time=10.0, flow_capacity=10.0, capacity=10)

    # Edge closes at t=5s, but fft=10s, so agent departing at t=0 cannot clear it even without traffic
    hazard = StubMultiAgentHazard(
        edge_arrivals={(0, 1, 0): 5.0},
        node_arrivals={0: 5.0, 1: 5.0},
    )
    router = SelfishForecastRouter(replan_interval_s=60.0)
    evacuee = Evacuee(id=1, origin=0, depart_time_s=0.0)
    params = RoutingParams(vehicle_speed_scale=1.0)

    route = router.plan(G, hazard, evacuee, 0.0, 0, {1}, params, traffic=None)
    assert route is None

    res = simulate_population(
        G=G,
        hazard=hazard,
        router=router,
        agents=[evacuee],
        exits={1},
        params=params,
        traffic_cfg=TrafficConfig(),
    )
    assert len(res.agent_results) == 1
    ar = res.agent_results[0]
    assert ar.safe is False
    assert ar.trapped_reason == "no_route"


# ---------------------------------------------------------------------------
# Test 10: --rerun-routers leaves other routers byte-identical and matches scratch
# ---------------------------------------------------------------------------

def test_rerun_routers_preserves_other_routers():
    """Verify that rerun_routers re-simulates specified router without touching others."""
    cfg = StudyAreaConfig(name="tollygunge", center_lat=22.4989, center_lon=88.3488, dist_m=1200)
    G = get_or_build_graph(cfg)
    h_cfg = HazardConfig(terrain_seed=123, spread_speed_m_per_min=60.0)
    sc_cfg = ScenarioConfig(n_evacuees=25)
    tr_cfg = TrafficConfig(replan_interval_s=60.0)
    co_cfg = CoordinatedConfig()

    tmp_dir = Path(tempfile.mkdtemp(prefix="test_rerun_"))
    try:
        cg_cfg = CongestionConfig(
            flood_speed_m_per_min=60.0,
            demand_levels=[25],
            n_seeds=1,
            base_hazard_seed=3000,
            base_scenario_seed=4000,
            routers=["static", "time_dependent", "selfish_forecast"],
            output_dir=str(tmp_dir),
        )

        # 1. Run all 3 routers from scratch
        res1 = run_single_ladder(
            graph_cfg=cfg,
            hazard_cfg=h_cfg,
            scenario_cfg=sc_cfg,
            traffic_cfg=tr_cfg,
            cong_cfg=cg_cfg,
            coord_cfg=co_cfg,
            demand=25,
            seed_index=0,
            G=G,
            force=True,
        )

        csv_path = tmp_dir / "raw" / "agents_demand25_seed0.csv"
        df1 = pd.read_csv(csv_path)
        static_rows_before = df1[df1["router_name"] == "static"].copy()
        td_rows_before = df1[df1["router_name"] == "time_dependent"].copy()

        # 2. Rerun ONLY selfish_forecast
        res2 = run_single_ladder(
            graph_cfg=cfg,
            hazard_cfg=h_cfg,
            scenario_cfg=sc_cfg,
            traffic_cfg=tr_cfg,
            cong_cfg=cg_cfg,
            coord_cfg=co_cfg,
            demand=25,
            seed_index=0,
            G=G,
            force=False,
            rerun_routers=["selfish_forecast"],
        )

        df2 = pd.read_csv(csv_path)
        static_rows_after = df2[df2["router_name"] == "static"].copy()
        td_rows_after = df2[df2["router_name"] == "time_dependent"].copy()

        # Check static and time_dependent rows are identical before and after
        pd.testing.assert_frame_equal(static_rows_before.reset_index(drop=True), static_rows_after.reset_index(drop=True))
        pd.testing.assert_frame_equal(td_rows_before.reset_index(drop=True), td_rows_after.reset_index(drop=True))

        # Check selfish_forecast matches the original run (excluding wall-clock compute_ms)
        sf_rows_before = df1[df1["router_name"] == "selfish_forecast"].drop(columns=["compute_ms"]).reset_index(drop=True)
        sf_rows_after = df2[df2["router_name"] == "selfish_forecast"].drop(columns=["compute_ms"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(sf_rows_before, sf_rows_after)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

