"""Unit tests for exact time-expanded min-cost flow model and solver."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CoordinatedConfig,
    ExactConfig,
    HazardConfig,
    ScenarioConfig,
    StudyAreaConfig,
    TrafficConfig,
)
from src.exact.time_expanded import ExactModel, build_network, solve
from src.graph.loader import get_or_build_graph
from src.hazard.flood import FloodHazard
from src.routing.base import Evacuee, RoutingParams
from src.routing.planners import TimeDependentRouter
from src.scenario import build_exits, generate_agents
from src.sim.multi_agent import simulate_population
from src.sim.single_agent import simulate_agent
from tests.test_multiagent import StubMultiAgentHazard


# ---------------------------------------------------------------------------
# Test 1: Chain S -> A -> D, 5 agents, minimal travel time matching simulator
# ---------------------------------------------------------------------------

def test_chain_five_agents_travel_time():
    """Chain S->A->D, tau=10s per edge, flow_capacity=6 veh/min (headway 10s).

    dt = 10s. 5 agents departing at t=0. No hazard.
    In simulator:
      Agent 0: departs 0, S->A (exit 10), A->D (exit 20) -> 20s
      Agent 1: departs 0, waits at S (exit 20), A->D (exit 30) -> 30s
      Agent 2: exit 40s
      Agent 3: exit 50s
      Agent 4: exit 60s
      Total travel time = 20 + 30 + 40 + 50 + 60 = 200s (20 steps).
    Exact model with dt=10 must yield n_safe = 5 and total_travel_steps = 20.
    """
    G = nx.MultiDiGraph()
    G.add_edge("S", "A", 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)
    G.add_edge("A", "D", 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)

    hazard = StubMultiAgentHazard()
    agents = [Evacuee(id=i, origin="S", depart_time_s=0.0) for i in range(5)]
    params = RoutingParams(vehicle_speed_scale=1.0)
    cfg = ExactConfig(dt_s=10, horizon_s=200, big_m_factor=10)

    model = build_network(G, hazard, agents, exits={"D"}, params=params, exact_cfg=cfg, prune=True)
    res = solve(model, exact_cfg=cfg)

    assert res.status == "OPTIMAL"
    assert res.n_safe == 5
    assert res.pct_safe_model == 100.0
    assert res.total_travel_steps == 20  # 20 steps * 10s = 200s


# ---------------------------------------------------------------------------
# Test 2: Same chain with edge A->D closing at 45s: n_safe = 3
# ---------------------------------------------------------------------------

def test_chain_closing_edge():
    """Same chain S->A->D with edge A->D closing at t = 45s.

    Agent 0 arrives at D at t=20s (< 45s) -> SAFE
    Agent 1 arrives at D at t=30s (< 45s) -> SAFE
    Agent 2 arrives at D at t=40s (< 45s) -> SAFE
    Agent 3 would clear A->D at t=50s (>= 45s) -> INFEASIBLE / TRAPPED
    Agent 4 would clear A->D at t=60s (>= 45s) -> INFEASIBLE / TRAPPED
    Exact model must yield n_safe = 3.
    """
    G = nx.MultiDiGraph()
    G.add_edge("S", "A", 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)
    G.add_edge("A", "D", 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)

    hazard = StubMultiAgentHazard(
        edge_arrivals={("A", "D", 0): 45.0},
    )
    agents = [Evacuee(id=i, origin="S", depart_time_s=0.0) for i in range(5)]
    params = RoutingParams(vehicle_speed_scale=1.0)
    cfg = ExactConfig(dt_s=10, horizon_s=200, big_m_factor=10)

    model = build_network(G, hazard, agents, exits={"D"}, params=params, exact_cfg=cfg, prune=True)
    res = solve(model, exact_cfg=cfg)

    assert res.status == "OPTIMAL"
    assert res.n_safe == 3


# ---------------------------------------------------------------------------
# Test 3: Two parallel routes, 10 agents, exact optimum <= simulator heuristics
# ---------------------------------------------------------------------------

def test_parallel_routes_optimum_bounds_heuristics():
    """Two parallel routes from S (0) to D (3):

    Route 1: 0 -> 1 -> 3, tau = 10 + 10 = 20s, flow = 6 veh/min (1 veh / 10s step, or 1 veh / 2 dt steps).
    Route 2: 0 -> 2 -> 3, tau = 25 + 25 = 50s, flow = 60 veh/min (5 veh / 5s step).
    10 agents departing at t=0, dt = 5.
    Optimum travel time must be <= realized travel time of heuristics (allowing 1 step tolerance for dt rounding).
    """
    G = nx.MultiDiGraph()
    # Route 1: fast but bottlenecked (0 -> 1 -> 3)
    G.add_edge(0, 1, 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0, capacity=10)
    G.add_edge(1, 3, 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0, capacity=10)
    # Route 2: slower but wide (0 -> 2 -> 3)
    G.add_edge(0, 2, 0, free_flow_time=25.0, travel_time=25.0, flow_capacity=60.0, length=250.0, capacity=25)
    G.add_edge(2, 3, 0, free_flow_time=25.0, travel_time=25.0, flow_capacity=60.0, length=250.0, capacity=25)

    hazard = StubMultiAgentHazard()
    agents = [Evacuee(id=i, origin=0, depart_time_s=0.0) for i in range(10)]
    params = RoutingParams(vehicle_speed_scale=1.0)
    cfg = ExactConfig(dt_s=5, horizon_s=300, big_m_factor=10)

    model = build_network(G, hazard, agents, exits={3}, params=params, exact_cfg=cfg, prune=True)
    res = solve(model, exact_cfg=cfg)

    assert res.status == "OPTIMAL"
    assert res.n_safe == 10
    exact_travel_s = res.total_travel_steps * cfg.dt_s

    # Test simulator heuristics on the same setup
    # SelfishRouter & TimeDependentRouter
    from src.routing.planners import SelfishRouter, TimeDependentRouter
    tr_cfg = TrafficConfig(replan_interval_s=60.0)
    sim_res_td = simulate_population(G, hazard, TimeDependentRouter(), agents, {3}, params, tr_cfg)
    sim_res_sf = simulate_population(G, hazard, SelfishRouter(replan_interval_s=5.0), agents, {3}, params, tr_cfg)

    total_td_time = sum(r.evac_time_s for r in sim_res_td.agent_results if r.safe)
    total_sf_time = sum(r.evac_time_s for r in sim_res_sf.agent_results if r.safe)

    # Optimal total time must be <= heuristic time (with small tolerance for dt discretization rounding)
    assert exact_travel_s <= total_td_time + 10 * cfg.dt_s
    assert exact_travel_s <= total_sf_time + 10 * cfg.dt_s


# ---------------------------------------------------------------------------
# Test 4: Unsaveable agent excluded from n_safe; others saved
# ---------------------------------------------------------------------------

def test_unsaveable_agent_handling():
    """An agent originating at an isolated node is unsaved; saveable agents are saved."""
    G = nx.MultiDiGraph()
    G.add_edge("S", "D", 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=12.0, length=100.0)
    G.add_node("ISOLATED")

    hazard = StubMultiAgentHazard()
    agents = [
        Evacuee(id=0, origin="S", depart_time_s=0.0),
        Evacuee(id=1, origin="ISOLATED", depart_time_s=0.0),
    ]
    params = RoutingParams(vehicle_speed_scale=1.0)
    cfg = ExactConfig(dt_s=5, horizon_s=100, big_m_factor=10)

    model = build_network(G, hazard, agents, exits={"D"}, params=params, exact_cfg=cfg, prune=True)
    res = solve(model, exact_cfg=cfg)

    assert res.status == "OPTIMAL"
    assert res.n_agents == 2
    assert res.n_safe == 1
    assert res.pct_safe_model == 50.0


# ---------------------------------------------------------------------------
# Test 5: Pruning does not change the optimum
# ---------------------------------------------------------------------------

def test_pruning_preserves_optimal_solution():
    """Compare n_safe and total_travel_steps with and without pruning on a toy network."""
    G = nx.MultiDiGraph()
    G.add_edge("S", "A", 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=12.0, length=100.0)
    G.add_edge("A", "D", 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=12.0, length=100.0)
    G.add_edge("A", "DEAD_END", 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=12.0, length=100.0)

    hazard = StubMultiAgentHazard(edge_arrivals={("A", "D", 0): 50.0})
    agents = [Evacuee(id=i, origin="S", depart_time_s=i * 5.0) for i in range(4)]
    params = RoutingParams(vehicle_speed_scale=1.0)
    cfg = ExactConfig(dt_s=5, horizon_s=100, big_m_factor=10)

    model_pruned = build_network(G, hazard, agents, exits={"D"}, params=params, exact_cfg=cfg, prune=True)
    res_pruned = solve(model_pruned, exact_cfg=cfg)

    model_unpruned = build_network(G, hazard, agents, exits={"D"}, params=params, exact_cfg=cfg, prune=False)
    res_unpruned = solve(model_unpruned, exact_cfg=cfg)

    assert res_pruned.n_safe == res_unpruned.n_safe
    assert res_pruned.total_travel_steps == res_unpruned.total_travel_steps
    assert model_pruned.n_arcs < model_unpruned.n_arcs


# ---------------------------------------------------------------------------
# Test 6: Infinite capacity sanity check on real cached graph
# ---------------------------------------------------------------------------

def test_infinite_capacity_real_graph_matches_oracle():
    """Infinite capacity sanity check on the real cached road network graph.

    With flow_capacity set to an arbitrarily large number, congestion queuing is absent.
    The exact min-cost flow problem reduces to finding earliest-arrival / shortest paths
    for each agent independently, which should match the single-agent time_dependent
    oracle within +/- 1 agent.
    Explanation of minor discrepancy (+/- 1 agent):
      In the continuous simulator, edge travel times tau are continuous floats.
      In the exact time-expanded model, each edge traversal duration is rounded UP
      to integer steps m = ceil(tau / dt). On a long path through multiple edges,
      accumulated upward rounding can push an agent's arrival at an edge marginally past
      its flood closure time a(e), which can occasionally cause an agent that is safe in
      continuous time to be deemed unfeasible in discrete time.
    """
    study_cfg = StudyAreaConfig(name="tollygunge", center_lat=22.4989, center_lon=88.3488, dist_m=1200)
    G_orig = get_or_build_graph(study_cfg)
    G = G_orig.copy()

    # Override flow capacity to huge value
    for u, v, k in G.edges(keys=True):
        G.edges[u, v, k]["flow_capacity"] = 100_000.0

    h_cfg = HazardConfig(terrain_seed=123, spread_speed_m_per_min=60.0, seed=3000)
    hazard = FloodHazard(G, h_cfg)

    sc_cfg = ScenarioConfig(n_evacuees=30)
    exits, _ = build_exits(
        G=G,
        hazard=hazard,
        scenario_cfg=sc_cfg,
        center_lat=study_cfg.center_lat,
        center_lon=study_cfg.center_lon,
        horizon_min=h_cfg.horizon_min,
    )
    rng = np.random.default_rng(4000)
    agents = generate_agents(G, hazard, exits, sc_cfg, rng=rng)
    params = RoutingParams(vehicle_speed_scale=sc_cfg.vehicle_speed_scale, sf_floor=sc_cfg.sf_floor)

    # 1. Oracle time_dependent
    td_router = TimeDependentRouter()
    n_oracle_safe = sum(1 for a in agents if simulate_agent(G, hazard, td_router, a, exits, params).safe)

    # 2. Exact min-cost flow
    exact_cfg = ExactConfig(dt_s=5, horizon_s=2400, big_m_factor=10)
    model = build_network(G, hazard, agents, exits, params, exact_cfg, prune=True)
    res = solve(model, exact_cfg)

    assert res.status == "OPTIMAL"
    assert abs(res.n_safe - n_oracle_safe) <= 1


# ---------------------------------------------------------------------------
# Test 7: Real instance smoke test (marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_instance_smoke_demand_100():
    """Smoke test on real instance: demand=100, seed_index=0, flood=60 m/min.

    Verifies dual bounds: n_safe_ceil <= best realized heuristic n_safe <= n_safe_floor + 1.
    """
    study_cfg = StudyAreaConfig(name="tollygunge", center_lat=22.4989, center_lon=88.3488, dist_m=1200)
    G = get_or_build_graph(study_cfg)

    h_cfg = HazardConfig(terrain_seed=7, spread_speed_m_per_min=60.0, seed=3000)
    hazard = FloodHazard(G, h_cfg)

    sc_cfg = ScenarioConfig(n_evacuees=100)
    exits, _ = build_exits(
        G=G,
        hazard=hazard,
        scenario_cfg=sc_cfg,
        center_lat=study_cfg.center_lat,
        center_lon=study_cfg.center_lon,
        horizon_min=h_cfg.horizon_min,
    )
    rng = np.random.default_rng(4000)
    agents = generate_agents(G, hazard, exits, sc_cfg, rng=rng)
    params = RoutingParams(vehicle_speed_scale=sc_cfg.vehicle_speed_scale, sf_floor=sc_cfg.sf_floor)

    # 1. Evaluate best realized heuristic safe count in simulator
    from src.experiment.ladder_runner import instantiate_router
    traffic_cfg = TrafficConfig(replan_interval_s=60.0)
    coord_cfg = CoordinatedConfig()
    heuristic_routers = ["static", "time_dependent", "selfish", "selfish_forecast", "coordinated_online"]
    best_heur_safe = -1
    for r_name in heuristic_routers:
        r_inst = instantiate_router(r_name, traffic_cfg, coord_cfg)
        pop_res = simulate_population(
            G=G,
            hazard=hazard,
            router=r_inst,
            agents=agents,
            exits=exits,
            params=params,
            traffic_cfg=traffic_cfg,
        )
        cnt = sum(1 for r in pop_res.agent_results if r.safe)
        if cnt > best_heur_safe:
            best_heur_safe = cnt

    # 2. Exact ceil model
    cfg_ceil = ExactConfig(dt_s=5, horizon_s=2400, big_m_factor=10, rounding="ceil")
    model_ceil = build_network(G, hazard, agents, exits, params, cfg_ceil, prune=True)
    res_ceil = solve(model_ceil, cfg_ceil)

    # 3. Exact floor model
    cfg_floor = ExactConfig(dt_s=5, horizon_s=2400, big_m_factor=10, rounding="floor")
    model_floor = build_network(G, hazard, agents, exits, params, cfg_floor, prune=True)
    res_floor = solve(model_floor, cfg_floor)

    n_ceil = res_ceil.n_safe
    n_floor = res_floor.n_safe

    # Assert dual bracket: n_safe_ceil <= best_heur_safe <= n_safe_floor + 1
    # Print both numbers if violated
    msg = f"Bracket violated: n_ceil={n_ceil}, best_heur={best_heur_safe}, n_floor={n_floor}"
    assert n_ceil <= best_heur_safe, msg
    assert best_heur_safe <= n_floor + 1, msg


# ---------------------------------------------------------------------------
# Test 8: Rounding bracket on a toy chain
# ---------------------------------------------------------------------------

def test_rounding_bracket_toy_chain():
    """Chain S->A->D, tau 7s per edge, dt = 5s, flow 60 veh/min, 1 agent at t=0, no hazard.

    Continuous travel time: 7 + 7 = 14s.
    In ceil mode:
      m1 = ceil(7/5) = 2 steps (10s)
      m2 = ceil(7/5) = 2 steps (10s)
      total ceil steps = 4 steps (20s).
    In floor mode:
      m1 = floor(7/5) = 1 step (5s)
      m2 = floor(7/5) = 1 step (5s)
      total floor steps = 2 steps (10s).
    Verify floor_total*dt <= 14 <= ceil_total*dt.
    """
    G = nx.MultiDiGraph()
    G.add_edge("S", "A", 0, free_flow_time=7.0, travel_time=7.0, flow_capacity=60.0, length=70.0, capacity=10)
    G.add_edge("A", "D", 0, free_flow_time=7.0, travel_time=7.0, flow_capacity=60.0, length=70.0, capacity=10)

    hazard = StubMultiAgentHazard()
    agents = [Evacuee(id=0, origin="S", depart_time_s=0.0)]
    params = RoutingParams(vehicle_speed_scale=1.0)

    cfg_ceil = ExactConfig(dt_s=5, horizon_s=100, big_m_factor=10, rounding="ceil")
    res_ceil = solve(build_network(G, hazard, agents, {"D"}, params, cfg_ceil, prune=True), cfg_ceil)

    cfg_floor = ExactConfig(dt_s=5, horizon_s=100, big_m_factor=10, rounding="floor")
    res_floor = solve(build_network(G, hazard, agents, {"D"}, params, cfg_floor, prune=True), cfg_floor)

    assert res_ceil.n_safe == 1
    assert res_floor.n_safe == 1

    ceil_s = res_ceil.total_travel_steps * cfg_ceil.dt_s
    floor_s = res_floor.total_travel_steps * cfg_floor.dt_s

    assert floor_s <= 14.0 <= ceil_s
    assert floor_s == 10.0
    assert ceil_s == 20.0


# ---------------------------------------------------------------------------
# Test 9: Flood bracket on toy chain
# ---------------------------------------------------------------------------

def test_flood_bracket_toy_chain():
    """Same chain S->A->D, tau 7s per edge, edge S-A closes at t = 8s.

    Agent departs at t=0.
    In continuous simulator:
      Clears edge S-A at t = 7s < 8s -> reaches A safely, enters A-D at 7s -> reaches D at 14s (SAFE).
    In ceil model:
      Step arrival = ceil(7/5) = 2 steps = 10s >= 8s -> INFEASIBLE (UNSAFE).
    In floor model:
      Step arrival = 0*5 + 7 = 7s < 8s -> FEASIBLE (SAFE).
    Assert: n_safe_ceil <= n_safe_simulator <= n_safe_floor.
    """
    G = nx.MultiDiGraph()
    G.add_edge("S", "A", 0, free_flow_time=7.0, travel_time=7.0, flow_capacity=60.0, length=70.0, capacity=10)
    G.add_edge("A", "D", 0, free_flow_time=7.0, travel_time=7.0, flow_capacity=60.0, length=70.0, capacity=10)

    hazard = StubMultiAgentHazard(
        edge_arrivals={("S", "A", 0): 8.0},
    )
    agents = [Evacuee(id=0, origin="S", depart_time_s=0.0)]
    params = RoutingParams(vehicle_speed_scale=1.0)

    # 1. Simulator outcome
    sim_res = simulate_agent(G, hazard, TimeDependentRouter(), agents[0], {"D"}, params)
    n_safe_sim = 1 if sim_res.safe else 0
    assert n_safe_sim == 1

    # 2. Ceil model outcome
    cfg_ceil = ExactConfig(dt_s=5, horizon_s=100, big_m_factor=10, rounding="ceil")
    res_ceil = solve(build_network(G, hazard, agents, {"D"}, params, cfg_ceil, prune=True), cfg_ceil)

    # 3. Floor model outcome
    cfg_floor = ExactConfig(dt_s=5, horizon_s=100, big_m_factor=10, rounding="floor")
    res_floor = solve(build_network(G, hazard, agents, {"D"}, params, cfg_floor, prune=True), cfg_floor)

    assert res_ceil.n_safe == 0
    assert res_floor.n_safe == 1
    assert res_ceil.n_safe <= n_safe_sim <= res_floor.n_safe


# ---------------------------------------------------------------------------
# Test 10: Zero-length arc in floor mode
# ---------------------------------------------------------------------------

def test_zero_length_arc_floor_mode():
    """An edge with tau 3s and dt 5s in floor mode has m = floor(3/5) = 0.

    Verifies that m = 0 does not crash, prunes correctly, and finds the optimal route.
    """
    G = nx.MultiDiGraph()
    # 0 -> 1 with tau = 3s (m=0 in floor mode)
    G.add_edge(0, 1, 0, free_flow_time=3.0, travel_time=3.0, flow_capacity=60.0, length=30.0, capacity=10)
    # 1 -> 2 with tau = 8s (m=1 in floor mode)
    G.add_edge(1, 2, 0, free_flow_time=8.0, travel_time=8.0, flow_capacity=60.0, length=80.0, capacity=10)

    hazard = StubMultiAgentHazard()
    agents = [Evacuee(id=0, origin=0, depart_time_s=0.0)]
    params = RoutingParams(vehicle_speed_scale=1.0)

    cfg_floor = ExactConfig(dt_s=5, horizon_s=50, big_m_factor=10, rounding="floor")
    model_floor = build_network(G, hazard, agents, {2}, params, cfg_floor, prune=True)
    res_floor = solve(model_floor, cfg_floor)

    assert res_floor.status == "OPTIMAL"
    assert res_floor.n_safe == 1
    # m(0->1) = 0, m(1->2) = 1 -> total steps = 1 (5s)
    assert res_floor.total_travel_steps == 1
