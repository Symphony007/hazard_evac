"""Unit tests for flow decomposition and ExactPlanRouter replay in the simulator."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.config import (
    ExactConfig,
    HazardConfig,
    ScenarioConfig,
    StudyAreaConfig,
    TrafficConfig,
    load_config,
)
from src.exact.decompose import AgentPlan, decompose_flow
from src.exact.time_expanded import ExactModel, build_network, solve
from src.graph.loader import get_or_build_graph
from src.hazard.base import HazardModel
from src.hazard.flood import FloodHazard
from src.routing.base import Evacuee, RoutingParams, make_routing_params
from src.routing.exact_plan import ExactPlanRouter
from src.scenario import build_exits, generate_agents
from src.sim.multi_agent import simulate_population
from src.sim.single_agent import simulate_agent
from tests.test_multiagent import StubMultiAgentHazard

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Test 1: Chain S -> A -> D, 5 agents at t=0
# ---------------------------------------------------------------------------

def test_chain_decompose_and_replay():
    """Chain S=0 -> A=1 -> D=2, tau 10s, flow 6 veh/min (headway 10s), dt 10s, 5 agents at t=0.

    Decomposed plans must each traverse [S-A, A-D].
    Replaying exact_plan in the simulator must produce evac times 20, 30, 40, 50, 60s.
    """
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)
    G.add_edge(1, 2, 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)

    hazard = StubMultiAgentHazard()
    agents = [Evacuee(id=i, origin=0, depart_time_s=0.0) for i in range(5)]
    params = RoutingParams(vehicle_speed_scale=1.0)
    cfg = ExactConfig(dt_s=10, horizon_s=200, big_m_factor=10, rounding="ceil")

    model = build_network(G, hazard, agents, exits={2}, params=params, exact_cfg=cfg, prune=True)
    res = solve(model, exact_cfg=cfg)

    assert res.status == "OPTIMAL"
    assert res.n_safe == 5

    plans = decompose_flow(model, res, agents)
    assert len(plans) == 5

    for i in range(5):
        p = plans[i]
        assert p is not None
        assert p.edges == [(0, 1, 0), (1, 2, 0)]
        assert p.nodes == [0, 1, 2]

    # Replay in point-queue simulator
    router = ExactPlanRouter(plans=plans)
    pop_res = simulate_population(G, hazard, router, agents, exits={2}, params=params)

    assert all(r.safe for r in pop_res.agent_results)
    evac_times = [r.evac_time_s for r in pop_res.agent_results]
    assert evac_times == pytest.approx([20.0, 30.0, 40.0, 50.0, 60.0])


# ---------------------------------------------------------------------------
# Test 2: Same chain with edge A -> D closing at 45s
# ---------------------------------------------------------------------------

def test_chain_closing_edge_decompose_and_replay():
    """Same chain with A->D closing at t = 45s: 3 safe plans, 2 agents with None.

    Replay in simulator: 3 safe (evac times 20, 30, 40s), 2 trapped (no_route).
    """
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)
    G.add_edge(1, 2, 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)

    hazard = StubMultiAgentHazard(edge_arrivals={(1, 2, 0): 45.0})
    agents = [Evacuee(id=i, origin=0, depart_time_s=0.0) for i in range(5)]
    params = RoutingParams(vehicle_speed_scale=1.0)
    cfg = ExactConfig(dt_s=10, horizon_s=200, big_m_factor=10, rounding="ceil")

    model = build_network(G, hazard, agents, exits={2}, params=params, exact_cfg=cfg, prune=True)
    res = solve(model, exact_cfg=cfg)

    assert res.status == "OPTIMAL"
    assert res.n_safe == 3

    plans = decompose_flow(model, res, agents)

    # First 3 agents get safe plans, last 2 agents get None
    for i in range(3):
        assert plans[i] is not None
        assert plans[i].edges == [(0, 1, 0), (1, 2, 0)]
    assert plans[3] is None
    assert plans[4] is None

    # Replay in point-queue simulator
    router = ExactPlanRouter(plans=plans)
    pop_res = simulate_population(G, hazard, router, agents, exits={2}, params=params)

    assert [r.safe for r in pop_res.agent_results] == [True, True, True, False, False]
    assert [r.evac_time_s for r in pop_res.agent_results[:3]] == pytest.approx([20.0, 30.0, 40.0])
    assert pop_res.agent_results[3].trapped_reason == "no_route"
    assert pop_res.agent_results[4].trapped_reason == "no_route"


# ---------------------------------------------------------------------------
# Test 3: Parallel routes toy (S->F->D, S->G->D) with 10 agents
# ---------------------------------------------------------------------------

def test_parallel_routes_decompose_and_replay():
    """Diamond graph with two parallel routes (S=0 -> F=1 -> D=3 and S=0 -> G=2 -> D=3).

    Both routes must be used, and replayed safe count >= n_safe_ceil_model - 1.
    """
    G = nx.MultiDiGraph()
    # Route F
    G.add_edge(0, 1, 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)
    G.add_edge(1, 3, 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)
    # Route G
    G.add_edge(0, 2, 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)
    G.add_edge(2, 3, 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)

    hazard = StubMultiAgentHazard()
    agents = [Evacuee(id=i, origin=0, depart_time_s=0.0) for i in range(10)]
    params = RoutingParams(vehicle_speed_scale=1.0)
    cfg = ExactConfig(dt_s=10, horizon_s=200, big_m_factor=10, rounding="ceil")

    model = build_network(G, hazard, agents, exits={3}, params=params, exact_cfg=cfg, prune=True)
    res = solve(model, exact_cfg=cfg)

    assert res.status == "OPTIMAL"
    assert res.n_safe == 10

    plans = decompose_flow(model, res, agents)

    # Check that both branches (F=1 and G=2) are utilized
    used_f = any(1 in p.nodes for p in plans.values() if p is not None)
    used_g = any(2 in p.nodes for p in plans.values() if p is not None)
    assert used_f, "Route F should be utilized"
    assert used_g, "Route G should be utilized"

    # Replay in simulator
    router = ExactPlanRouter(plans=plans)
    pop_res = simulate_population(G, hazard, router, agents, exits={3}, params=params)

    realized_safe = sum(1 for r in pop_res.agent_results if r.safe)
    assert realized_safe >= res.n_safe - 1


# ---------------------------------------------------------------------------
# Test 4: Parallel edge keys preservation
# ---------------------------------------------------------------------------

def test_parallel_edge_keys():
    """Verify that parallel edges with different keys carry the exact key chosen by the flow solver."""
    G = nx.MultiDiGraph()
    # S=0 -> A=1 has two parallel edges: key 0 (fast, 10s) and key 1 (slow, 30s)
    G.add_edge(0, 1, 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)
    G.add_edge(0, 1, 1, free_flow_time=30.0, travel_time=30.0, flow_capacity=6.0, length=300.0)
    G.add_edge(1, 2, 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)

    hazard = StubMultiAgentHazard()
    # 1 agent departing at t=0
    agents = [Evacuee(id=0, origin=0, depart_time_s=0.0)]
    params = RoutingParams(vehicle_speed_scale=1.0)
    cfg = ExactConfig(dt_s=10, horizon_s=100, big_m_factor=10, rounding="ceil")

    model = build_network(G, hazard, agents, exits={2}, params=params, exact_cfg=cfg, prune=True)
    res = solve(model, exact_cfg=cfg)

    assert res.status == "OPTIMAL"
    assert res.n_safe == 1

    plans = decompose_flow(model, res, agents)
    p = plans[0]
    assert p is not None
    # Solver strictly prefers faster key 0
    assert p.edges[0] == (0, 1, 0)
    assert p.edges[1] == (1, 2, 0)


# ---------------------------------------------------------------------------
# Test 5: Flow conservation and error handling on corrupted flow
# ---------------------------------------------------------------------------

def test_conservation_and_corruption_error():
    """Verify that validation checks pass on valid flow and raise ValueError on corrupted flow."""
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)
    G.add_edge(1, 2, 0, free_flow_time=10.0, travel_time=10.0, flow_capacity=6.0, length=100.0)

    hazard = StubMultiAgentHazard()
    agents = [Evacuee(id=i, origin=0, depart_time_s=float(i * 10)) for i in range(3)]
    params = RoutingParams(vehicle_speed_scale=1.0)
    cfg = ExactConfig(dt_s=10, horizon_s=200, big_m_factor=10, rounding="ceil")

    model = build_network(G, hazard, agents, exits={2}, params=params, exact_cfg=cfg, prune=True)
    res = solve(model, exact_cfg=cfg)

    # Valid flow succeeds
    plans = decompose_flow(model, res, agents)
    assert len(plans) == 3

    # Corrupt 1: result with invalid status
    bad_res_status = replace(res, status="INFEASIBLE")
    with pytest.raises(ValueError, match="non-optimal"):
        decompose_flow(model, bad_res_status, agents)

    # Corrupt 2: arc flows modified arbitrarily (breaks flow conservation)
    bad_flows = list(res.arc_flows)
    bad_flows[0] += 5  # Add spurious flow
    bad_res_flow = replace(res, arc_flows=bad_flows)
    with pytest.raises(ValueError):
        decompose_flow(model, bad_res_flow, agents)


# ---------------------------------------------------------------------------
# Test 6: Slow test on real instance
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_instance_replay_bracket():
    """Demand 100, seed 0 on real graph: replayed safe count satisfies n_safe_ceil - 1 <= realized <= ceiling_n."""
    graph_cfg = load_config(PROJECT_ROOT / "configs" / "study_area.yaml", StudyAreaConfig)
    hazard_cfg = load_config(PROJECT_ROOT / "configs" / "hazard.yaml", HazardConfig)
    scenario_cfg = load_config(PROJECT_ROOT / "configs" / "scenario.yaml", ScenarioConfig)
    traffic_cfg = load_config(PROJECT_ROOT / "configs" / "traffic.yaml", TrafficConfig)

    G = get_or_build_graph(graph_cfg)
    h_cfg = replace(
        hazard_cfg,
        spread_speed_m_per_min=60.0,
        seed=3000,
        terrain_seed=int(hazard_cfg.terrain_seed),
    )
    hazard = FloodHazard(G, h_cfg)
    exits, _ = build_exits(
        G=G,
        hazard=hazard,
        scenario_cfg=scenario_cfg,
        center_lat=graph_cfg.center_lat,
        center_lon=graph_cfg.center_lon,
        horizon_min=h_cfg.horizon_min,
    )

    rng = np.random.default_rng(4000)
    agents = generate_agents(
        G=G,
        hazard=hazard,
        exits=exits,
        scenario_cfg=replace(scenario_cfg, n_evacuees=100),
        rng=rng,
    )

    params = make_routing_params(scenario_cfg)
    exact_cfg = ExactConfig(dt_s=5, horizon_s=3600, big_m_factor=10, rounding="ceil")

    # 1. Oracle ceiling count
    ceiling_safe = 0
    from src.routing.planners import TimeDependentRouter
    td_router = TimeDependentRouter()
    for a in agents:
        s_res = simulate_agent(G, hazard, td_router, a, exits, params)
        if s_res.safe:
            ceiling_safe += 1

    # 2. Exact ceil model
    model = build_network(G, hazard, agents, exits, params, exact_cfg, prune=True)
    res = solve(model, exact_cfg=exact_cfg)

    # 3. Decompose
    plans = decompose_flow(model, res, agents)

    # 4. Replay in simulator
    router = ExactPlanRouter(plans=plans)
    pop_res = simulate_population(G, hazard, router, agents, exits, params, traffic_cfg)
    realized_safe = sum(1 for r in pop_res.agent_results if r.safe)

    print(f"\nReal instance (demand 100, seed 0):")
    print(f"  Exact ceil model n_safe: {res.n_safe}")
    print(f"  Replayed realized n_safe: {realized_safe}")
    print(f"  Oracle ceiling n_safe:    {ceiling_safe}")

    assert res.n_safe - 1 <= realized_safe <= ceiling_safe
