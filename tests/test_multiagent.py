"""Unit tests for point-queue multi-agent evacuation simulation and selfish routing."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HazardConfig, ScenarioConfig, StudyAreaConfig, TrafficConfig, load_config
from src.graph.loader import get_or_build_graph
from src.hazard.base import HazardModel
from src.hazard.flood import FloodHazard
from src.routing.base import Evacuee, RoutingParams
from src.routing.planners import ReactiveRouter, SelfishRouter, StaticRouter, TimeDependentRouter
from src.scenario import build_exits, generate_agents
from src.sim.multi_agent import simulate_population
from src.sim.single_agent import simulate_agent


class StubMultiAgentHazard(HazardModel):
    """Stub hazard simulator with custom edge and node arrival times."""

    def __init__(
        self,
        edge_arrivals: dict[tuple[int, int, int], float] | None = None,
        node_arrivals: dict[int, float] | None = None,
    ) -> None:
        self._edge_arrivals = edge_arrivals or {}
        self._node_arrivals = node_arrivals or {}
        self.sources = []

    def node_arrival_time(self, node: int) -> float:
        return self._node_arrivals.get(node, float("inf"))

    def edge_arrival_time(self, u: int, v: int, k: int) -> float:
        return self._edge_arrivals.get((u, v, k), float("inf"))

    def is_edge_closed(self, u: int, v: int, k: int, t: float) -> bool:
        return t >= self.edge_arrival_time(u, v, k)

    def edge_speed_factor(self, u: int, v: int, k: int, t: float) -> float:
        return 0.0 if self.is_edge_closed(u, v, k, t) else 1.0

    def is_node_flooded(self, node: int, t: float) -> bool:
        return t >= self.node_arrival_time(node)

    def closed_edges(self, t: float) -> list[tuple[int, int, int]]:
        return [e for e, arr in self._edge_arrivals.items() if t >= arr]

    def closure_schedule(self) -> list[tuple[float, tuple[int, int, int]]]:
        return sorted([(arr, e) for e, arr in self._edge_arrivals.items()])


# ---------------------------------------------------------------------------
# Test 1: Queue Arithmetic (Toy 1)
# ---------------------------------------------------------------------------

def test_toy1_queue_arithmetic():
    """Verify queue wait and exit times on S -> A -> D linear graph with 5 agents."""
    G = nx.MultiDiGraph()
    # S=0, A=1, D=2
    # free_flow_time = 10s, flow_capacity = 6 veh/min -> h = 10s
    G.add_node(0, y=0.0, x=0.0)
    G.add_node(1, y=0.0, x=1.0)
    G.add_node(2, y=0.0, x=2.0)
    G.add_edge(0, 1, key=0, free_flow_time=10.0, flow_capacity=6.0, capacity=10, length=100.0)
    G.add_edge(1, 2, key=0, free_flow_time=10.0, flow_capacity=6.0, capacity=10, length=100.0)

    hazard = StubMultiAgentHazard()
    router = StaticRouter()
    agents = [Evacuee(id=i, origin=0, depart_time_s=0.0) for i in range(1, 6)]
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)

    pop_res = simulate_population(G, hazard, router, agents, exits={2}, params=params)

    # Expected evac times: 20, 30, 40, 50, 60 seconds
    expected_evac = [20.0, 30.0, 40.0, 50.0, 60.0]
    actual_evac = [r.evac_time_s for r in pop_res.agent_results]
    assert actual_evac == pytest.approx(expected_evac)

    # Queue wait on S-A for agent i is 10 * (i - 1)
    # Agent 1: 0s, Agent 2: 10s, Agent 3: 20s, Agent 4: 30s, Agent 5: 40s
    expected_waits = [0.0, 10.0, 20.0, 30.0, 40.0]
    actual_waits = [r.queue_delay_s for r in pop_res.agent_results]
    assert actual_waits == pytest.approx(expected_waits)

    edge_sa = pop_res.edge_stats[
        (pop_res.edge_stats["u"] == 0) & (pop_res.edge_stats["v"] == 1)
    ].iloc[0]
    assert edge_sa["max_queue_wait_s"] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Test 2: Trapping in Queue (Toy 2)
# ---------------------------------------------------------------------------

def test_toy2_trapping_in_queue():
    """Verify that vehicles trapped while waiting in queue are marked edge_closed_en_route."""
    G = nx.MultiDiGraph()
    G.add_node(0, y=0.0, x=0.0)
    G.add_node(1, y=0.0, x=1.0)
    G.add_node(2, y=0.0, x=2.0)
    G.add_edge(0, 1, key=0, free_flow_time=10.0, flow_capacity=6.0, capacity=10, length=100.0)
    G.add_edge(1, 2, key=0, free_flow_time=10.0, flow_capacity=6.0, capacity=10, length=100.0)

    # Edge A-D (1->2) floods at 45s
    hazard = StubMultiAgentHazard(edge_arrivals={(1, 2, 0): 45.0})
    router = StaticRouter()
    agents = [Evacuee(id=i, origin=0, depart_time_s=0.0) for i in range(1, 6)]
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)

    pop_res = simulate_population(G, hazard, router, agents, exits={2}, params=params)

    # Agents 1-3 safe (reach exit at 20, 30, 40 < 45)
    for i in range(3):
        assert pop_res.agent_results[i].safe is True
    # Agents 4 and 5 trapped edge_closed_en_route
    assert pop_res.agent_results[3].safe is False
    assert pop_res.agent_results[3].trapped_reason == "edge_closed_en_route"
    assert pop_res.agent_results[4].safe is False
    assert pop_res.agent_results[4].trapped_reason == "edge_closed_en_route"


# ---------------------------------------------------------------------------
# Test 3: Selfish Load Balancing (Toy 3)
# ---------------------------------------------------------------------------

def test_toy3_selfish_load_balancing():
    """Verify that SelfishRouter diverts agents to longer route under congestion."""
    G = nx.MultiDiGraph()
    # Route A: 0 -> 1 -> 3 (tau = 10 + 10 = 20)
    # Route B: 0 -> 2 -> 3 (tau = 16 + 16 = 32)
    # flow_capacity = 6 veh/min on all edges (h = 10s)
    for n in range(4):
        G.add_node(n, y=0.0, x=float(n))
    G.add_edge(0, 1, key=0, free_flow_time=10.0, flow_capacity=6.0, capacity=10, length=100.0)
    G.add_edge(1, 3, key=0, free_flow_time=10.0, flow_capacity=6.0, capacity=10, length=100.0)
    G.add_edge(0, 2, key=0, free_flow_time=16.0, flow_capacity=6.0, capacity=10, length=160.0)
    G.add_edge(2, 3, key=0, free_flow_time=16.0, flow_capacity=6.0, capacity=10, length=160.0)

    hazard = StubMultiAgentHazard()
    agents = [Evacuee(id=i, origin=0, depart_time_s=0.0) for i in range(1, 7)]
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)

    # 1. SelfishRouter: agents 1 and 2 use A, agent 3 uses B
    selfish = SelfishRouter(replan_interval_s=60.0)
    res_selfish = simulate_population(G, hazard, selfish, agents, exits={3}, params=params)

    nodes_a1 = res_selfish.agent_results[0].path_nodes
    nodes_a2 = res_selfish.agent_results[1].path_nodes
    nodes_a3 = res_selfish.agent_results[2].path_nodes

    # Agent 1 uses A (0 -> 1 -> 3)
    assert 1 in nodes_a1
    # Agent 2 uses A (0 -> 1 -> 3)
    assert 1 in nodes_a2
    # Agent 3 diverts to B (0 -> 2 -> 3)
    assert 2 in nodes_a3

    # At least one agent uses route B
    b_users = sum(1 for r in res_selfish.agent_results if 2 in r.path_nodes)
    assert b_users >= 1

    # 2. TimeDependentRouter: congestion-blind, NO agent uses B
    td = TimeDependentRouter()
    res_td = simulate_population(G, hazard, td, agents, exits={3}, params=params)
    td_b_users = sum(1 for r in res_td.agent_results if 2 in r.path_nodes)
    assert td_b_users == 0


# ---------------------------------------------------------------------------
# Test 4: Determinism
# ---------------------------------------------------------------------------

def test_multiagent_determinism():
    """Verify that two simulation runs produce identical outcomes."""
    G = nx.MultiDiGraph()
    for n in range(4):
        G.add_node(n, y=0.0, x=float(n))
    G.add_edge(0, 1, key=0, free_flow_time=10.0, flow_capacity=6.0, capacity=10, length=100.0)
    G.add_edge(1, 3, key=0, free_flow_time=10.0, flow_capacity=6.0, capacity=10, length=100.0)
    G.add_edge(0, 2, key=0, free_flow_time=16.0, flow_capacity=6.0, capacity=10, length=160.0)
    G.add_edge(2, 3, key=0, free_flow_time=16.0, flow_capacity=6.0, capacity=10, length=160.0)

    hazard = StubMultiAgentHazard()
    agents = [Evacuee(id=i, origin=0, depart_time_s=float(i * 2)) for i in range(1, 10)]
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)
    router = SelfishRouter()

    res1 = simulate_population(G, hazard, router, agents, exits={3}, params=params)
    res2 = simulate_population(G, hazard, router, agents, exits={3}, params=params)

    for r1, r2 in zip(res1.agent_results, res2.agent_results):
        assert r1.safe == r2.safe
        assert r1.evac_time_s == pytest.approx(r2.evac_time_s)
        assert r1.queue_delay_s == pytest.approx(r2.queue_delay_s)
        assert r1.path_nodes == r2.path_nodes


# ---------------------------------------------------------------------------
# Test 5: Infinite Capacity Regression vs simulate_agent
# ---------------------------------------------------------------------------

def test_infinite_capacity_equals_single_agent():
    """When flow_capacity is infinite (1e9), simulate_population matches simulate_agent."""
    graph_cfg = load_config(PROJECT_ROOT / "configs" / "study_area.yaml", StudyAreaConfig)
    hazard_cfg = load_config(PROJECT_ROOT / "configs" / "hazard.yaml", HazardConfig)
    scenario_cfg = load_config(PROJECT_ROOT / "configs" / "scenario.yaml", ScenarioConfig)

    G = get_or_build_graph(graph_cfg)
    hazard = FloodHazard(G, hazard_cfg)
    exits, _ = build_exits(G, hazard, scenario_cfg, center_lat=graph_cfg.center_lat, center_lon=graph_cfg.center_lon)

    rng = np.random.default_rng(42)
    agents = generate_agents(G, hazard, exits, replace(scenario_cfg, n_evacuees=50), rng=rng)
    params = RoutingParams(vehicle_speed_scale=scenario_cfg.vehicle_speed_scale, sf_floor=scenario_cfg.sf_floor)

    # Create graph with infinite flow capacity
    G_inf = G.copy()
    for u, v, k in G_inf.edges(keys=True):
        G_inf.edges[u, v, k]["flow_capacity"] = 1e9

    for router_cls in (StaticRouter, ReactiveRouter, TimeDependentRouter):
        router = router_cls()
        pop_res = simulate_population(G_inf, hazard, router, agents, exits, params)
        single_res = [simulate_agent(G, hazard, router, a, exits, params) for a in agents]

        for p_r, s_r in zip(pop_res.agent_results, single_res):
            assert p_r.safe == s_r.safe
            if p_r.safe:
                assert p_r.evac_time_s == pytest.approx(s_r.evac_time_s, abs=1e-6)
            else:
                assert p_r.trapped_reason == s_r.trapped_reason


# ---------------------------------------------------------------------------
# Test 6: Single Agent (N=1) Equals simulate_agent Under Real Capacities
# ---------------------------------------------------------------------------

def test_single_agent_n1_equals_simulate_agent():
    """When N=1, an evacuee never experiences queuing and matches simulate_agent."""
    graph_cfg = load_config(PROJECT_ROOT / "configs" / "study_area.yaml", StudyAreaConfig)
    hazard_cfg = load_config(PROJECT_ROOT / "configs" / "hazard.yaml", HazardConfig)
    scenario_cfg = load_config(PROJECT_ROOT / "configs" / "scenario.yaml", ScenarioConfig)

    G = get_or_build_graph(graph_cfg)
    hazard = FloodHazard(G, hazard_cfg)
    exits, _ = build_exits(G, hazard, scenario_cfg, center_lat=graph_cfg.center_lat, center_lon=graph_cfg.center_lon)

    rng = np.random.default_rng(99)
    agents = generate_agents(G, hazard, exits, replace(scenario_cfg, n_evacuees=1), rng=rng)
    params = RoutingParams(vehicle_speed_scale=scenario_cfg.vehicle_speed_scale, sf_floor=scenario_cfg.sf_floor)

    for router_cls in (StaticRouter, ReactiveRouter, TimeDependentRouter):
        router = router_cls()
        pop_res = simulate_population(G, hazard, router, agents, exits, params)
        single_res = simulate_agent(G, hazard, router, agents[0], exits, params)

        p_r = pop_res.agent_results[0]
        assert p_r.safe == single_res.safe
        if p_r.safe:
            assert p_r.evac_time_s == pytest.approx(single_res.evac_time_s, abs=1e-6)
        else:
            assert p_r.trapped_reason == single_res.trapped_reason
