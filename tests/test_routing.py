"""Tests for single-agent evacuation routing and planners."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HazardConfig, ScenarioConfig, StudyAreaConfig, load_config
from src.graph.loader import get_or_build_graph
from src.hazard.base import HazardModel
from src.hazard.flood import FloodHazard
from src.routing.base import Evacuee, RoutingParams
from src.routing.planners import ReactiveRouter, StaticRouter, TimeDependentRouter
from src.scenario import build_exits, generate_agents
from src.sim.single_agent import simulate_agent


# ---------------------------------------------------------------------------
# Test stub: HazardModel implementation for unit tests
# ---------------------------------------------------------------------------

class StubHazard(HazardModel):
    """Stub hazard model for isolated unit tests with explicit arrival times."""

    def __init__(
        self,
        edge_arrivals: dict[tuple[Any, Any, int], float] | None = None,
        node_arrivals: dict[Any, float] | None = None,
    ) -> None:
        self._edge_arrivals: dict[tuple[Any, Any, int], float] = edge_arrivals or {}
        self._node_arrivals: dict[Any, float] = node_arrivals or {}

    def node_arrival_time(self, n: Any) -> float:
        return self._node_arrivals.get(n, float("inf"))

    def edge_arrival_time(self, u: Any, v: Any, k: int) -> float:
        return self._edge_arrivals.get((u, v, k), float("inf"))

    def is_edge_closed(self, u: Any, v: Any, k: int, t: float) -> bool:
        arr = self.edge_arrival_time(u, v, k)
        return bool(np.isfinite(arr) and t >= arr)

    def edge_speed_factor(self, u: Any, v: Any, k: int, t: float) -> float:
        arr = self.edge_arrival_time(u, v, k)
        if not np.isfinite(arr):
            return 1.0
        return 0.0 if t >= arr else 1.0

    def is_node_flooded(self, n: Any, t: float) -> bool:
        arr = self.node_arrival_time(n)
        return bool(np.isfinite(arr) and t >= arr)

    def closed_edges(self, t: float) -> set[tuple[Any, Any, int]]:
        return {
            edge for edge, arr in self._edge_arrivals.items()
            if np.isfinite(arr) and t >= arr
        }

    def closure_schedule(self) -> list[tuple[float, tuple[Any, Any, int]]]:
        items = [
            (arr, edge) for edge, arr in self._edge_arrivals.items()
            if np.isfinite(arr)
        ]
        items.sort()
        return items


def make_tiny_graph() -> nx.MultiDiGraph:
    """Build a tiny test road network:

        (40s)       (30s)
      A <======> B <======> D [EXIT]
      |                     ^
      | (50s)               | (50s)
      v                     |
      C <===================+

    All edges bidirectional, length_m = 100.
    """
    G = nx.MultiDiGraph()
    edges = [
        ("A", "B", 40.0),
        ("B", "D", 30.0),
        ("A", "C", 50.0),
        ("C", "D", 50.0),
    ]
    for u, v, fft in edges:
        # Forward
        G.add_edge(u, v, key=0, free_flow_time=fft, travel_time=fft, length=100.0)
        # Reverse
        G.add_edge(v, u, key=0, free_flow_time=fft, travel_time=fft, length=100.0)
    return G


# ---------------------------------------------------------------------------
# Tests 1-5: Stub network unit tests
# ---------------------------------------------------------------------------

def test_1_edge_closes_while_in_transit():
    """Test 1: B-D arrival = 50s.

    Static plans A-B-D (cost 70 < 100) and is TRAPPED en-route (enters B-D at
    40, would exit at 70 >= 50).
    Reactive reaches B at 40; closure is not yet observable at t=40 (arrival 50),
    so it enters B-D and is also TRAPPED.
    TimeDependent plans A-C-D from departure, SAFE, evac_time 100.
    """
    G = make_tiny_graph()
    hazard = StubHazard(
        edge_arrivals={
            ("B", "D", 0): 50.0,
            ("D", "B", 0): 50.0,
        }
    )
    evacuee = Evacuee(id=1, origin="A", depart_time_s=0.0)
    exits = ["D"]
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)

    res_static = simulate_agent(G, hazard, StaticRouter(), evacuee, exits, params)
    assert not res_static.safe
    assert res_static.trapped_reason == "edge_closed_en_route"

    res_reactive = simulate_agent(G, hazard, ReactiveRouter(), evacuee, exits, params)
    assert not res_reactive.safe
    assert res_reactive.trapped_reason == "edge_closed_en_route"

    res_td = simulate_agent(G, hazard, TimeDependentRouter(), evacuee, exits, params)
    assert res_td.safe
    assert res_td.trapped_reason is None
    assert res_td.evac_time_s == pytest.approx(100.0)
    assert res_td.path_nodes == ["A", "C", "D"]


def test_2_reactive_replan_on_observed_closure():
    """Test 2: B-D arrival = 20s.

    Static plans A-B-D and is TRAPPED.
    Reactive reaches B at 40, sees B-D closed (since 20 <= 40), replans
    B-A-C-D, reaches exit D at t=180, SAFE, n_replans == 1.
    TimeDependent plans A-C-D, SAFE, evac_time 100.
    """
    G = make_tiny_graph()
    hazard = StubHazard(
        edge_arrivals={
            ("B", "D", 0): 20.0,
            ("D", "B", 0): 20.0,
        }
    )
    evacuee = Evacuee(id=2, origin="A", depart_time_s=0.0)
    exits = ["D"]
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)

    res_static = simulate_agent(G, hazard, StaticRouter(), evacuee, exits, params)
    assert not res_static.safe
    assert res_static.trapped_reason == "edge_closed_en_route"

    res_reactive = simulate_agent(G, hazard, ReactiveRouter(), evacuee, exits, params)
    assert res_reactive.safe
    assert res_reactive.trapped_reason is None
    assert res_reactive.evac_time_s == pytest.approx(180.0)
    assert res_reactive.n_replans == 1
    assert res_reactive.path_nodes == ["A", "B", "A", "C", "D"]

    res_td = simulate_agent(G, hazard, TimeDependentRouter(), evacuee, exits, params)
    assert res_td.safe
    assert res_td.evac_time_s == pytest.approx(100.0)
    assert res_td.path_nodes == ["A", "C", "D"]


def test_3_no_closures_identical_routes():
    """Test 3: no closures anywhere (all arrival inf).

    All three routers yield the identical fastest route A-B-D and evac_time 70.
    """
    G = make_tiny_graph()
    hazard = StubHazard()  # All arrivals inf
    evacuee = Evacuee(id=3, origin="A", depart_time_s=0.0)
    exits = ["D"]
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)

    for router_cls in (StaticRouter, ReactiveRouter, TimeDependentRouter):
        res = simulate_agent(G, hazard, router_cls(), evacuee, exits, params)
        assert res.safe
        assert res.evac_time_s == pytest.approx(70.0)
        assert res.path_nodes == ["A", "B", "D"]
        assert res.n_replans == 0


def test_4_parallel_edge_selection():
    """Test 4: parallel edges between A and B with key=1 faster (30s vs 40s).

    Planners must select key=1 and route edges must carry key=1.
    """
    G = make_tiny_graph()
    # Add parallel edge key=1 with free_flow_time = 30s
    G.add_edge("A", "B", key=1, free_flow_time=30.0, travel_time=30.0, length=100.0)
    G.add_edge("B", "A", key=1, free_flow_time=30.0, travel_time=30.0, length=100.0)

    hazard = StubHazard()
    evacuee = Evacuee(id=4, origin="A", depart_time_s=0.0)
    exits = ["D"]
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)

    for router_cls in (StaticRouter, ReactiveRouter, TimeDependentRouter):
        router = router_cls()
        plan = router.plan(G, hazard, evacuee, 0.0, "A", exits, params)
        assert plan is not None
        assert plan.edges[0] == ("A", "B", 1)
        res = simulate_agent(G, hazard, router, evacuee, exits, params)
        assert res.safe
        assert res.evac_time_s == pytest.approx(60.0)  # 30s + 30s


def test_5_source_is_exit():
    """Test 5: source is an exit -> empty route, SAFE, evac_time 0."""
    G = make_tiny_graph()
    hazard = StubHazard()
    evacuee = Evacuee(id=5, origin="D", depart_time_s=15.0)
    exits = ["D"]
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)

    for router_cls in (StaticRouter, ReactiveRouter, TimeDependentRouter):
        router = router_cls()
        plan = router.plan(G, hazard, evacuee, 15.0, "D", exits, params)
        assert plan is not None
        assert plan.edges == []
        assert plan.nodes == ["D"]

        res = simulate_agent(G, hazard, router, evacuee, exits, params)
        assert res.safe
        assert res.evac_time_s == pytest.approx(0.0)
        assert res.finish_time_s == pytest.approx(15.0)
        assert res.path_nodes == ["D"]


# ---------------------------------------------------------------------------
# Test 6: Full Kolkata road network integration test
# ---------------------------------------------------------------------------

def test_6_time_dependent_correctness_on_kolkata():
    """Test 6: Real graph + FloodHazard with 100 sampled agents.

    Properties verified:
    1. Time-dependent router NEVER ends trapped with "edge_closed_en_route".
    2. Whenever both static and time-dependent are SAFE,
       td evac_time <= static evac_time + 1e-6.
    """
    graph_cfg = load_config(PROJECT_ROOT / "configs" / "study_area.yaml", StudyAreaConfig)
    hazard_cfg = load_config(PROJECT_ROOT / "configs" / "hazard.yaml", HazardConfig)
    scenario_cfg = load_config(PROJECT_ROOT / "configs" / "scenario.yaml", ScenarioConfig)

    G = get_or_build_graph(graph_cfg)
    hazard = FloodHazard(G, hazard_cfg)
    exits, _ = build_exits(
        G,
        hazard,
        scenario_cfg,
        center_lat=graph_cfg.center_lat,
        center_lon=graph_cfg.center_lon,
        horizon_min=hazard_cfg.horizon_min,
    )

    rng = np.random.default_rng(scenario_cfg.scenario_seed)
    # Test on 100 agents for rapid test execution
    sub_cfg = ScenarioConfig(
        scenario_seed=scenario_cfg.scenario_seed,
        n_exits=scenario_cfg.n_exits,
        exit_min_dist_from_center_m=scenario_cfg.exit_min_dist_from_center_m,
        n_evacuees=100,
        depart_window_min=scenario_cfg.depart_window_min,
        vehicle_speed_scale=scenario_cfg.vehicle_speed_scale,
        sf_floor=scenario_cfg.sf_floor,
    )
    agents = generate_agents(G, hazard, exits, sub_cfg, rng)
    params = RoutingParams(
        vehicle_speed_scale=scenario_cfg.vehicle_speed_scale,
        sf_floor=scenario_cfg.sf_floor,
    )

    static_router = StaticRouter()
    td_router = TimeDependentRouter()

    for agent in agents:
        res_td = simulate_agent(G, hazard, td_router, agent, exits, params)
        # Property 1: Perfect forecast means agent never clears edge too late
        assert res_td.trapped_reason != "edge_closed_en_route", (
            f"Agent {agent.id} unexpectedly trapped en-route under time-dependent routing!"
        )

        res_static = simulate_agent(G, hazard, static_router, agent, exits, params)
        # Property 2: If both are safe, time-dependent is at least as fast
        if res_static.safe and res_td.safe:
            assert res_td.evac_time_s is not None and res_static.evac_time_s is not None
            assert res_td.evac_time_s <= (res_static.evac_time_s + 1e-6), (
                f"Agent {agent.id}: TD evac time ({res_td.evac_time_s:.2f}s) > "
                f"Static evac time ({res_static.evac_time_s:.2f}s)"
            )

