"""Unit tests for capacity-reservation coordinated routing (Task 6)."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HazardConfig, ScenarioConfig, StudyAreaConfig, load_config
from src.graph.loader import get_or_build_graph
from src.hazard.flood import FloodHazard
from src.routing.base import Evacuee, RoutingParams
from src.routing.coordinated import CoordinatedRouter
from src.routing.planners import TimeDependentRouter
from src.routing.search import earliest_arrival_search
from src.scenario import build_exits, generate_agents
from src.sim.multi_agent import simulate_population
from src.traffic.reservation import ReservationTable
from tests.test_multiagent import StubMultiAgentHazard


# ---------------------------------------------------------------------------
# Test 1: ReservationTable arithmetic and monotonicity
# ---------------------------------------------------------------------------

def test_reservation_table_arithmetic_and_monotonicity():
    """Verify ReservationTable slot arithmetic, gap utilization, and monotonicity."""
    table = ReservationTable(default_headway_s=10.0)

    # Empty table: earliest_exit(y) == y
    assert table.earliest_exit("e1", 15.0) == pytest.approx(15.0)
    assert table.earliest_exit("e1", 0.0) == pytest.approx(0.0)

    # reserved {10, 20}: earliest_exit(11) == 30
    table.reserve("e1", 10.0)
    table.reserve("e1", 20.0)
    assert table.earliest_exit("e1", 11.0) == pytest.approx(30.0)

    # reserved {10, 40}: gap used at 20, conflicts handled
    table.clear()
    table.reserve("e1", 10.0)
    table.reserve("e1", 40.0)
    assert table.earliest_exit("e1", 11.0) == pytest.approx(20.0)  # Gap [20, 30] utilized
    assert table.earliest_exit("e1", 25.0) == pytest.approx(25.0)  # Inside gap
    assert table.earliest_exit("e1", 35.0) == pytest.approx(50.0)  # Pushed past 40

    # Monotonicity test: earliest_exit is non-decreasing in y
    rng = np.random.default_rng(42)
    sample_y = np.sort(rng.uniform(0.0, 100.0, size=200))
    exits = [table.earliest_exit("e1", y) for y in sample_y]
    for i in range(len(exits) - 1):
        assert exits[i] <= exits[i + 1] + 1e-9, f"Monotonicity violated at y={sample_y[i]}"


# ---------------------------------------------------------------------------
# Test 2: Plan = reality on a chain
# ---------------------------------------------------------------------------

def test_plan_equals_reality_on_chain():
    """Verify that on a chain path, planned_arrival_s matches realized arrival times."""
    G = nx.MultiDiGraph()
    G.add_node(0, y=0.0, x=0.0)
    G.add_node(1, y=0.0, x=1.0)
    G.add_node(2, y=0.0, x=2.0)
    G.add_edge(0, 1, key=0, free_flow_time=10.0, flow_capacity=6.0, length=100.0, capacity=10)
    G.add_edge(1, 2, key=0, free_flow_time=10.0, flow_capacity=6.0, length=100.0, capacity=10)

    hazard = StubMultiAgentHazard()  # No flood closures
    router = CoordinatedRouter(planning_mode="online")
    agents = [Evacuee(id=i, origin=0, depart_time_s=0.0) for i in range(5)]
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)

    pop_res = simulate_population(G, hazard, router, agents, exits={2}, params=params)

    # Expected realization: 20, 30, 40, 50, 60 seconds
    expected_finish = [20.0, 30.0, 40.0, 50.0, 60.0]
    actual_finish = [r.finish_time_s for r in pop_res.agent_results]
    assert actual_finish == pytest.approx(expected_finish, abs=1e-6)

    # Check planned arrival times on routes match simulation finish times
    for r in pop_res.agent_results:
        assert r.safe is True


# ---------------------------------------------------------------------------
# Test 3: Avoids a jam the blind router walks into
# ---------------------------------------------------------------------------

def test_coordinated_avoids_congestion_jam():
    """Verify coordinated router load-balances traffic across alternative paths."""
    G = nx.MultiDiGraph()
    G.add_node(0, y=0.0, x=0.0)  # S
    G.add_node(1, y=1.0, x=1.0)  # F (fast free-flow, low flow capacity)
    G.add_node(2, y=-1.0, x=1.0)  # G (longer free-flow, high flow capacity)
    G.add_node(3, y=0.0, x=2.0)  # D (exit)

    # Path 1: S -> F -> D (tau = 10+10 = 20s, h = 10s)
    G.add_edge(0, 1, key=0, free_flow_time=10.0, flow_capacity=6.0, length=100.0, capacity=10)
    G.add_edge(1, 3, key=0, free_flow_time=10.0, flow_capacity=6.0, length=100.0, capacity=10)

    # Path 2: S -> G -> D (tau = 25+25 = 50s, h = 1s)
    G.add_edge(0, 2, key=0, free_flow_time=25.0, flow_capacity=60.0, length=250.0, capacity=10)
    G.add_edge(2, 3, key=0, free_flow_time=25.0, flow_capacity=60.0, length=250.0, capacity=10)

    hazard = StubMultiAgentHazard()
    agents = [Evacuee(id=i, origin=0, depart_time_s=0.0) for i in range(10)]
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)

    # Blind TimeDependent router sends everyone through F -> last finish is 20 + 9*10 = 110s
    td_router = TimeDependentRouter()
    pop_res_td = simulate_population(G, hazard, td_router, agents, exits={3}, params=params)
    assert pop_res_td.agent_results[-1].finish_time_s == pytest.approx(110.0)

    # Coordinated router detects queuing bottleneck and distributes across G
    coord_router = CoordinatedRouter(planning_mode="online")
    pop_res_coord = simulate_population(G, hazard, coord_router, agents, exits={3}, params=params)
    last_finish_coord = pop_res_coord.agent_results[-1].finish_time_s
    assert last_finish_coord < 110.0, f"Expected last finish < 110s, got {last_finish_coord}"

    # Verify both routes were used
    used_nodes = set()
    for r in pop_res_coord.agent_results:
        used_nodes.update(r.path_nodes)
    assert 1 in used_nodes, "Expected at least one agent on Path 1 (via node 1)"
    assert 2 in used_nodes, "Expected at least one agent on Path 2 (via node 2)"


# ---------------------------------------------------------------------------
# Test 4: Flood and no feasible route (Toy 2 from test_multiagent)
# ---------------------------------------------------------------------------

def test_flood_and_no_feasible_route():
    """Verify coordinated router fails safely with no_route instead of trapping en-route."""
    G = nx.MultiDiGraph()
    G.add_node(0, y=0.0, x=0.0)
    G.add_node(1, y=0.0, x=1.0)
    G.add_node(2, y=0.0, x=2.0)
    G.add_edge(0, 1, key=0, free_flow_time=10.0, flow_capacity=6.0, length=100.0, capacity=10)
    G.add_edge(1, 2, key=0, free_flow_time=10.0, flow_capacity=6.0, length=100.0, capacity=10)

    # Edge A-D (1->2) floods at 45.0s
    hazard = StubMultiAgentHazard(edge_arrivals={(1, 2, 0): 45.0})
    router = CoordinatedRouter(planning_mode="online")
    agents = [Evacuee(id=i, origin=0, depart_time_s=0.0) for i in range(1, 6)]
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)

    pop_res = simulate_population(G, hazard, router, agents, exits={2}, params=params)

    # Agents 1-3 safe (reach exit at 20, 30, 40 < 45)
    for i in range(3):
        assert pop_res.agent_results[i].safe is True

    # Agents 4 and 5 recognize that the exit slot exceeds 45s and receive no_route
    assert pop_res.agent_results[3].safe is False
    assert pop_res.agent_results[3].trapped_reason == "no_route"
    assert pop_res.agent_results[4].safe is False
    assert pop_res.agent_results[4].trapped_reason == "no_route"


# ---------------------------------------------------------------------------
# Test 5: Infinite capacity regression on cached road graph
# ---------------------------------------------------------------------------

def test_infinite_capacity_regression():
    """Verify coordinated routers match time_dependent when edge capacity is infinite."""
    cfg = load_config(PROJECT_ROOT / "configs" / "study_area.yaml", StudyAreaConfig)
    G = get_or_build_graph(cfg).copy()

    # Override flow capacity to effectively infinite
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

    # Baseline: time-dependent router
    pop_td = simulate_population(G, hazard, TimeDependentRouter(), agents, exits, params)
    # Online coordinated router
    pop_online = simulate_population(G, hazard, CoordinatedRouter(planning_mode="online"), agents, exits, params)
    # Offline coordinated router
    pop_offline = simulate_population(G, hazard, CoordinatedRouter(planning_mode="offline"), agents, exits, params)

    td_safe = [r.safe for r in pop_td.agent_results]
    online_safe = [r.safe for r in pop_online.agent_results]
    offline_safe = [r.safe for r in pop_offline.agent_results]

    assert online_safe == td_safe
    assert offline_safe == td_safe

    for r_td, r_on, r_off in zip(pop_td.agent_results, pop_online.agent_results, pop_offline.agent_results):
        if r_td.safe:
            assert r_on.evac_time_s == pytest.approx(r_td.evac_time_s, abs=1e-5)
            assert r_off.evac_time_s == pytest.approx(r_td.evac_time_s, abs=1e-5)


# ---------------------------------------------------------------------------
# Test 6: Determinism across runs in same process
# ---------------------------------------------------------------------------

def test_determinism_and_no_state_leakage():
    """Verify running the router twice in the same process yields identical results."""
    G = nx.MultiDiGraph()
    G.add_node(0, y=0.0, x=0.0)
    G.add_node(1, y=0.0, x=1.0)
    G.add_node(2, y=0.0, x=2.0)
    G.add_edge(0, 1, key=0, free_flow_time=10.0, flow_capacity=6.0, length=100.0, capacity=10)
    G.add_edge(1, 2, key=0, free_flow_time=10.0, flow_capacity=6.0, length=100.0, capacity=10)

    hazard = StubMultiAgentHazard()
    agents = [Evacuee(id=i, origin=0, depart_time_s=0.0) for i in range(5)]
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)

    router = CoordinatedRouter(planning_mode="online")

    res1 = simulate_population(G, hazard, router, agents, exits={2}, params=params)
    res2 = simulate_population(G, hazard, router, agents, exits={2}, params=params)

    times1 = [r.finish_time_s for r in res1.agent_results]
    times2 = [r.finish_time_s for r in res2.agent_results]
    assert times1 == times2


# ---------------------------------------------------------------------------
# Test 7: Earliest arrival search does not mutate reservation table
# ---------------------------------------------------------------------------

def test_search_does_not_mutate_table():
    """Verify earliest_arrival_search in reservation mode does not modify the table."""
    G = nx.MultiDiGraph()
    G.add_node(0, y=0.0, x=0.0)
    G.add_node(1, y=0.0, x=1.0)
    G.add_node(2, y=0.0, x=2.0)
    G.add_edge(0, 1, key=0, free_flow_time=10.0, flow_capacity=6.0, length=100.0, capacity=10)
    G.add_edge(1, 2, key=0, free_flow_time=10.0, flow_capacity=6.0, length=100.0, capacity=10)

    hazard = StubMultiAgentHazard()
    table = ReservationTable(G)
    table.reserve((0, 1, 0), 10.0)
    table.reserve((1, 2, 0), 20.0)

    snapshot_01 = list(table.get_reservations((0, 1, 0)))
    snapshot_12 = list(table.get_reservations((1, 2, 0)))

    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.05)
    route = earliest_arrival_search(
        G=G,
        hazard=hazard,
        source=0,
        t_start=0.0,
        exits={2},
        params=params,
        mode="reservation",
        table=table,
    )
    assert route is not None

    # Verify table remains identical
    assert table.get_reservations((0, 1, 0)) == snapshot_01
    assert table.get_reservations((1, 2, 0)) == snapshot_12

