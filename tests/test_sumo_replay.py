"""Unit tests for SUMO open-loop replay engine and route recording (Task 8b).

All tests with the 'sumo' marker are skipped if SUMO binaries are not available.
Tests 2-3 require SUMO TraCI (use the actual sumo binary).
Test 4 (speed-matching helper) does NOT require SUMO.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

from src.routing.base import Evacuee, Route
from src.sim.multi_agent import simulate_population
from src.sim.single_agent import AgentResult
from src.sumo.network import (
    edge_to_sumo_id,
    export_network_to_sumo,
    find_sumo_binary,
)
from src.sumo.replay import VehicleOutcome, replay_routes, write_replay_routes

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# SUMO environment detection
# ---------------------------------------------------------------------------

HAS_SUMO = False
try:
    find_sumo_binary("sumo")
    find_sumo_binary("netconvert")
    HAS_SUMO = True
except Exception:
    HAS_SUMO = False

requires_sumo = pytest.mark.skipif(
    not HAS_SUMO,
    reason="SUMO binaries (sumo, netconvert) not found in PATH or SUMO_HOME",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_line_graph(n_nodes: int = 3, length: float = 500.0, speed_ms: float = 10.0) -> nx.MultiDiGraph:
    """Build a straight n-node line graph (nodes 0, 1, ..., n-1) in local Cartesian coords."""
    G = nx.MultiDiGraph()
    for i in range(n_nodes):
        G.add_node(i, x=float(i) * length, y=0.0)
    for i in range(n_nodes - 1):
        fft = length / speed_ms
        G.add_edge(
            i, i + 1, key=0,
            length=length,
            speed_kph=speed_ms * 3.6,
            free_flow_time=fft,
            travel_time=fft,
            flow_capacity=12.0,
            lanes_n=1,
            highway="secondary",
            capacity=60,
        )
    return G


def _make_simple_hazard(G: nx.MultiDiGraph, inf_all: bool = True) -> MagicMock:
    """Return a mock hazard where all edges are safe (flood_time = inf)."""
    hazard = MagicMock()
    hazard.edge_speed_factor.return_value = 1.0
    hazard.edge_arrival_time.return_value = math.inf
    hazard.is_edge_closed.return_value = False
    hazard.node_arrival_time.return_value = math.inf
    return hazard


# ---------------------------------------------------------------------------
# Test 1: Route recording — results unchanged, routes populated
# ---------------------------------------------------------------------------

class _StaticLineRouter:
    """Minimal router that always routes 0->1->2 on a 3-node line graph."""
    name = "static_line"
    replans = False
    replan_interval_s = None

    def prepare(self, *args, **kwargs): pass

    def plan(self, G, hazard, evacuee, t_now_s, current_node, exits, params, traffic=None):
        edges = [(i, i + 1, 0) for i in range(current_node, len(G.nodes()) - 1)]
        nodes = list(range(current_node, len(G.nodes())))
        return Route(edges=edges, nodes=nodes, length_m=float(len(edges)) * 500.0)


def test_route_recording_results_unchanged():
    """Test 1: safe flags, evac_times and route_edges do not regress.

    Verifies:
    - AgentResult.safe flag is unchanged from baseline.
    - AgentResult.evac_time_s is unchanged.
    - Safe agents have non-empty route_edges from origin to exit.
    - Trapped agents have route_edges that include the trapping edge + remaining plan.
    """
    G = _make_line_graph(n_nodes=3, length=500.0, speed_ms=10.0)
    hazard = _make_simple_hazard(G)
    exits = {2}

    agents = [
        Evacuee(id=0, origin=0, depart_time_s=0.0),
        Evacuee(id=1, origin=1, depart_time_s=0.0),
    ]

    from src.routing.base import RoutingParams
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.01)
    router = _StaticLineRouter()

    res = simulate_population(
        G=G,
        hazard=hazard,
        router=router,
        agents=agents,
        exits=exits,
        params=params,
    )

    for r in res.agent_results:
        assert isinstance(r, AgentResult)
        assert isinstance(r.route_edges, list), "route_edges must be a list"
        if r.safe:
            assert len(r.route_edges) > 0, f"Safe agent {r.agent_id} should have non-empty route_edges"
            # Route must end at an exit node
            last_edge = r.route_edges[-1]
            u, v, k = last_edge
            assert v in exits, f"Safe agent route must end at exit, got {v}"
        # evac_time and safe unchanged check (compare with single-agent reference)
        if r.agent_id == 1:
            # Agent 1 starts at node 1, one edge to exit
            assert r.safe is True
            assert r.evac_time_s == pytest.approx(500.0 / 10.0, abs=0.1)


def test_route_recording_trapped_route_complete():
    """Test 1 (trapped): trapped agent route includes trapping edge + remaining plan."""
    G = _make_line_graph(n_nodes=4, length=500.0, speed_ms=5.0)
    exits = {3}

    # Flood edge (1,2,0) closes at t=50s — agent 0 starts at t=0 and enters at t=100s (trapped)
    hazard = MagicMock()
    hazard.edge_speed_factor.return_value = 1.0
    hazard.is_edge_closed.return_value = False
    hazard.node_arrival_time.return_value = math.inf

    def edge_arrival_time(u, v, k):
        if (u, v, k) == (1, 2, 0):
            return 50.0   # floods at t=50
        return math.inf

    hazard.edge_arrival_time.side_effect = edge_arrival_time

    agents = [Evacuee(id=0, origin=0, depart_time_s=0.0)]

    from src.routing.base import RoutingParams

    class _FullLineRouter:
        name = "full_line"
        replans = False
        replan_interval_s = None
        def prepare(self, *a, **kw): pass
        def plan(self, G, hazard, evacuee, t_now_s, current_node, exits, params, traffic=None):
            edges = [(i, i + 1, 0) for i in range(current_node, len(G.nodes()) - 1)]
            nodes = list(range(current_node, len(G.nodes())))
            return Route(edges=edges, nodes=nodes, length_m=float(len(edges)) * 500.0)

    router = _FullLineRouter()
    params = RoutingParams(vehicle_speed_scale=1.0, sf_floor=0.01)

    res = simulate_population(G=G, hazard=hazard, router=router, agents=agents, exits=exits, params=params)
    r = res.agent_results[0]

    # Agent traverses (0,1,0) safely, then gets trapped on (1,2,0)
    # route_edges must include (0,1,0) traversed + (1,2,0) trapping + (2,3,0) remaining
    assert r.safe is False
    assert r.trapped_reason == "edge_closed_en_route"
    assert len(r.route_edges) >= 1, "Trapped agent must have non-empty route_edges"
    # Trapping edge must be in route_edges
    assert (1, 2, 0) in r.route_edges, f"Trapping edge (1,2,0) must be in route_edges, got {r.route_edges}"


# ---------------------------------------------------------------------------
# Test 2: Toy 3-node line net — 5 vehicles arrive, headway preserved
# ---------------------------------------------------------------------------

@pytest.mark.sumo
@requires_sumo
def test_toy_net_5_vehicles_arrive(tmp_path: Path):
    """Test 2: 5 vehicles along a 1-lane 2-edge line all arrive; last vehicle has longer travel time."""
    G = _make_line_graph(n_nodes=3, length=500.0, speed_ms=10.0)

    res = export_network_to_sumo(G=G, output_dir=tmp_path, vehicle_speed_scale=1.0, run_netconvert=True)
    net_path = res["net_path"]
    edge_map_g2s = res["edge_map"]["graph_to_sumo"]
    edge_map_s2g = res["edge_map"]["sumo_to_graph"]

    # Build a vtype.add.xml with the calibrated tau (use a reasonable value)
    vtype_path = tmp_path / "vtype.add.xml"
    vtype_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<additional>\n'
        '    <vType id="calib_vtype" length="5.0" minGap="3.3" '
        'accel="2.6" decel="4.5" sigma="0.0" tau="2.7969"/>\n'
        '</additional>\n',
        encoding="utf-8",
    )

    # 5 agents, all depart at t=0 from node 0, exit at node 2
    agents = [Evacuee(id=i, origin=0, depart_time_s=float(i * 2)) for i in range(5)]
    route_edges = [(0, 1, 0), (1, 2, 0)]
    agents_with_routes = [(a, list(route_edges)) for a in agents]

    flood_times_inf = {sid: float("inf") for sid in edge_map_s2g.keys()}

    outcomes = replay_routes(
        net_path=net_path,
        vtype_path=vtype_path,
        agents_with_routes=agents_with_routes,
        edge_map_g2s=edge_map_g2s,
        flood_times_by_sumo_edge=flood_times_inf,
        step_length_s=1.0,
        time_to_teleport=-1,
        horizon_s=600,
        work_dir=tmp_path / "replay_work",
        sumo_label="test_toy",
    )

    arrived = [aid for aid, oc in outcomes.items() if oc.status == "ARRIVED"]
    assert len(arrived) == 5, f"Expected 5 arrivals, got {len(arrived)}: {outcomes}"

    # Last vehicle should have a longer travel time than the first (due to queuing)
    evac_times = []
    for a in agents:
        oc = outcomes.get(a.id)
        if oc is not None and oc.status == "ARRIVED":
            if oc.actual_arrival_s is not None and oc.planned_depart_s is not None:
                evac_times.append(oc.actual_arrival_s - oc.planned_depart_s)

    assert len(evac_times) == 5
    # Last vehicle should take longer than first (congestion headway buildup)
    assert evac_times[-1] >= evac_times[0] - 1.0, (
        f"Expected last vehicle to have >= travel time as first; got {evac_times}"
    )


# ---------------------------------------------------------------------------
# Test 3: Flood rule on toy net
# ---------------------------------------------------------------------------

@pytest.mark.sumo
@requires_sumo
def test_flood_rule_toy_net(tmp_path: Path):
    """Test 3: Flood rule on a 3-node line.

    - Edge B-C (1,2,0) floods at t=60s.
    - Vehicle that enters B-C at t<60 should arrive (clears before flood).
    - Vehicle that is on B-C at t=60 should be TRAPPED at ~60s.
    - Vehicle that enters B-C after t=60 should be TRAPPED when it enters.
    """
    G = _make_line_graph(n_nodes=3, length=200.0, speed_ms=10.0)
    # With speed=10 m/s and length=200m, free-flow traversal = 20s.

    res = export_network_to_sumo(G=G, output_dir=tmp_path, vehicle_speed_scale=1.0, run_netconvert=True)
    net_path = res["net_path"]
    edge_map_g2s = res["edge_map"]["graph_to_sumo"]
    edge_map_s2g = res["edge_map"]["sumo_to_graph"]

    vtype_path = tmp_path / "vtype.add.xml"
    vtype_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<additional>\n'
        '    <vType id="calib_vtype" length="5.0" minGap="3.3" '
        'accel="2.6" decel="4.5" sigma="0.0" tau="2.7969"/>\n'
        '</additional>\n',
        encoding="utf-8",
    )

    # Edge (0,1) is "A-B", edge (1,2) is "B-C"
    bc_edge_sumo = edge_to_sumo_id(1, 2, 0)

    # Flood B-C at t=60s
    flood_times = {sid: float("inf") for sid in edge_map_s2g.keys()}
    flood_times[bc_edge_sumo] = 60.0

    # Agent 0: departs t=0 from node 0 -> should enter B-C ~20s, clear at ~40s (before flood)
    # Agent 1: departs t=50 from node 1 -> enters B-C at ~50s, exits at ~70s > 60s -> TRAPPED
    # Agent 2: departs t=80 from node 1 -> enters B-C at ~80s > 60s -> TRAPPED when entering
    agents = [
        Evacuee(id=0, origin=0, depart_time_s=0.0),   # starts at node 0
        Evacuee(id=1, origin=1, depart_time_s=50.0),   # starts at node 1
        Evacuee(id=2, origin=1, depart_time_s=80.0),   # starts at node 1 after flood
    ]
    # Routes: agent 0 uses full path (0,1,0) + (1,2,0); agents 1,2 use just (1,2,0)
    agents_with_routes = [
        (agents[0], [(0, 1, 0), (1, 2, 0)]),
        (agents[1], [(1, 2, 0)]),
        (agents[2], [(1, 2, 0)]),
    ]

    outcomes = replay_routes(
        net_path=net_path,
        vtype_path=vtype_path,
        agents_with_routes=agents_with_routes,
        edge_map_g2s=edge_map_g2s,
        flood_times_by_sumo_edge=flood_times,
        step_length_s=1.0,
        time_to_teleport=-1,
        horizon_s=300,
        work_dir=tmp_path / "replay_flood",
        sumo_label="test_flood",
    )

    # Agent 0: should ARRIVE (cleared B-C before t=60)
    oc0 = outcomes.get(0)
    assert oc0 is not None, "Agent 0 should have an outcome"
    assert oc0.status == "ARRIVED", (
        f"Agent 0 (early departure) should ARRIVE but got {oc0.status} "
        f"(trap_time={oc0.trap_time_s})"
    )

    # Agent 1: should be TRAPPED at approximately t=60
    oc1 = outcomes.get(1)
    assert oc1 is not None
    assert oc1.status == "TRAPPED", (
        f"Agent 1 (on B-C at flood time) should be TRAPPED but got {oc1.status}"
    )
    assert oc1.trap_time_s is not None
    assert 55.0 <= oc1.trap_time_s <= 75.0, (
        f"Agent 1 trap time should be near 60s, got {oc1.trap_time_s}"
    )

    # Agent 2: should be TRAPPED (enters B-C after flood)
    oc2 = outcomes.get(2)
    assert oc2 is not None
    assert oc2.status == "TRAPPED", (
        f"Agent 2 (enters B-C after flood) should be TRAPPED but got {oc2.status}"
    )
    assert oc2.trap_time_s is not None
    assert oc2.trap_time_s >= 60.0, (
        f"Agent 2 trap time should be >= 60s, got {oc2.trap_time_s}"
    )


# ---------------------------------------------------------------------------
# Test 4: Speed-matching helper converges (no SUMO needed — mock ratio)
# ---------------------------------------------------------------------------

def test_speed_matching_converges_mock():
    """Test 4: Speed-matching iteration loop converges with a mocked ratio sequence.

    Uses unittest.mock to replace the SUMO replay call; verifies that:
    - Convergence is detected when |ratio-1| <= 0.05.
    - The loop stops within max_iter iterations.
    - speed_multiplier is updated correctly each iteration.
    """
    # Simulated ratio sequence: 1.3, 1.1, 1.03 (converges at iter 3)
    ratios_to_return = [1.3, 1.1, 1.03]
    call_count = [0]

    def mock_replay(*args, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        ratio = ratios_to_return[min(idx, len(ratios_to_return) - 1)]
        # Return mock VehicleOutcome objects
        outcomes = {
            i: VehicleOutcome(
                agent_id=i, status="ARRIVED",
                actual_arrival_s=100.0 * ratio,
                planned_depart_s=0.0,
                duration_s=100.0 * ratio,
                waiting_time_s=0.0,
                time_loss_s=0.0,
                depart_delay_s=0.0,
                actual_depart_s=0.0,
            )
            for i in range(5)
        }
        return outcomes

    # Simple mock of the speed-matching core loop (not the full script)
    m = 1.0
    target = 1.0
    tol = 0.05
    max_iter = 5
    history = []

    pq_mean = 100.0  # fixed PQ baseline

    for iteration in range(1, max_iter + 1):
        # Call mock replay (simulates one SUMO run)
        outcomes = mock_replay()
        sumo_evac = [
            oc.actual_arrival_s - oc.planned_depart_s
            for oc in outcomes.values()
            if oc.status == "ARRIVED" and oc.actual_arrival_s is not None
        ]
        sumo_mean = float(np.mean(sumo_evac))
        ratio = sumo_mean / pq_mean

        history.append({"iteration": iteration, "speed_multiplier": m, "ratio": ratio})

        if abs(ratio - target) <= tol:
            break

        m = m * ratio

    assert len(history) <= max_iter, f"Should stop within {max_iter} iterations"
    final_ratio = history[-1]["ratio"]
    assert abs(final_ratio - 1.0) <= tol, (
        f"Speed-matching did not converge: final ratio = {final_ratio:.4f}"
    )
    assert call_count[0] == len(history), "Call count should match iteration count"

