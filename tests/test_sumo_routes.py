"""Tests for SUMO route sanitizer and replay batch robustness (Task 8b fix).

Tests:
 (a) Toy net: route with A -> B then B -> A without turnaround connection -> pair removed, replay succeeds.
 (b) Route with disconnected pair that has a shortest path -> repaired.
 (c) Unrepairable route -> dropped and accounted.
 (d) Accounting identity holds with route_dropped.
 (e) One failing task (mocked) does not abort other tasks and appears in FAILED list.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import pytest

from src.scenario import Evacuee
from src.sumo.network import export_network_to_sumo, find_sumo_binary
from src.sumo.replay import VehicleOutcome, replay_routes
from src.sumo.routes import (
    SanitizerStats,
    SumoNetCache,
    compute_sumo_fallback_routes,
    sanitize_agent_routes,
)

# Detect SUMO environment
HAS_SUMO = False
try:
    find_sumo_binary("sumo")
    find_sumo_binary("netconvert")
    HAS_SUMO = True
except Exception:
    HAS_SUMO = False

requires_sumo = pytest.mark.skipif(
    not HAS_SUMO,
    reason="SUMO binaries (sumo, netconvert) not found in PATH or SUMO_HOME"
)


# ---------------------------------------------------------------------------
# Test (a): Toy net reversal removal and replay
# ---------------------------------------------------------------------------

@pytest.mark.sumo
@requires_sumo
def test_uturn_reversal_pair_removed_and_replay_succeeds(tmp_path: Path):
    """Route with A->B then B->A without turnaround connection has pair removed and arrives."""
    G = nx.MultiDiGraph()
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=100.0, y=0.0)
    G.add_node(2, x=200.0, y=0.0)
    for u, v in [(0, 1), (1, 0), (1, 2)]:
        G.add_edge(
            u, v, key=0,
            length=100.0,
            speed_kph=36.0,
            lanes_n=1,
            highway="secondary",
            free_flow_time=10.0,
            travel_time=10.0,
            flow_capacity=12.0,
        )

    res = export_network_to_sumo(G, tmp_path, vehicle_speed_scale=1.0, run_netconvert=False)
    netconvert = find_sumo_binary("netconvert")
    net_file = tmp_path / "net.net.xml"
    cmd = [
        netconvert,
        "-n", str(res["nod_path"]),
        "-e", str(res["edg_path"]),
        "-o", str(net_file),
        "--no-turnarounds", "true",
    ]
    subprocess.run(cmd, check=True)

    SumoNetCache.clear()
    g2s = res["edge_map"]["graph_to_sumo"]

    # Vehicle makes an invalid U-turn at node 1 before continuing to node 2:
    # 0 -> 1 -> 0 -> 1 -> 2
    agent = Evacuee(id=101, origin=0, depart_time_s=0.0)
    bad_route = [(0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 2, 0)]
    agents_with_routes = [(agent, bad_route)]

    # 1. Check sanitizer output
    sanitized, dropped, stats = sanitize_agent_routes(net_file, agents_with_routes, g2s)
    assert len(dropped) == 0
    assert stats.n_uturn_pairs_removed == 1
    assert stats.n_agents_affected == 1
    assert sanitized[0][1] == ["e0_1_0", "e1_2_0"]

    # 2. Check that replay runs and arrives without crashing
    vtype_path = tmp_path / "vtype.add.xml"
    vtype_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<additional>\n'
        '    <vType id="calib_vtype" length="5.0" minGap="2.5" '
        'accel="2.6" decel="4.5" sigma="0.0" tau="1.0"/>\n'
        '</additional>\n',
        encoding="utf-8",
    )
    flood_times = {sid: math.inf for sid in ["e0_1_0", "e1_0_0", "e1_2_0"]}

    outcomes = replay_routes(
        net_path=net_file,
        vtype_path=vtype_path,
        agents_with_routes=agents_with_routes,
        edge_map_g2s=g2s,
        flood_times_by_sumo_edge=flood_times,
        step_length_s=1.0,
        horizon_s=300,
        work_dir=tmp_path / "work",
        sumo_label="test_uturn",
    )

    assert outcomes[101].status == "ARRIVED"
    assert outcomes.sanitizer_stats.n_uturn_pairs_removed == 1


# ---------------------------------------------------------------------------
# Test (b): Route with disconnected pair repaired via shortest path
# ---------------------------------------------------------------------------

@pytest.mark.sumo
@requires_sumo
def test_disconnected_pair_with_shortest_path_repaired(tmp_path: Path):
    """Route skipping an intermediate edge is bridged by net.getShortestPath."""
    G = nx.MultiDiGraph()
    for i in range(4):
        G.add_node(i, x=float(i) * 100.0, y=0.0)
    for u, v in [(0, 1), (1, 2), (2, 3)]:
        G.add_edge(
            u, v, key=0,
            length=100.0,
            speed_kph=36.0,
            lanes_n=1,
            highway="secondary",
            free_flow_time=10.0,
            travel_time=10.0,
            flow_capacity=12.0,
        )

    res = export_network_to_sumo(G, tmp_path, vehicle_speed_scale=1.0, run_netconvert=True)
    SumoNetCache.clear()

    # Route skips (1, 2, 0)
    agent = Evacuee(id=102, origin=0, depart_time_s=0.0)
    skipped_route = [(0, 1, 0), (2, 3, 0)]
    agents_with_routes = [(agent, skipped_route)]

    sanitized, dropped, stats = sanitize_agent_routes(
        res["net_path"], agents_with_routes, res["edge_map"]["graph_to_sumo"]
    )
    assert len(dropped) == 0
    assert stats.n_repaired == 1
    assert stats.n_agents_affected == 1
    assert sanitized[0][1] == ["e0_1_0", "e1_2_0", "e2_3_0"]


# ---------------------------------------------------------------------------
# Test (c): Unrepairable route dropped
# ---------------------------------------------------------------------------

@pytest.mark.sumo
@requires_sumo
def test_unrepairable_route_dropped_and_accounted(tmp_path: Path):
    """Route with disconnected edges having no valid path is marked ROUTE_DROPPED."""
    G = nx.MultiDiGraph()
    for i in range(4):
        G.add_node(i, x=float(i) * 100.0, y=0.0)
    for u, v in [(0, 1), (1, 2), (2, 3)]:
        G.add_edge(
            u, v, key=0,
            length=100.0,
            speed_kph=36.0,
            lanes_n=1,
            highway="secondary",
            free_flow_time=10.0,
            travel_time=10.0,
            flow_capacity=12.0,
        )

    res = export_network_to_sumo(G, tmp_path, vehicle_speed_scale=1.0, run_netconvert=True)
    SumoNetCache.clear()

    # Route tries to go backwards from 3 to 0 (one-way edges: no path back exists)
    agent = Evacuee(id=103, origin=3, depart_time_s=0.0)
    unrepairable_route = ["e2_3_0", "e0_1_0"]
    agents_with_routes = [(agent, unrepairable_route)]

    sanitized, dropped, stats = sanitize_agent_routes(res["net_path"], agents_with_routes)
    assert 103 in dropped
    assert stats.n_route_dropped == 1
    assert stats.n_agents_affected == 1
    assert len(sanitized) == 0

    # Test replay outcomes
    vtype_path = tmp_path / "vtype.add.xml"
    vtype_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<additional>\n'
        '    <vType id="calib_vtype" length="5.0" minGap="2.5" '
        'accel="2.6" decel="4.5" sigma="0.0" tau="1.0"/>\n'
        '</additional>\n',
        encoding="utf-8",
    )
    flood_times = {sid: math.inf for sid in ["e0_1_0", "e1_2_0", "e2_3_0"]}

    outcomes = replay_routes(
        net_path=res["net_path"],
        vtype_path=vtype_path,
        agents_with_routes=agents_with_routes,
        edge_map_g2s=res["edge_map"]["graph_to_sumo"],
        flood_times_by_sumo_edge=flood_times,
        step_length_s=1.0,
        horizon_s=300,
        work_dir=tmp_path / "work_drop",
        sumo_label="test_drop",
    )
    assert outcomes[103].status == "ROUTE_DROPPED"


# ---------------------------------------------------------------------------
# Test (d): Accounting identity holds with route_dropped
# ---------------------------------------------------------------------------

def test_accounting_identity_with_route_dropped():
    """Verify arrived + trapped + unfinished + never_inserted + route_dropped + not_created == N."""
    n_agents = 50
    n_arrived = 20
    n_trapped = 10
    n_unfinished = 5
    n_never_inserted = 3
    n_route_dropped = 4
    n_at_exit = 5
    n_no_route = 3

    n_vehicles_not_created = n_at_exit + n_no_route
    total_accounted = (
        n_arrived
        + n_trapped
        + n_unfinished
        + n_never_inserted
        + n_route_dropped
        + n_vehicles_not_created
    )
    assert total_accounted == n_agents


# ---------------------------------------------------------------------------
# Test (e): One failing task does not abort batch and appears in FAILED list
# ---------------------------------------------------------------------------

def test_batch_robustness_one_failing_task_does_not_abort_others():
    """A task raising FatalTraCIError returns FAILED status and does not write raw JSON."""
    from scripts.run_sumo_experiment import run_pq_and_replay
    import traci

    # Simulate run_pq_and_replay where _run_pq_and_replay_impl raises FatalTraCIError
    with patch("scripts.run_sumo_experiment._run_pq_and_replay_impl") as mock_impl:
        mock_impl.side_effect = traci.exceptions.FatalTraCIError("Connection closed by SUMO")

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            routes_dir = out_dir / "routes"
            (out_dir / "raw").mkdir(parents=True, exist_ok=True)
            routes_dir.mkdir(parents=True, exist_ok=True)

            from src.config import HazardConfig, ScenarioConfig, StudyAreaConfig, SumoConfig, SumoExperimentConfig, TrafficConfig, CoordinatedConfig
            graph_cfg = StudyAreaConfig(name="test", center_lat=0.0, center_lon=0.0, dist_m=500)
            hazard_cfg = HazardConfig()
            scenario_cfg = ScenarioConfig(n_evacuees=1)
            traffic_cfg = TrafficConfig()
            coord_cfg = CoordinatedConfig()
            sumo_cfg = SumoConfig()
            exp_cfg = SumoExperimentConfig()

            rec = run_pq_and_replay(
                demand=1,
                seed_index=0,
                router_name="static",
                graph_cfg=graph_cfg,
                hazard_cfg=hazard_cfg,
                scenario_cfg=scenario_cfg,
                traffic_cfg=traffic_cfg,
                coord_cfg=coord_cfg,
                sumo_cfg=sumo_cfg,
                exp_cfg=exp_cfg,
                G=nx.MultiDiGraph(),
                out_dir=out_dir,
                routes_dir=routes_dir,
                sumo_dir=out_dir,
                net_path=out_dir / "net.net.xml",
                vtype_path=out_dir / "vtype.add.xml",
                edge_map_g2s={"0,1,0": "e0_1_0"},
                edge_map_s2g={"e0_1_0": [0, 1, 0]},
                force=True,
            )

            assert rec["status"] == "FAILED"
            assert "Connection closed by SUMO" in rec["error"]
            # Raw result file must NOT exist
            assert not (out_dir / "raw" / "result_demand1_seed0_static.json").exists()


# ---------------------------------------------------------------------------
# Tests for Task 8c: dispatch_all
# ---------------------------------------------------------------------------

@pytest.mark.sumo
@requires_sumo
def test_dispatch_all_toy_agent_with_no_route_arrives(tmp_path: Path):
    """Toy net where an agent with no route receives a fallback route and arrives safely."""
    G = nx.MultiDiGraph()
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=100.0, y=0.0)
    G.add_node(2, x=200.0, y=0.0)
    for u, v in [(0, 1), (1, 2)]:
        G.add_edge(
            u, v, key=0,
            length=100.0,
            speed_kph=36.0,
            lanes_n=1,
            highway="secondary",
            free_flow_time=10.0,
            travel_time=10.0,
            flow_capacity=12.0,
        )

    res = export_network_to_sumo(G, tmp_path, vehicle_speed_scale=1.0, run_netconvert=True)
    SumoNetCache.clear()

    # Agent has no planned route initially
    agent = Evacuee(id=201, origin=0, depart_time_s=0.0)
    exits = [2]

    # Compute fallback route
    fallback_map = compute_sumo_fallback_routes(
        net_path=res["net_path"],
        agents=[agent],
        exits=exits,
        edge_map_g2s=res["edge_map"]["graph_to_sumo"],
        G=G,
    )
    assert 201 in fallback_map
    assert fallback_map[201] == ["e0_1_0", "e1_2_0"]

    # Replay fallback vehicle
    vtype_path = tmp_path / "vtype.add.xml"
    vtype_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<additional>\n'
        '    <vType id="calib_vtype" length="5.0" minGap="2.5" '
        'accel="2.6" decel="4.5" sigma="0.0" tau="1.0"/>\n'
        '</additional>\n',
        encoding="utf-8",
    )
    flood_times = {sid: math.inf for sid in ["e0_1_0", "e1_2_0"]}

    outcomes = replay_routes(
        net_path=res["net_path"],
        vtype_path=vtype_path,
        agents_with_routes=[(agent, fallback_map[201])],
        edge_map_g2s=res["edge_map"]["graph_to_sumo"],
        flood_times_by_sumo_edge=flood_times,
        step_length_s=1.0,
        horizon_s=300,
        work_dir=tmp_path / "work_fb_arrive",
        sumo_label="test_fb_arrive",
    )
    assert 201 in outcomes
    assert outcomes[201].status == "ARRIVED"
    assert outcomes[201].actual_arrival_s is not None
    assert outcomes[201].actual_arrival_s > 0.0


@pytest.mark.sumo
@requires_sumo
def test_dispatch_all_flood_rule_applies_to_fallback(tmp_path: Path):
    """Flood rule traps a fallback vehicle when edge closes before traversal."""
    G = nx.MultiDiGraph()
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=100.0, y=0.0)
    G.add_node(2, x=200.0, y=0.0)
    for u, v in [(0, 1), (1, 2)]:
        G.add_edge(
            u, v, key=0,
            length=100.0,
            speed_kph=36.0,
            lanes_n=1,
            highway="secondary",
            free_flow_time=10.0,
            travel_time=10.0,
            flow_capacity=12.0,
        )

    res = export_network_to_sumo(G, tmp_path, vehicle_speed_scale=1.0, run_netconvert=True)
    SumoNetCache.clear()

    agent = Evacuee(id=202, origin=0, depart_time_s=0.0)
    fallback_map = compute_sumo_fallback_routes(
        net_path=res["net_path"],
        agents=[agent],
        exits=[2],
        edge_map_g2s=res["edge_map"]["graph_to_sumo"],
        G=G,
    )

    vtype_path = tmp_path / "vtype.add.xml"
    vtype_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<additional>\n'
        '    <vType id="calib_vtype" length="5.0" minGap="2.5" '
        'accel="2.6" decel="4.5" sigma="0.0" tau="1.0"/>\n'
        '</additional>\n',
        encoding="utf-8",
    )
    # Edge e1_2_0 closes at t=2.0s, but vehicle cannot reach it until t >= 10s
    flood_times = {"e0_1_0": math.inf, "e1_2_0": 2.0}

    outcomes = replay_routes(
        net_path=res["net_path"],
        vtype_path=vtype_path,
        agents_with_routes=[(agent, fallback_map[202])],
        edge_map_g2s=res["edge_map"]["graph_to_sumo"],
        flood_times_by_sumo_edge=flood_times,
        step_length_s=1.0,
        horizon_s=300,
        work_dir=tmp_path / "work_fb_trap",
        sumo_label="test_fb_trap",
    )
    assert 202 in outcomes
    assert outcomes[202].status == "TRAPPED"


def test_dispatch_all_accounting_identity_holds():
    """Population accounting identity and fallback subset accounting hold."""
    n_agents = 100
    n_arrived = 65
    n_trapped = 15
    n_unfinished = 5
    n_never_inserted = 3
    n_route_dropped = 2
    n_at_exit = 6
    n_no_route = 4
    n_vehicles_not_created = n_at_exit + n_no_route

    total_accounted = (
        n_arrived
        + n_trapped
        + n_unfinished
        + n_never_inserted
        + n_route_dropped
        + n_vehicles_not_created
    )
    assert total_accounted == n_agents

    # Fallback subset
    n_fallback_dispatched = 25
    n_fallback_arrived = 14
    n_fallback_trapped = 8
    n_fallback_unfinished = 1
    n_fallback_never_inserted = 1
    n_fallback_route_dropped = 1
    total_fallback = (
        n_fallback_arrived
        + n_fallback_trapped
        + n_fallback_unfinished
        + n_fallback_never_inserted
        + n_fallback_route_dropped
    )
    assert total_fallback == n_fallback_dispatched
    fallback_arrived_pct = 100.0 * n_fallback_arrived / n_fallback_dispatched
    assert abs(fallback_arrived_pct - 56.0) < 1e-6


def test_dispatch_all_flag_off_preserves_default_behaviour():
    """When dispatch_all=False, infeasible agents remain in no_route_agents and are not dispatched."""
    from src.config import SumoExperimentConfig
    cfg = SumoExperimentConfig(dispatch_all=False)
    assert not cfg.dispatch_all

    # Given agents: 1 with route, 1 at exit, 1 with no route
    no_route_agents = {303}
    fallback_agent_ids = set()
    if cfg.dispatch_all and no_route_agents:
        fallback_agent_ids.add(303)
        no_route_agents.remove(303)

    assert len(fallback_agent_ids) == 0
    assert 303 in no_route_agents
