"""Unit tests for SUMO network export, calibration, and TraCI simulation (Task 8a)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import networkx as nx
import pytest

from src.sumo.calibrate import measure_discharge
from src.sumo.network import (
    edge_to_sumo_id,
    export_network_to_sumo,
    find_sumo_binary,
    sumo_id_to_edge,
    validate_sumo_network,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
# Test 1: No SUMO needed - Edge ID format, round-trip mapping, and XML attributes
# ---------------------------------------------------------------------------

def test_edge_id_format_and_xml_attributes(tmp_path: Path):
    """Verify edge id format, edge_map round-trip, and plain XML node/edge attributes."""
    # Build toy 3-node graph with multiple edges
    G = nx.MultiDiGraph()
    G.add_node(101, x=100.0, y=200.0)
    G.add_node(102, x=200.0, y=200.0)
    G.add_node(103, x=300.0, y=200.0)

    # Edge 1: primary road, 2 lanes, 40 kph -> 40 * 0.5 / 3.6 = 5.556 m/s
    G.add_edge(
        101, 102, key=0,
        lanes_n=2,
        speed_kph=40.0,
        length=100.0,
        highway="primary",
    )
    # Edge 2: parallel edge
    G.add_edge(
        101, 102, key=1,
        lanes_n=1,
        speed_kph=30.0,
        length=105.0,
        highway="secondary",
    )
    # Edge 3: tertiary
    G.add_edge(
        102, 103, key=0,
        lanes_n=1,
        speed_kph=30.0,
        length=100.0,
        highway="tertiary",
    )

    # Export XML only (no netconvert needed)
    res = export_network_to_sumo(
        G=G,
        output_dir=tmp_path,
        vehicle_speed_scale=0.5,
        run_netconvert=False,
    )

    # 1. Round-trip edge_map check
    edge_map = res["edge_map"]
    g2s = edge_map["graph_to_sumo"]
    s2g = edge_map["sumo_to_graph"]

    for u, v, k in G.edges(keys=True):
        gkey = f"{u},{v},{k}"
        sumo_id = edge_to_sumo_id(u, v, k)
        assert g2s[gkey] == sumo_id
        assert s2g[sumo_id] == [u, v, k]
        # Helper function round-trip
        assert sumo_id_to_edge(sumo_id) == (u, v, k)

    # 2. Check nodes.nod.xml structure and attributes
    nod_tree = ET.parse(res["nod_path"])
    nod_root = nod_tree.getroot()
    node_elements = {elem.get("id"): elem for elem in nod_root.findall("node")}
    assert len(node_elements) == 3
    for n in G.nodes():
        elem = node_elements[str(n)]
        assert elem is not None
        assert float(elem.get("x")) == pytest.approx(G.nodes[n]["x"])
        assert float(elem.get("y")) == pytest.approx(G.nodes[n]["y"])

    # 3. Check edges.edg.xml attributes (lanes, speed in m/s, length, priority)
    edg_tree = ET.parse(res["edg_path"])
    edg_root = edg_tree.getroot()
    edge_elements = {elem.get("id"): elem for elem in edg_root.findall("edge")}
    assert len(edge_elements) == 3

    # Check edge e101_102_0: primary, 2 lanes, speed 5.556 m/s, length 100.0, priority 3
    e0 = edge_elements["e101_102_0"]
    assert e0.get("from") == "101"
    assert e0.get("to") == "102"
    assert int(e0.get("numLanes")) == 2
    assert float(e0.get("speed")) == pytest.approx(40.0 * 0.5 / 3.6, abs=1e-3)
    assert float(e0.get("length")) == pytest.approx(100.0)
    assert int(e0.get("priority")) == 3

    # Check edge e101_102_1: secondary, 1 lane, priority 2
    e1 = edge_elements["e101_102_1"]
    assert int(e1.get("numLanes")) == 1
    assert int(e1.get("priority")) == 2

    # Check edge e102_103_0: tertiary, 1 lane, priority 1
    e2 = edge_elements["e102_103_0"]
    assert int(e2.get("numLanes")) == 1
    assert int(e2.get("priority")) == 1


# ---------------------------------------------------------------------------
# Test 2: SUMO needed - Export 4-node graph, netconvert succeeds, edge count matches
# ---------------------------------------------------------------------------

@pytest.mark.sumo
@requires_sumo
def test_export_toy_4node_netconvert_matches(tmp_path: Path):
    """Verify netconvert successfully compiles a 4-node network and edge counts match."""
    G = nx.MultiDiGraph()
    G.add_node(1, x=0.0, y=50.0)
    G.add_node(2, x=50.0, y=100.0)
    G.add_node(3, x=50.0, y=0.0)
    G.add_node(4, x=100.0, y=50.0)

    # Diamond: 1->2, 1->3, 2->4, 3->4
    edges_to_add = [(1, 2), (1, 3), (2, 4), (3, 4)]
    for u, v in edges_to_add:
        G.add_edge(
            u, v, key=0,
            lanes_n=1,
            speed_kph=36.0,
            length=70.71,
            highway="secondary",
            free_flow_time=7.07,
        )

    res = export_network_to_sumo(
        G=G,
        output_dir=tmp_path,
        vehicle_speed_scale=0.5,
        run_netconvert=True,
    )
    assert res["net_path"] is not None
    assert res["net_path"].exists()

    val = validate_sumo_network(
        G=G,
        net_path=res["net_path"],
        edge_map=res["edge_map"],
        sample_routes_n=10,
    )

    assert val["sumo_edge_count"] == 4
    assert val["graph_edge_count"] == 4
    assert len(val["missing_edges"]) == 0
    assert val["length_diff_pct"] <= 0.5
    assert val["consecutive_pairs_connected"] == val["consecutive_pairs_checked"]


# ---------------------------------------------------------------------------
# Test 3: SUMO needed - Calibration helper returns discharge within 15%
# ---------------------------------------------------------------------------

@pytest.mark.sumo
@requires_sumo
def test_calibration_discharge_within_tolerance():
    """Verify queue discharge measurement on a single-lane edge returns ~12 veh/min within 15%."""
    # Test at median speed 4.57 m/s with calibrated tau = 2.7969
    # Short measurement window (warmup 20s, window 60s) for rapid unit test
    q = measure_discharge(
        speed_ms=4.57,
        tau=2.7969,
        n_lanes=1,
        n_vehicles=50,
        warmup_s=20.0,
        window_s=60.0,
    )
    # Target: 12 veh/min/lane (+-15% is [10.2, 13.8])
    target = 12.0
    err_pct = abs(q - target) / target * 100.0
    assert err_pct <= 15.0, (
        f"Measured discharge {q:.2f} veh/min/lane deviates by {err_pct:.1f}% "
        f"from target {target:.1f} (allowed <= 15%)"
    )

