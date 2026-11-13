"""SUMO road network export, projection, and validation utilities.

Key Modeling Assumptions:
-------------------------
1. No traffic lights: Evacuation under emergency assumes un-signaled priority junctions
   (--tls.discard-loaded true, --tls.guess false).
2. Straight edges: SUMO edges connect node coordinates directly without intermediate
   geometry shapes (straight edges).
3. Priority junctions: Right-of-way priority is determined by OpenStreetMap highway type
   (motorway/trunk > primary > secondary > tertiary > residential/unclassified).
4. Speed limits: Scaled by scenario vehicle_speed_scale (default 0.5) to model urban
   congestion/crawl during disaster evacuation.
5. Turnarounds allowed: Full turnaround capability at junctions (--no-turnarounds false)
   to ensure route feasibility.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    import sumolib


def find_sumo_binary(name: str) -> str:
    """Locate a SUMO binary ('sumo', 'netconvert', etc.) safely.

    Checks PATH via shutil.which first, falling back to SUMO_HOME/bin/<name>.exe.

    Parameters
    ----------
    name : str
        Base name of the executable (without extension).

    Returns
    -------
    str
        Full path to the executable.

    Raises
    ------
    FileNotFoundError
        If binary cannot be located.
    """
    path = shutil.which(name)
    if path:
        return path

    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        for ext in [".exe", ".EXE", ""]:
            candidate = os.path.join(sumo_home, "bin", f"{name}{ext}")
            if os.path.exists(candidate):
                return candidate

    raise FileNotFoundError(
        f"Could not locate SUMO binary '{name}'. Ensure SUMO 1.28.0 is installed "
        f"and SUMO_HOME is set in your environment."
    )


def get_highway_priority(highway: Any) -> int:
    """Return junction priority integer based on OpenStreetMap highway classification.

    Assumption:
    Priority follows: motorway/trunk (4) > primary (3) > secondary (2) > tertiary (1) > others (0).
    """
    if isinstance(highway, list):
        highway = highway[0] if highway else ""
    hw = str(highway).lower()
    if any(k in hw for k in ["motorway", "trunk"]):
        return 4
    elif "primary" in hw:
        return 3
    elif "secondary" in hw:
        return 2
    elif "tertiary" in hw:
        return 1
    return 0


def edge_to_sumo_id(u: int | str, v: int | str, k: int = 0) -> str:
    """Return the canonical SUMO edge identifier for graph edge (u, v, k)."""
    return f"e{u}_{v}_{k}"


def sumo_id_to_edge(sumo_id: str) -> tuple[int, int, int]:
    """Parse graph edge (u, v, k) from canonical SUMO edge identifier."""
    if not sumo_id.startswith("e"):
        raise ValueError(f"Invalid SUMO edge id: {sumo_id}")
    parts = sumo_id[1:].split("_")
    if len(parts) != 3:
        raise ValueError(f"Malformed SUMO edge id: {sumo_id}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def project_graph_nodes(G: nx.MultiDiGraph) -> dict[Any, tuple[float, float]]:
    """Return metric coordinates (x_meters, y_meters) for each node in G.

    If G has a geographic CRS (e.g. EPSG:4326 / lat-lon), projects to UTM in meters.
    If G is already projected or unprojected toy graph in local Cartesian coordinates,
    uses 'x' and 'y' directly.
    """
    crs = G.graph.get("crs")
    # If explicit geographic CRS (e.g. EPSG:4326)
    if crs and ("4326" in str(crs) or "wgs84" in str(crs).lower()):
        import osmnx as ox
        G_proj = ox.projection.project_graph(G)
        return {n: (float(d["x"]), float(d["y"])) for n, d in G_proj.nodes(data=True)}

    # Otherwise coordinates are already in metric/local Cartesian space
    return {n: (float(d.get("x", 0.0)), float(d.get("y", 0.0))) for n, d in G.nodes(data=True)}


def export_network_to_sumo(
    G: nx.MultiDiGraph,
    output_dir: str | Path,
    vehicle_speed_scale: float = 0.5,
    run_netconvert: bool = True,
) -> dict[str, Any]:
    """Export OSMnx road graph to SUMO plain XML files and optionally compile with netconvert.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Input road network graph.
    output_dir : str or Path
        Directory to write XML files, compiled network, and mapping table.
    vehicle_speed_scale : float
        Multiplier on edge speed limits to match urban crawl evacuation assumption.
    run_netconvert : bool
        Whether to invoke netconvert compiler (default: True). Set False for pure XML export.

    Returns
    -------
    dict[str, Any]
        Metadata dict with file paths, counts, and configuration.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nod_path = out_dir / "nodes.nod.xml"
    edg_path = out_dir / "edges.edg.xml"
    net_path = out_dir / "net.net.xml"
    log_path = out_dir / "netconvert.log"
    map_path = out_dir / "edge_map.json"

    # 1. Project node coordinates to metric positions (x, y in meters)
    coords = project_graph_nodes(G)

    # 2. Write nodes.nod.xml
    with open(nod_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">\n')
        for n in G.nodes():
            x, y = coords[n]
            f.write(f'    <node id="{n}" x="{x:.3f}" y="{y:.3f}"/>\n')
        f.write("</nodes>\n")

    # 3. Write edges.edg.xml (one SUMO edge per graph edge, straight edges)
    edge_map_g2s: dict[str, str] = {}
    edge_map_s2g: dict[str, list[int]] = {}

    with open(edg_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">\n')
        for u, v, k, d in G.edges(keys=True, data=True):
            eid = edge_to_sumo_id(u, v, k)
            lanes = int(d.get("lanes_n", d.get("lanes", 1)))
            speed_kph = float(d.get("speed_kph", 30.0))
            # Speed in m/s scaled by vehicle_speed_scale (urban crawl assumption)
            speed_ms = speed_kph * vehicle_speed_scale / 3.6
            length_m = float(d.get("length", 10.0))
            priority = get_highway_priority(d.get("highway", ""))

            f.write(
                f'    <edge id="{eid}" from="{u}" to="{v}" priority="{priority}" '
                f'numLanes="{lanes}" speed="{speed_ms:.3f}" length="{length_m:.3f}"/>\n'
            )
            gkey = f"{u},{v},{k}"
            edge_map_g2s[gkey] = eid
            edge_map_s2g[eid] = [int(u), int(v), int(k)]
        f.write("</edges>\n")

    # 4. Write edge_map.json
    edge_map = {
        "graph_to_sumo": edge_map_g2s,
        "sumo_to_graph": edge_map_s2g,
    }
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(edge_map, f, indent=2)

    # 5. Compile network via netconvert if requested
    if run_netconvert:
        netconvert_bin = find_sumo_binary("netconvert")
        cmd = [
            netconvert_bin,
            "--node-files", str(nod_path.resolve()),
            "--edge-files", str(edg_path.resolve()),
            "-o", str(net_path.resolve()),
            "--no-turnarounds", "false",          # Turnarounds allowed
            "--tls.discard-loaded", "true",        # No traffic lights (priority junctions)
            "--tls.guess", "false",               # Do not guess traffic lights
            "--log", str(log_path.resolve()),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"netconvert failed with returncode {proc.returncode}.\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

    return {
        "nod_path": nod_path,
        "edg_path": edg_path,
        "net_path": net_path if run_netconvert else None,
        "log_path": log_path if run_netconvert else None,
        "map_path": map_path,
        "edge_map": edge_map,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
    }


def validate_sumo_network(
    G: nx.MultiDiGraph,
    net_path: str | Path,
    edge_map: dict[str, Any] | None = None,
    sample_routes_n: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Validate that compiled SUMO network matches the source graph and connections hold.

    Validates:
    1. Edge count matches graph edge count exactly (lists any missing).
    2. Node count matches.
    3. Total road length is within 0.5% of graph length.
    4. Consecutive edges along 200 Dijkstra routes have active connections in SUMO.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Source road network graph.
    net_path : str or Path
        Path to compiled net.net.xml.
    edge_map : dict, optional
        Edge mapping dict (loaded from edge_map.json if None).
    sample_routes_n : int
        Number of random shortest paths to test for consecutive edge connectivity.
    seed : int
        RNG seed for route sampling.

    Returns
    -------
    dict[str, Any]
        Validation statistics and checks.
    """
    import sumolib

    net = sumolib.net.readNet(str(net_path))
    sumo_edges = {e.getID(): e for e in net.getEdges(withInternal=False)}
    sumo_nodes = {n.getID(): n for n in net.getNodes()}

    graph_edge_ids = {edge_to_sumo_id(u, v, k) for u, v, k in G.edges(keys=True)}
    missing_edges = sorted(list(graph_edge_ids - set(sumo_edges.keys())))
    extra_edges = sorted(list(set(sumo_edges.keys()) - graph_edge_ids))

    n_nodes_sumo = len(sumo_nodes)
    n_nodes_graph = G.number_of_nodes()

    # Total length comparison
    sumo_len = sum(e.getLength() for e in sumo_edges.values())
    graph_len = sum(float(d.get("length", 0.0)) for _, _, d in G.edges(data=True))
    length_diff_pct = abs(sumo_len - graph_len) / graph_len * 100.0 if graph_len > 0 else 0.0

    # Connectivity check: sample routes and verify consecutive edge connections
    rng = np.random.default_rng(seed)
    nodes_list = list(G.nodes())
    routes_checked = 0
    consecutive_pairs_checked = 0
    consecutive_pairs_connected = 0
    disconnected_pairs: list[tuple[str, str]] = []

    attempts = 0
    max_attempts = sample_routes_n * 10

    while routes_checked < sample_routes_n and attempts < max_attempts:
        attempts += 1
        s, t = rng.choice(nodes_list, size=2, replace=False)
        try:
            path_nodes = nx.shortest_path(G, s, t, weight="free_flow_time")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        if len(path_nodes) < 2:
            continue

        # Extract sequence of edges
        route_edges: list[tuple[int, int, int]] = []
        for i in range(len(path_nodes) - 1):
            u_n = path_nodes[i]
            v_n = path_nodes[i + 1]
            # Pick edge with minimal free_flow_time
            best_k = min(
                G[u_n][v_n].keys(),
                key=lambda k: float(G[u_n][v_n][k].get("free_flow_time", 9999.0))
            )
            route_edges.append((u_n, v_n, best_k))

        # Check consecutive pairs
        for i in range(len(route_edges) - 1):
            e1 = route_edges[i]
            e2 = route_edges[i + 1]
            seid1 = edge_to_sumo_id(e1[0], e1[1], e1[2])
            seid2 = edge_to_sumo_id(e2[0], e2[1], e2[2])

            consecutive_pairs_checked += 1
            sumo_e1 = sumo_edges.get(seid1)
            sumo_e2 = sumo_edges.get(seid2)

            if sumo_e1 is not None and sumo_e2 is not None:
                # Connected if sumo_e2 is in outgoing of sumo_e1
                if sumo_e2 in sumo_e1.getOutgoing() or len(sumo_e1.getConnections(sumo_e2)) > 0:
                    consecutive_pairs_connected += 1
                else:
                    disconnected_pairs.append((seid1, seid2))
            else:
                disconnected_pairs.append((seid1, seid2))

        routes_checked += 1

    return {
        "sumo_edge_count": len(sumo_edges),
        "graph_edge_count": G.number_of_edges(),
        "missing_edges": missing_edges,
        "extra_edges": extra_edges,
        "sumo_node_count": n_nodes_sumo,
        "graph_node_count": n_nodes_graph,
        "sumo_total_length_m": sumo_len,
        "graph_total_length_m": graph_len,
        "length_diff_pct": length_diff_pct,
        "routes_sampled": routes_checked,
        "consecutive_pairs_checked": consecutive_pairs_checked,
        "consecutive_pairs_connected": consecutive_pairs_connected,
        "disconnected_pairs": disconnected_pairs,
    }
