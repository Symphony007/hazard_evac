"""
main.py -- Layer 2 demo: Multi-agent congestion-aware evacuation routing.

Demonstrates ~20 evacuees with varied origins converging on a few safe
destinations in central Kolkata, with a hazard at the centre.  Runs both
naive and congestion-aware routing side by side and prints a comparison
table showing that congestion-aware routing produces better load
distribution.

Run
---
    .\\venv\\Scripts\\python.exe layer2/main.py

Output
------
    layer2/output/evacuation_multi_agent.html
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Ensure layer2/ root is importable
LAYER2_ROOT = Path(__file__).resolve().parent
if str(LAYER2_ROOT) not in sys.path:
    sys.path.insert(0, str(LAYER2_ROOT))

from src.graph import load_graph
from src.hazard import HazardSimulation
from src.congestion import CongestionModel
from src.multi_agent_router import route_all_agents, route_all_agents_naive
from src.visualize import render_multi_agent_map


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CENTRE_POINT = (22.5726, 88.3639)
GRAPH_RADIUS_M = 2000
NETWORK_TYPE = "drive"

NUM_AGENTS = 20
SPREAD_RATE_M = 150
HAZARD_TIME_STEP = 4       # simulate hazard at t=4 (600 m radius)

ORIGIN_SAMPLING_RADIUS_M = 1000  # sample evacuees within 1 km of hazard epicenter

OUTPUT_DIR = LAYER2_ROOT / "output"
MAP_FILENAME = OUTPUT_DIR / "evacuation_multi_agent.html"

# Reproducible random seed for demo
random.seed(42)


# ---------------------------------------------------------------------------
# Helper: pick OD pairs
# ---------------------------------------------------------------------------

import math

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in metres."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _pick_od_pairs(
    graph,
    num_agents: int,
    hazard_state,
    origin_sampling_radius_m: float = 1000.0,
):
    """
    Generate origin-destination pairs for evacuees.

    Strategy:
      - Origins are sampled from nodes within `origin_sampling_radius_m` (1 km)
        of the hazard epicenter that are NOT inside the hazard zone.
        This models a realistic scenario of people near the disaster escaping outward.
      - Destinations are 3 fixed safe-zone nodes at the graph periphery,
        ensuring agents travel outward away from danger.
      - Each agent is assigned to one of the safe zone destinations (cycled).
    """
    nodes = list(graph.nodes)
    safe_nodes = [n for n in nodes if n not in hazard_state.hazardous_nodes]

    if len(safe_nodes) < num_agents:
        safe_nodes = nodes

    # Pick 3 destination nodes at different quadrants of the graph (periphery safe zones)
    dests = [
        safe_nodes[0],
        safe_nodes[len(safe_nodes) // 3],
        safe_nodes[2 * len(safe_nodes) // 3],
    ]

    # Constrain origins to within origin_sampling_radius_m of the hazard epicenter
    h_node = hazard_state.origin_node
    h_lat = graph.nodes[h_node]["y"]
    h_lon = graph.nodes[h_node]["x"]

    near_safe_nodes = [
        n for n in safe_nodes
        if _haversine_m(h_lat, h_lon, graph.nodes[n]["y"], graph.nodes[n]["x"]) <= origin_sampling_radius_m
    ]

    # Fall back to all safe nodes if fewer than num_agents are in the radius
    sample_pool = near_safe_nodes if len(near_safe_nodes) >= num_agents else safe_nodes

    # Sample origins randomly
    origins = random.sample(sample_pool, min(num_agents, len(sample_pool)))

    # Assign each origin to a destination (cycle through the 3)
    od_pairs = []
    for i, origin in enumerate(origins):
        dest = dests[i % len(dests)]
        if origin == dest:
            dest = dests[(i + 1) % len(dests)]
        od_pairs.append((origin, dest))

    return od_pairs, dests


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  HAZARD_EVAC -- Layer 2 Demo")
    print("  Multi-Agent Congestion-Aware Evacuation Routing")
    print("  Kolkata, West Bengal")
    print("=" * 70)

    # ---- Step 1: Load graph ----
    print("\n[1/5] Loading road network ...")
    print(f"      Centre point    : {CENTRE_POINT} (lat, lon)")
    print(f"      Query radius    : {GRAPH_RADIUS_M} m ({GRAPH_RADIUS_M/1000:.1f} km)")
    print(f"      Network type    : {NETWORK_TYPE}")
    G = load_graph(point=CENTRE_POINT, dist=GRAPH_RADIUS_M, network_type=NETWORK_TYPE)
    lats = [G.nodes[n]["y"] for n in G.nodes]
    lons = [G.nodes[n]["x"] for n in G.nodes]
    lat_span_km = (max(lats) - min(lats)) * 111.13
    lon_span_km = (max(lons) - min(lons)) * 102.8
    print(f"      Graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
    print(f"      Bounding box    : lat [{min(lats):.4f}, {max(lats):.4f}] ({lat_span_km:.2f} km N-S), "
          f"lon [{min(lons):.4f}, {max(lons):.4f}] ({lon_span_km:.2f} km E-W)")

    # ---- Step 2: Set up hazard ----
    print(f"\n[2/5] Setting up hazard (t={HAZARD_TIME_STEP}, "
          f"spread={SPREAD_RATE_M} m/step) ...")
    nodes = list(G.nodes)
    hazard_origin = nodes[len(nodes) // 2]
    sim = HazardSimulation(G, origin_node=hazard_origin, spread_rate_m=SPREAD_RATE_M)
    hazard_state = sim.get_hazard_state(HAZARD_TIME_STEP)
    print(f"      Hazard origin  : node {hazard_origin} ({G.nodes[hazard_origin]['y']:.4f}, {G.nodes[hazard_origin]['x']:.4f})")
    print(f"      Hazard radius  : {hazard_state.radius_m:.0f} m")
    print(f"      Nodes affected : {len(hazard_state.hazardous_nodes)}")

    # ---- Step 3: Generate OD pairs ----
    print(f"\n[3/5] Generating {NUM_AGENTS} evacuee OD pairs (origins within {ORIGIN_SAMPLING_RADIUS_M}m of hazard) ...")
    od_pairs, dests = _pick_od_pairs(
        G, NUM_AGENTS, hazard_state,
        origin_sampling_radius_m=ORIGIN_SAMPLING_RADIUS_M,
    )
    print(f"      Origins  : {len(od_pairs)} unique evacuees")
    print(f"      Safe zones: {len(dests)} destinations")
    for i, d in enumerate(dests):
        print(f"        Safe zone {i+1}: node {d} at ({G.nodes[d]['y']:.4f}, {G.nodes[d]['x']:.4f})")
    print("      Evacuee origin coordinates:")
    for i, (orig, dest) in enumerate(od_pairs):
        dest_idx = dests.index(dest) + 1 if dest in dests else "?"
        print(f"        Agent {i:2d}: node {orig} at ({G.nodes[orig]['y']:.4f}, {G.nodes[orig]['x']:.4f}) "
              f"-> Safe zone {dest_idx} (node {dest})")

    # ---- Step 4: Route -- naive vs congestion-aware ----
    print(f"\n[4/5] Routing {NUM_AGENTS} agents ...")
    print("-" * 70)

    # Naive
    print("  [A] Naive routing (no congestion awareness) ...")
    naive_result = route_all_agents_naive(G, od_pairs, hazard_state=hazard_state)
    if naive_result.failed_agents:
        print(f"      [WARN] {len(naive_result.failed_agents)} agent(s) FAILED in naive routing:")
        for f_idx in naive_result.failed_agents:
            orig, dest = od_pairs[f_idx]
            reason = getattr(naive_result, "failure_reasons", {}).get(f_idx, "No path found")
            print(f"             - Agent {f_idx}: origin {orig} -> dest {dest} | Reason: {reason}")
    else:
        print(f"      [OK] All {len(naive_result.agent_routes)}/{len(od_pairs)} agents successfully routed.")

    # Congestion-aware
    print("\n  [B] Congestion-aware routing (BPR feedback) ...")
    congestion_model = CongestionModel(G)
    aware_result = route_all_agents(
        G, od_pairs,
        hazard_state=hazard_state,
        congestion_model=congestion_model,
    )
    if aware_result.failed_agents:
        print(f"      [WARN] {len(aware_result.failed_agents)} agent(s) FAILED in congestion-aware routing:")
        for f_idx in aware_result.failed_agents:
            orig, dest = od_pairs[f_idx]
            reason = getattr(aware_result, "failure_reasons", {}).get(f_idx, "No path found")
            print(f"             - Agent {f_idx}: origin {orig} -> dest {dest} | Reason: {reason}")
    else:
        print(f"      [OK] All {len(aware_result.agent_routes)}/{len(od_pairs)} agents successfully routed.")

    # ---- Comparison table ----
    print("\n" + "=" * 70)
    print("  COMPARISON: Naive vs Congestion-Aware Routing")
    print("=" * 70)
    print(f"  {'Metric':<35} {'Naive':>12} {'Congestion':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    print(f"  {'Agents routed':<35} "
          f"{len(naive_result.agent_routes):>12} "
          f"{len(aware_result.agent_routes):>12}")
    print(f"  {'Failed agents':<35} "
          f"{len(naive_result.failed_agents):>12} "
          f"{len(aware_result.failed_agents):>12}")
    print(f"  {'Unique routes':<35} "
          f"{naive_result.num_unique_routes:>12} "
          f"{aware_result.num_unique_routes:>12}")
    print(f"  {'Max edge load':<35} "
          f"{naive_result.max_edge_load:>12} "
          f"{aware_result.max_edge_load:>12}")
    print(f"  {'Avg path length (m)':<35} "
          f"{naive_result.avg_path_length_m:>12.0f} "
          f"{aware_result.avg_path_length_m:>12.0f}")
    print(f"  {'Loaded edges':<35} "
          f"{naive_result.congestion_stats.get('loaded_edges', 0):>12} "
          f"{aware_result.congestion_stats.get('loaded_edges', 0):>12}")
    print(f"  {'Total edge volume':<35} "
          f"{naive_result.congestion_stats.get('total_volume', 0):>12} "
          f"{aware_result.congestion_stats.get('total_volume', 0):>12}")
    print("=" * 70)

    # Highlight the key win
    if aware_result.max_edge_load < naive_result.max_edge_load:
        improvement = (
            (naive_result.max_edge_load - aware_result.max_edge_load)
            / naive_result.max_edge_load * 100
        )
        print(f"\n  >> Congestion-aware routing reduced max edge load by "
              f"{improvement:.0f}%")
    elif aware_result.max_edge_load == naive_result.max_edge_load:
        print("\n  >> Max edge load is equal (agents may have different ODs)")
    print(f"  >> Route diversity: {aware_result.num_unique_routes} unique "
          f"routes vs {naive_result.num_unique_routes} (naive)")

    # ---- Step 5: Render map ----
    print(f"\n[5/5] Rendering multi-agent map -> {MAP_FILENAME}")
    render_multi_agent_map(
        G,
        aware_result.agent_routes,
        hazard_state,
        congestion_model=congestion_model,
        filename=str(MAP_FILENAME),
    )
    print(f"      [OK] Map saved! Open in your browser:")
    print(f"         file:///{MAP_FILENAME.as_posix()}")
    print(f"      Visualization summary:")
    print(f"        - Routes rendered: {len(aware_result.agent_routes)} agent routes")
    print(f"        - Loaded edges highlighted: {aware_result.congestion_stats.get('loaded_edges', 0)} segments")
    print()


if __name__ == "__main__":
    main()
