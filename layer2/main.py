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

OUTPUT_DIR = LAYER2_ROOT / "output"
MAP_FILENAME = OUTPUT_DIR / "evacuation_multi_agent.html"

# Reproducible random seed for demo
random.seed(42)


# ---------------------------------------------------------------------------
# Helper: pick OD pairs
# ---------------------------------------------------------------------------

def _pick_od_pairs(graph, num_agents: int, hazard_state):
    """
    Generate origin-destination pairs for evacuees.

    Strategy:
      - Origins are randomly sampled from nodes NOT in the hazard zone
        (people need to evacuate FROM safe areas around the hazard).
      - Destinations are 3 fixed nodes at the graph periphery (safe zones).
      - Each agent is assigned to the nearest safe zone destination.
    """
    import networkx as nx

    nodes = list(graph.nodes)
    safe_nodes = [n for n in nodes if n not in hazard_state.hazardous_nodes]

    if len(safe_nodes) < num_agents:
        # Fall back to all nodes if hazard is too large
        safe_nodes = nodes

    # Pick 3 destination nodes at different quadrants of the graph
    # (use first, middle-ish, and last nodes as proxies for spread)
    dests = [
        safe_nodes[0],
        safe_nodes[len(safe_nodes) // 3],
        safe_nodes[2 * len(safe_nodes) // 3],
    ]

    # Sample origins randomly
    origins = random.sample(safe_nodes, min(num_agents, len(safe_nodes)))

    # Assign each origin to a destination (cycle through the 3)
    od_pairs = []
    for i, origin in enumerate(origins):
        dest = dests[i % len(dests)]
        # Skip if origin == dest
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
    G = load_graph(point=CENTRE_POINT, dist=GRAPH_RADIUS_M, network_type=NETWORK_TYPE)
    print(f"      Graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

    # ---- Step 2: Set up hazard ----
    print(f"\n[2/5] Setting up hazard (t={HAZARD_TIME_STEP}, "
          f"spread={SPREAD_RATE_M} m/step) ...")
    nodes = list(G.nodes)
    hazard_origin = nodes[len(nodes) // 2]
    sim = HazardSimulation(G, origin_node=hazard_origin, spread_rate_m=SPREAD_RATE_M)
    hazard_state = sim.get_hazard_state(HAZARD_TIME_STEP)
    print(f"      Hazard origin  : node {hazard_origin}")
    print(f"      Hazard radius  : {hazard_state.radius_m:.0f} m")
    print(f"      Nodes affected : {len(hazard_state.hazardous_nodes)}")

    # ---- Step 3: Generate OD pairs ----
    print(f"\n[3/5] Generating {NUM_AGENTS} evacuee OD pairs ...")
    od_pairs, dests = _pick_od_pairs(G, NUM_AGENTS, hazard_state)
    print(f"      Origins  : {len(od_pairs)} unique evacuees")
    print(f"      Safe zones: {len(dests)} destinations")

    # ---- Step 4: Route -- naive vs congestion-aware ----
    print(f"\n[4/5] Routing {NUM_AGENTS} agents ...")
    print("-" * 70)

    # Naive
    print("  [A] Naive routing (no congestion awareness) ...")
    naive_result = route_all_agents_naive(G, od_pairs, hazard_state=hazard_state)

    # Congestion-aware
    print("  [B] Congestion-aware routing (BPR feedback) ...")
    congestion_model = CongestionModel(G)
    aware_result = route_all_agents(
        G, od_pairs,
        hazard_state=hazard_state,
        congestion_model=congestion_model,
    )

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
    print()


if __name__ == "__main__":
    main()
