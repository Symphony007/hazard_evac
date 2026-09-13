"""
main.py — Layer 1 demo: Dynamic Evacuation Routing on Kolkata's Road Network.

Workflow
--------
1. Fetch / cache a small road-network graph around central Kolkata.
2. Pick two nodes ~1 km apart as origin (evacuation start) and destination
   (safe zone).
3. Compute the initial shortest evacuation route.
4. Place a hazard at a node in the middle of the route.
5. Simulate the hazard spreading over several time steps.
6. At each step, check the route and reroute if it enters the hazard zone.
7. Render a final interactive map and save it as HTML.

Run
---
    .\\venv\\Scripts\\python.exe main.py

Output
------
    output/evacuation_map.html   — open in any browser
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import load_graph
from src.hazard import HazardSimulation
from src.router import compute_route, reroute_if_needed, NoRouteError
from src.visualize import render_map


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Central Kolkata: Esplanade / BBD Bagh area
CENTRE_POINT = (22.5726, 88.3639)
GRAPH_RADIUS_M = 2000          # 2 km radius — gives ~2-5K nodes
NETWORK_TYPE = "drive"

SPREAD_RATE_M = 150            # hazard expands 150 m per time step
TIME_STEPS = 8                 # simulate 8 discrete steps

OUTPUT_DIR = PROJECT_ROOT / "output"
MAP_FILENAME = OUTPUT_DIR / "evacuation_map.html"


# ---------------------------------------------------------------------------
# Main demo logic
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print("  HAZARD_EVAC -- Layer 1 Demo")
    print("  Dynamic Evacuation Routing - Kolkata, West Bengal")
    print("=" * 64)

    # ---- Step 1: Load the road network ----
    print("\n[1/6] Loading road network ...")
    G = load_graph(point=CENTRE_POINT, dist=GRAPH_RADIUS_M, network_type=NETWORK_TYPE)
    print(f"      Graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

    # ---- Step 2: Pick origin and destination ----
    print("\n[2/6] Selecting origin & destination ...")
    nodes = list(G.nodes)
    origin = nodes[0]
    destination = nodes[-1]
    print(f"      Origin node      : {origin}")
    print(f"      Destination node : {destination}")
    print(f"      Origin coords    : ({G.nodes[origin]['y']:.4f}, {G.nodes[origin]['x']:.4f})")
    print(f"      Dest coords      : ({G.nodes[destination]['y']:.4f}, {G.nodes[destination]['x']:.4f})")

    # ---- Step 3: Compute initial route ----
    print("\n[3/6] Computing initial shortest route ...")
    try:
        route = compute_route(G, origin, destination)
    except NoRouteError:
        print("      [FAIL] No route found between the selected nodes.")
        print("         Try a different origin/destination or increase GRAPH_RADIUS_M.")
        return
    print(f"      Initial route: {len(route)} nodes")

    # ---- Step 4: Place hazard at the midpoint of the route ----
    mid_idx = len(route) // 2
    hazard_origin = route[mid_idx]
    print(f"\n[4/6] Placing hazard at node {hazard_origin} "
          f"(route midpoint, index {mid_idx})")
    print(f"      Hazard coords    : ({G.nodes[hazard_origin]['y']:.4f}, "
          f"{G.nodes[hazard_origin]['x']:.4f})")
    print(f"      Spread rate      : {SPREAD_RATE_M} m/step")

    sim = HazardSimulation(G, origin_node=hazard_origin, spread_rate_m=SPREAD_RATE_M)

    # ---- Step 5: Simulate hazard spread + adaptive rerouting ----
    print(f"\n[5/6] Simulating hazard spread over {TIME_STEPS} time steps ...")
    print("-" * 64)

    current_route = route
    final_state = sim.get_hazard_state(0)

    for t in range(1, TIME_STEPS + 1):
        state = sim.get_hazard_state(t)

        try:
            new_route, did_reroute = reroute_if_needed(G, current_route, state)
        except NoRouteError:
            print(f"  t={t:>2}  radius={state.radius_m:>6.0f} m  "
                  f"nodes_affected={len(state.hazardous_nodes):>4}  "
                  f"[BLOCKED] NO SAFE ROUTE -- evacuation blocked!")
            final_state = state
            break

        status = ">> REROUTED" if did_reroute else "-- safe"
        print(f"  t={t:>2}  radius={state.radius_m:>6.0f} m  "
              f"nodes_affected={len(state.hazardous_nodes):>4}  "
              f"route_len={len(new_route):>3}  {status}")

        current_route = new_route
        final_state = state

    print("-" * 64)

    # ---- Step 6: Render map ----
    print(f"\n[6/6] Rendering map -> {MAP_FILENAME}")
    render_map(G, current_route, final_state, filename=str(MAP_FILENAME))
    print(f"      [OK] Map saved! Open in your browser:")
    print(f"         file:///{MAP_FILENAME.as_posix()}")
    print()


if __name__ == "__main__":
    main()
