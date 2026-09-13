# Layer 1 Progress: Baseline Dynamic Evacuation Routing

This document summarizes the progress and architecture of Layer 1 of the `hazard_evac` project.

## Overview
Layer 1 implements the foundational dynamic evacuation routing system. It models a single-agent scenario where an optimal route is computed on a real city's road network, but the environment is not static. A hazard (such as a flood, earthquake shockwave, or chemical spill) spreads outward from an epicenter over time. As the hazard expands, the system dynamically checks if the current evacuation route intersects the danger zone and adaptively reroutes to avoid it.

## Modules

The core functionality is split into four modules:

1. **`graph.py`**: Handles loading and parsing real-world road networks from OpenStreetMap using the `osmnx` library. To prevent hitting the external API repeatedly and speed up development, it caches the downloaded road networks locally as `.graphml` files.
2. **`hazard.py`**: Simulates the hazard spread. It uses a deterministic, memoryless model based on the Haversine formula (great-circle distance) to expand a hazard radius at a constant rate per time step. Any road network nodes or edges falling within this radius are marked as hazardous.
3. **`router.py`**: Computes the optimal evacuation route using Dijkstra's shortest path algorithm (weighted by road length). When the route enters the hazard zone at any time step, this module intelligently reroutes by penalizing the hazardous edges (setting their traversal cost to infinity) and recalculating the path, ensuring the new route safely avoids the danger.
4. **`visualize.py`**: Renders the road network, hazard zone, and evacuation routes into an interactive Leaflet map using the `folium` library.

## Test Suite

The test suite (`tests/test_layer1.py`) rigorously verifies the behavior of the system end-to-end on a real OSM extract (~1.5 km radius around central Kolkata). It verifies:
- **Graph fetching and caching:** Ensures the graph is properly downloaded and cached.
- **Routing logic:** Validates that Dijkstra's algorithm returns continuous, valid node sequences.
- **Hazard growth:** Confirms the hazard radius expands correctly and newly engulfed nodes are appropriately marked.
- **Rerouting behavior:** Ensures the system correctly identifies when a route is compromised and computes a new path that strictly avoids all hazardous edges.

## Demo (`main.py`)

The standalone demo script ties all modules together into a complete simulation:
- It loads a road network for central Kolkata.
- Selects an origin (evacuation start) and destination (safe zone).
- Places a hazard near the middle of the initial route and simulates its spread over 8 time steps.
- At each step, it logs the hazard's growth, checks the route's safety, and automatically recalculates the route when the hazard intersects it.
- Finally, it renders the complete sequence into an interactive map (`output/evacuation_map.html`), allowing for visual confirmation of the dynamic rerouting behavior.
