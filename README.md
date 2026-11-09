# Dynamic Multi-Agent Evacuation Routing Under Real-Time Hazard Spread

City-scale outdoor evacuation routing on a real road network (Kolkata via
OpenStreetMap/OSMnx). A flood hazard spreads over time, closing and slowing
roads. Multiple evacuees move simultaneously on capacity-constrained edges.

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Build the Road Graph

```bash
python scripts/build_graph.py --config configs/study_area.yaml
```

This downloads the Kolkata (Tollygunge area) drivable road network from
OpenStreetMap, enriches edges with speed, travel time, and capacity
attributes, caches the graph to `data/graph.graphml`, and saves an
overview plot to `outputs/graph_overview.png`.

## Run Tests

```bash
pytest -q
```

## Project Structure

```
configs/          YAML configuration files
data/             Cached graph data (git-ignored)
outputs/          Plots and results (git-ignored)
scripts/          CLI entry-point scripts
src/              Core library code
  config.py       YAML config loader
  graph/          Road-network graph builder & loader
tests/            pytest test suite
```

## Solver Ladder (planned)

1. Static Dijkstra
2. Selfish re-routing
3. Time-dependent Dijkstra
4. CCRP-style coordinated heuristic
5. Exact min-cost flow (OR-Tools)
6. RL policy (Stable-Baselines3)

