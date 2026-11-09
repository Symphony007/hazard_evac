# Dynamic Multi-Agent Evacuation Routing Under Real-Time Hazard Spread

## Overview
Final-year B.Tech CSE project. City-scale OUTDOOR evacuation on a real
road network (OpenStreetMap via OSMnx, Kolkata study area). A hazard
(flood) spreads over time and closes/slows roads. Many evacuees move at
once and roads have capacity.

## Core Research Question
When does classical dynamic routing break down under hazard spread and
congestion, and how close are heuristic and learned methods to the
optimal solution?

## Methods (Solver Ladder)
1. Static Dijkstra (route computed once from initial hazard snapshot)
2. Selfish routing (each evacuee re-routes to own fastest path, no coordination)
3. Time-dependent Dijkstra (edge cost depends on arrival time, uses hazard forecast)
4. CCRP-style coordinated heuristic (capacity-aware, spreads evacuees)
5. Exact time-expanded min-cost flow (OR-Tools), optimal benchmark on small instances
6. RL policy (Stable-Baselines3 + Gymnasium), 8th semester

Later: uncertainty-aware hazard belief, SUMO validation.

## Metrics (fixed names)
- **pct_safe**: % of evacuees that reach an exit before being caught
- **evac_time**: time until last safe evacuee reaches an exit (and mean)
- **path_overhead**: route length / shortest static length
- **max_congestion**: max (vehicles on edge / edge capacity)
- **compute_ms**: compute time per replan

## Architecture
```
data layer (OSMnx graph)
  -> hazard engine
  -> routing core (all methods behind one interface: plan(state) -> routes)
  -> execution layer (fast Python sim; SUMO via TraCI for validation/demo)
  -> evaluation (metrics, confidence intervals, plots)
```

## Tech Stack
- Python 3.11+, osmnx>=2.0, networkx, numpy, pandas, matplotlib, pyyaml, pytest
- Every experiment is seed-controlled (numpy Generator, seed in config)
- All parameters in configs/*.yaml, never hardcoded
- All routing methods behind the same interface in src/routing/base.py

## Team
4-person team, 6-week first phase, goal is a strong mid-semester demo.

## Status and known assumptions (update as we go)
- Done: Task 1 (graph loader), Task 2 (flood hazard simulator).
- Terrain is SYNTHETIC (generated, not a real DEM). A real DEM and a
  flood calibration against a documented Kolkata event are planned as
  "Task 2b"; the ElevationProvider interface in src/hazard/terrain.py
  is the swap point. Never describe the terrain as real.
- Flood spread parameters are assumptions, not calibrated values.
- Flow capacity is nearly uniform (most edges are 1 lane because OSM
  lane tags are sparse); road-type lane defaults are planned for Task 5.
- The flood front (~0.5 m/s at default settings) is much slower than
  vehicles, so static-routing failures depend on departure delay,
  route length, flood speed and (from Task 5) congestion.

