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
- Done: Task 1 (graph loader), Task 2 (flood hazard simulator), Task 3 (single-agent routers & failure demo), Task 4 (metrics & statistical experiment runner), Task 4b (sector-based exit placement), Task 5 (multi-agent point-queue simulation, selfish routing, lane-count defaults, demand sweep), Task 6 (coordinated capacity-reservation routing, CCRP-style, exit throughput bounds), Task 6b (ablation router selfish_forecast, router-aware resume, common-safe evacuation metrics).
- Exit placement: Sector-based boundary exits (`exit_mode: sector`) partition the perimeter ring into `n_exits` angular sectors and select the latest-flooding node safe through `(depart_window_min + exit_safe_margin_min)`. Dropping early-flooded sectors prevents artificial corner-clustering bottlenecks.
- Traffic Model: Discrete event-driven point-queue (Vickrey) simulator. Edges enforce minimum discharge headways `headway_s = 60.0 / flow_capacity` (FIFO discharge) with free-flow traversal delay plus queuing delay. Storage capacity violations are tracked as an edge diagnostic without blocking upstream entries (standard point queue). Trapping in queue checks if the hazard overtakes an edge during queue wait.
- Coordinated Capacity Reservation: ReservationTable discretizes edge flow capacity into discharge slots of width h(e) = 60 / flow_capacity(e) seconds. Earliest-arrival search in reservation mode finds earliest feasible slot before edge flood arrival. Online mode reserves at departure time; offline mode pre-plans all agents ordered by flood arrival time at origin (oracle upper bound).
- Selfish Forecast Ablation Router: Combines dynamic congestion snapshot delay d(e) = traffic.extra_delay_s(u,v,k,t_now) with true time-dependent flood forecast speed degradation and closure checks. Isolates the value of central reservation coordination vs pure information provision.
- Common-Safe Evaluation Metric: Comparing mean evacuation time across routers with unequal survivor sets is confounded. Paired comparisons evaluate mean_evac_min_common_safe and poa_evac_common strictly over evacuees that reach safety under both compared routers.
- Lane Defaults: Per-direction lane count parsing with highway type fallback table (`motorway/trunk: 2`, `primary: 2`, `secondary: 2`, `tertiary/residential/unclassified: 1`). Two-way roads with aggregate OSM lane tags are halved.
- Terrain is SYNTHETIC (generated, not a real DEM). A real DEM and a
  flood calibration against a documented Kolkata event are planned as
  "Task 2b"; the ElevationProvider interface in src/hazard/terrain.py
  is the swap point. Never describe the terrain as real.
- Flood spread parameters are assumptions, not calibrated values.
- The flood front (~0.5 m/s at default settings) is much slower than
  vehicles, so static-routing failures depend on departure delay,
  route length, flood speed and congestion. Under high demand (1000 evacuees),
  time-dependent routing without coordination funnels traffic onto optimal arterial corridors,
  creating severe congestion bottlenecks (queue delay 167s vs 70s for selfish routing).

