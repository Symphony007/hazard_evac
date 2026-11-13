#!/usr/bin/env python
"""SUMO microscopic simulation smoke test (Task 8a).

Replays TimeDependent routes in SUMO without hazard closures (pure physics check):
- Demand: 50 evacuees, seed 0, flood speed 60 m/min exit layout
- Step length: 1.0 s, time-to-teleport: -1, horizon: 3600 s
- Calibrated Krauss car-following vehicle type (vtype.add.xml)

Metrics compared:
- Vehicles inserted vs arrived at exits
- Vehicles never arrived (if any)
- Mean travel time in SUMO vs Point-Queue simulator evac time and their ratio
- Number of vehicles whose insertion was delayed due to road capacity/storage
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import traci

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    HazardConfig,
    ScenarioConfig,
    StudyAreaConfig,
    SumoConfig,
    TrafficConfig,
    load_config,
)
from src.graph.loader import get_or_build_graph
from src.hazard.flood import FloodHazard
from src.routing.base import make_routing_params
from src.routing.planners import TimeDependentRouter
from src.scenario import build_exits, generate_agents
from src.sim.multi_agent import simulate_population
from src.sumo.network import find_sumo_binary


def write_sumo_routes_file(
    rou_path: Path,
    agents: list,
    agent_routes: dict,
    edge_map_g2s: dict[str, str],
) -> list[int]:
    """Write route definitions XML for SUMO and return list of agent IDs written.

    SUMO requires vehicles in route files to be sorted strictly by departure time.
    """
    valid_agent_ids: list[int] = []

    # Sort agents chronologically by departure time, breaking ties by ID
    sorted_agents = sorted(agents, key=lambda a: (float(a.depart_time_s), int(a.id)))

    with open(rou_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("<routes>\n")

        for a in sorted_agents:
            route = agent_routes.get(a.id)
            if route is None or not route.edges:
                # Agent starts at an exit or has no route
                continue

            sumo_edges = []
            for u, v, k in route.edges:
                seid = edge_map_g2s.get(f"{u},{v},{k}")
                if seid is not None:
                    sumo_edges.append(seid)

            if not sumo_edges:
                continue

            edges_str = " ".join(sumo_edges)
            f.write(f'    <route id="r_{a.id}" edges="{edges_str}"/>\n')
            f.write(
                f'    <vehicle id="veh_{a.id}" type="calib_vtype" route="r_{a.id}" '
                f'depart="{a.depart_time_s:.2f}" departSpeed="0"/>\n'
            )
            valid_agent_ids.append(a.id)

        f.write("</routes>\n")

    return valid_agent_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="SUMO microscopic simulation smoke test.")
    parser.add_argument("--graph-config", type=str, default="configs/study_area.yaml")
    parser.add_argument("--hazard-config", type=str, default="configs/hazard.yaml")
    parser.add_argument("--scenario-config", type=str, default="configs/scenario.yaml")
    parser.add_argument("--traffic-config", type=str, default="configs/traffic.yaml")
    parser.add_argument("--sumo-config", type=str, default="configs/sumo.yaml")
    parser.add_argument("--demand", type=int, default=50)
    parser.add_argument("--seed-index", type=int, default=0)
    args = parser.parse_args()

    graph_cfg = load_config(PROJECT_ROOT / args.graph_config, StudyAreaConfig)
    hazard_cfg = load_config(PROJECT_ROOT / args.hazard_config, HazardConfig)
    scenario_cfg = load_config(PROJECT_ROOT / args.scenario_config, ScenarioConfig)
    traffic_cfg = load_config(PROJECT_ROOT / args.traffic_config, TrafficConfig)
    sumo_cfg = load_config(PROJECT_ROOT / args.sumo_config, SumoConfig)

    out_dir = PROJECT_ROOT / sumo_cfg.sumo_dir
    net_file = out_dir / "net.net.xml"
    vtype_file = out_dir / "vtype.add.xml"
    map_file = out_dir / "edge_map.json"
    smoke_rou_file = out_dir / "smoke_routes.rou.xml"

    print("=" * 70)
    print("TASK 8a: SUMO MICROSCOPIC SMOKE TEST (TraCI)")
    print("=" * 70)
    print(f"Demand:            {args.demand} evacuees")
    print(f"Seed Index:        {args.seed_index}")
    print(f"Network file:      {net_file}")
    print(f"Vehicle type:      {vtype_file}")
    print(f"Time-to-teleport:  {sumo_cfg.time_to_teleport}")
    print(f"Step length:       {sumo_cfg.step_length_s} s")
    print("-" * 70)

    # 1. Setup scenario
    G = get_or_build_graph(graph_cfg)
    params = make_routing_params(scenario_cfg)

    h_cfg = replace(
        hazard_cfg,
        spread_speed_m_per_min=60.0,
        seed=3000 + args.seed_index,
        terrain_seed=int(hazard_cfg.terrain_seed),
    )
    hazard = FloodHazard(G, h_cfg)

    exits, _ = build_exits(
        G=G,
        hazard=hazard,
        scenario_cfg=scenario_cfg,
        center_lat=graph_cfg.center_lat,
        center_lon=graph_cfg.center_lon,
        horizon_min=h_cfg.horizon_min,
    )

    rng = np.random.default_rng(4000 + args.seed_index)
    agents = generate_agents(
        G=G,
        hazard=hazard,
        exits=exits,
        scenario_cfg=replace(scenario_cfg, n_evacuees=args.demand),
        rng=rng,
    )
    agents_map = {a.id: a for a in agents}
    print(f"Generated {len(agents)} agents across {len(exits)} perimeter exits.")

    # 2. Plan routes using TimeDependentRouter
    router = TimeDependentRouter()
    agent_routes = {}
    for a in agents:
        rt = router.plan(
            G=G,
            hazard=hazard,
            evacuee=a,
            t_now_s=a.depart_time_s,
            current_node=int(a.origin),
            exits=exits,
            params=params,
        )
        agent_routes[a.id] = rt

    # 3. Simulate with our Point-Queue simulator for baseline comparison
    print("Running Point-Queue simulator baseline...")
    pq_res = simulate_population(
        G=G,
        hazard=hazard,
        router=router,
        agents=agents,
        exits=exits,
        params=params,
        traffic_cfg=traffic_cfg,
    )
    pq_safe_results = {r.agent_id: r for r in pq_res.agent_results if r.safe}
    pq_mean_evac = (
        float(np.mean([r.evac_time_s for r in pq_safe_results.values()]))
        if pq_safe_results else 0.0
    )
    print(f"Point-Queue: {len(pq_safe_results)}/{len(agents)} safe | Mean evac time: {pq_mean_evac:.2f} s ({pq_mean_evac/60.0:.2f} min)")

    # 4. Write SUMO route file
    with open(map_file, "r", encoding="utf-8") as f:
        edge_map_g2s = json.load(f)["graph_to_sumo"]

    valid_agent_ids = write_sumo_routes_file(
        rou_path=smoke_rou_file,
        agents=agents,
        agent_routes=agent_routes,
        edge_map_g2s=edge_map_g2s,
    )
    print(f"Wrote SUMO routes for {len(valid_agent_ids)} mobile agents to {smoke_rou_file}")

    # 5. Run SUMO headless via TraCI
    print("Running SUMO simulation through TraCI...")
    sumo_bin = find_sumo_binary("sumo")
    sumo_cmd = [
        sumo_bin,
        "-n", str(net_file.resolve()),
        "-r", str(smoke_rou_file.resolve()),
        "-a", str(vtype_file.resolve()),
        "--no-step-log", "true",
        "--time-to-teleport", str(sumo_cfg.time_to_teleport),
        "--step-length", str(sumo_cfg.step_length_s),
    ]

    traci.start(sumo_cmd)

    departed_ids: set[str] = set()
    arrived_ids: set[str] = set()
    actual_depart_times: dict[str, float] = {}
    actual_arrival_times: dict[str, float] = {}
    insertion_delayed_count: int = 0

    max_steps = 3600
    for step in range(max_steps):
        traci.simulationStep()
        t_now = traci.simulation.getTime()

        for vid in traci.simulation.getDepartedIDList():
            departed_ids.add(vid)
            actual_depart_times[vid] = t_now
            aid = int(vid.replace("veh_", ""))
            planned_dep = agents_map[aid].depart_time_s
            # If actual depart is delayed past the first discrete step opportunity ceil(planned_dep)
            planned_step = np.ceil(planned_dep)
            if t_now > planned_step + 1.0:
                insertion_delayed_count += 1

        for vid in traci.simulation.getArrivedIDList():
            arrived_ids.add(vid)
            actual_arrival_times[vid] = t_now

        # Stop early if all inserted vehicles have finished
        if len(departed_ids) == len(valid_agent_ids) and traci.simulation.getMinExpectedNumber() == 0:
            break

    traci.close()

    # 6. Metric computation & Comparison
    never_arrived = departed_ids - arrived_ids

    # Travel time per arrived vehicle: arrival - planned_depart
    sumo_evac_times: dict[int, float] = {}
    sumo_travel_times: dict[int, float] = {}
    for vid in arrived_ids:
        aid = int(vid.replace("veh_", ""))
        t_arr = actual_arrival_times[vid]
        t_dep_act = actual_depart_times[vid]
        t_dep_plan = agents_map[aid].depart_time_s
        sumo_evac_times[aid] = t_arr - t_dep_plan
        sumo_travel_times[aid] = t_arr - t_dep_act

    common_agents = sorted(list(set(sumo_evac_times.keys()) & set(pq_safe_results.keys())))
    sumo_common_evac = [sumo_evac_times[aid] for aid in common_agents]
    pq_common_evac = [pq_safe_results[aid].evac_time_s for aid in common_agents]

    sumo_mean_evac = float(np.mean(sumo_common_evac)) if sumo_common_evac else 0.0
    pq_mean_common = float(np.mean(pq_common_evac)) if pq_common_evac else 0.0
    ratio = sumo_mean_evac / pq_mean_common if pq_mean_common > 0 else 1.0

    print("\n" + "=" * 70)
    print("--- SUMO SMOKE TEST RESULTS ---")
    print("=" * 70)
    print(f"Total scenario agents:          {len(agents)}")
    print(f"Mobile agents with routes:      {len(valid_agent_ids)}")
    print(f"Vehicles inserted into SUMO:    {len(departed_ids)} / {len(valid_agent_ids)}")
    print(f"Vehicles arrived at exits:      {len(arrived_ids)} / {len(departed_ids)}")
    if never_arrived:
        print(f"WARNING: Vehicles never arrived ({len(never_arrived)}): {sorted(list(never_arrived))}")
    else:
        print("Vehicles never arrived:         0 (100% arrival rate)")

    print(f"Insertion delayed vehicles:     {insertion_delayed_count} / {len(departed_ids)}")
    print("-" * 70)
    print(f"Common safe agents evaluated:   {len(common_agents)}")
    print(f"Point-Queue mean evac time:     {pq_mean_common:.2f} s ({pq_mean_common/60.0:.2f} min)")
    print(f"SUMO microscopic mean evac time:{sumo_mean_evac:.2f} s ({sumo_mean_evac/60.0:.2f} min)")
    print(f"Ratio (SUMO / Point-Queue):     {ratio:.3f}x")
    print("=" * 70)

    # Acceptance checks
    assert len(departed_ids) == len(valid_agent_ids), "Not all mobile agents departed"
    assert len(never_arrived) == 0, f"{len(never_arrived)} vehicles never arrived"
    assert 0.7 <= ratio <= 1.5, f"Travel time ratio {ratio:.3f}x outside reasonable [0.7, 1.5] bounds"
    print("SUCCESS: SUMO smoke test completed and passed all physics checks.\n")


if __name__ == "__main__":
    main()
