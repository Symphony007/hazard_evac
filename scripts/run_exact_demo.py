#!/usr/bin/env python
"""CLI demo and size benchmark for exact time-expanded min-cost flow.

Usage:
  python scripts/run_exact_demo.py --demand 100 --seed-index 0 --size-only
  python scripts/run_exact_demo.py --demand 100 --seed-index 0
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CoordinatedConfig,
    ExactConfig,
    HazardConfig,
    ScenarioConfig,
    StudyAreaConfig,
    TrafficConfig,
    load_config,
)
from src.exact.time_expanded import build_network, solve  # noqa: E402
from src.experiment.ladder_runner import instantiate_router  # noqa: E402
from src.graph.loader import get_or_build_graph  # noqa: E402
from src.hazard.flood import FloodHazard  # noqa: E402
from src.routing.base import RoutingParams  # noqa: E402
from src.routing.planners import TimeDependentRouter  # noqa: E402
from src.scenario import build_exits, generate_agents  # noqa: E402
from src.sim.multi_agent import simulate_population  # noqa: E402
from src.sim.single_agent import simulate_agent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact time-expanded min-cost flow demo.")
    parser.add_argument("--graph-config", type=str, default="configs/study_area.yaml")
    parser.add_argument("--hazard-config", type=str, default="configs/hazard.yaml")
    parser.add_argument("--scenario-config", type=str, default="configs/scenario.yaml")
    parser.add_argument("--traffic-config", type=str, default="configs/traffic.yaml")
    parser.add_argument("--coordinated-config", type=str, default="configs/coordinated.yaml")
    parser.add_argument("--exact-config", type=str, default="configs/exact.yaml")
    parser.add_argument("--demand", type=int, default=100)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--flood-speed", type=float, default=60.0)
    parser.add_argument("--rounding", type=str, choices=["ceil", "floor", "both"], default="both")
    parser.add_argument("--size-only", action="store_true", help="Print model size and memory without solving")
    args = parser.parse_args()

    graph_cfg = load_config(args.graph_config, StudyAreaConfig)
    hazard_cfg = load_config(args.hazard_config, HazardConfig)
    scenario_cfg = load_config(args.scenario_config, ScenarioConfig)
    traffic_cfg = load_config(args.traffic_config, TrafficConfig)
    coord_cfg = load_config(args.coordinated_config, CoordinatedConfig)
    exact_cfg = load_config(args.exact_config, ExactConfig)

    print("=" * 75)
    print("TASK 7a: TIME-EXPANDED MIN-COST FLOW BENCHMARK")
    print(f"Demand: {args.demand}, Seed Index: {args.seed_index}, Flood Speed: {args.flood_speed} m/min")
    print(f"Discretization dt: {exact_cfg.dt_s}s, Horizon: {exact_cfg.horizon_s}s (K={exact_cfg.horizon_s // exact_cfg.dt_s} steps)")
    print(f"Requested Rounding: {args.rounding}")
    print("=" * 75)

    # 1. Build Road Network Graph
    G = get_or_build_graph(graph_cfg)

    # 2. Setup Flood Hazard (matching Task 5/6 seeds: base_hazard_seed=3000, base_scenario_seed=4000)
    base_hazard_seed = 3000
    base_scenario_seed = 4000
    hazard_seed = base_hazard_seed + args.seed_index
    scenario_seed = base_scenario_seed + args.seed_index

    h_cfg = replace(
        hazard_cfg,
        spread_speed_m_per_min=float(args.flood_speed),
        seed=int(hazard_seed),
        terrain_seed=int(hazard_cfg.terrain_seed),
    )
    hazard = FloodHazard(G, h_cfg)

    # 3. Build Exits
    exits, exit_info = build_exits(
        G=G,
        hazard=hazard,
        scenario_cfg=scenario_cfg,
        center_lat=graph_cfg.center_lat,
        center_lon=graph_cfg.center_lon,
        horizon_min=h_cfg.horizon_min,
    )

    # 4. Generate Agents (matching nested prefix rule from ladder runner)
    max_demand = max(args.demand, 2000)
    rng = np.random.default_rng(scenario_seed)
    full_sample = generate_agents(
        G=G,
        hazard=hazard,
        exits=exits,
        scenario_cfg=replace(scenario_cfg, n_evacuees=max_demand),
        rng=rng,
    )
    agents = full_sample[:args.demand]
    routing_params = RoutingParams(
        vehicle_speed_scale=scenario_cfg.vehicle_speed_scale,
        sf_floor=scenario_cfg.sf_floor,
    )

    modes_to_run = ["ceil", "floor"] if args.rounding == "both" else [args.rounding]

    # 5. Build and inspect sizes
    models = {}
    for mode in modes_to_run:
        mode_cfg = replace(exact_cfg, rounding=mode)
        print(f"\n[model] Building time-expanded network for {len(agents)} agents (rounding={mode})...")
        t0_b = time.perf_counter()
        m = build_network(
            G=G,
            hazard=hazard,
            agents=agents,
            exits=exits,
            params=routing_params,
            exact_cfg=mode_cfg,
            prune=True,
        )
        t_build = time.perf_counter() - t0_b
        est_memory_mb = (m.n_arcs * 64 + m.n_nodes * 64) / (1024 * 1024)

        print("-" * 75)
        print(f"NETWORK SIZE & PRUNING STATISTICS (rounding={mode}):")
        print(f"  Raw space-time nodes (upper bound) : {m.n_nodes_raw:,}")
        print(f"  Raw space-time arcs  (upper bound) : {m.n_arcs_raw:,}")
        print(f"  Pruned nodes (|V_T|)                : {m.n_nodes:,} ({100.0 * m.n_nodes / m.n_nodes_raw:.2f}% retained)")
        print(f"  Pruned arcs  (|A_T|)                : {m.n_arcs:,} ({100.0 * m.n_arcs / m.n_arcs_raw:.2f}% retained)")
        print(f"  Estimated memory                    : ~{est_memory_mb:.1f} MB")
        print(f"  Network construction time           : {t_build:.2f} s")
        print("-" * 75)

        if m.n_arcs > exact_cfg.max_arcs:
            print(f"[error] Model arcs {m.n_arcs} exceed max_arcs {exact_cfg.max_arcs}! Aborting.")
            sys.exit(1)

        models[mode] = (m, mode_cfg)

    if args.size_only:
        print("[size-only] Exiting without solving.")
        return

    # 6. Oracle Ceiling Benchmark (single-agent time_dependent with unlimited capacity)
    print("\n[benchmark] Computing single-agent time_dependent oracle ceiling...")
    td_router = TimeDependentRouter()
    n_oracle_safe = 0
    for a in agents:
        res_oa = simulate_agent(G, hazard, td_router, a, exits, routing_params)
        if res_oa.safe:
            n_oracle_safe += 1
    oracle_ceiling_pct = 100.0 * n_oracle_safe / len(agents)
    print(f"  Oracle Ceiling: {oracle_ceiling_pct:.1f}% ({n_oracle_safe}/{len(agents)} agents)")

    # 7. Evaluate Heuristic Routers in Simulator
    print("\n[heuristics] Running benchmark routers in point-queue simulator...")
    heuristic_routers = ["static", "time_dependent", "selfish", "selfish_forecast", "coordinated_online"]
    heur_results = {}
    best_heur_safe = -1
    best_heur_name = ""

    for r_name in heuristic_routers:
        r_inst = instantiate_router(r_name, traffic_cfg, coord_cfg)
        pop_res = simulate_population(
            G=G,
            hazard=hazard,
            router=r_inst,
            agents=agents,
            exits=exits,
            params=routing_params,
            traffic_cfg=traffic_cfg,
        )
        safe_cnt = sum(1 for r in pop_res.agent_results if r.safe)
        pct = 100.0 * safe_cnt / len(agents)
        heur_results[r_name] = (safe_cnt, pct)
        if safe_cnt > best_heur_safe:
            best_heur_safe = safe_cnt
            best_heur_name = r_name
        print(f"  {r_name:<20}: {safe_cnt:3d}/{len(agents)} safe ({pct:5.1f}%)")

    print(f"  Best Heuristic        : {best_heur_name} ({best_heur_safe}/{len(agents)}, {100.0 * best_heur_safe / len(agents):.1f}%)")

    # 8. Solve Min-Cost Flow for requested modes
    solved_results = {}
    for mode in modes_to_run:
        m, mode_cfg = models[mode]
        print(f"\n[solve] Solving SimpleMinCostFlow with OR-Tools (rounding={mode}, timeout={mode_cfg.max_solve_s}s)...")
        sol = solve(m, exact_cfg=mode_cfg)
        solved_results[mode] = sol
        print(f"  Status                  : {sol.status}")
        print(f"  Solve Time              : {sol.solve_time_s:.2f} s")
        print(f"  Safe Agents             : {sol.n_safe} / {sol.n_agents} ({sol.pct_safe_model:.2f}%)")
        print(f"  Total Travel Steps      : {sol.total_travel_steps:,} steps ({sol.total_travel_steps * exact_cfg.dt_s:.1f} s)")

    # 9. Comparison & Bracket Summary
    print("\n" + "=" * 75)
    print("TASK 7a / 7a2 SUMMARY: EXACT BOUNDS & HEURISTIC BRACKET")
    print("=" * 75)
    print(f"Oracle Ceiling (continuous infinite capacity) : {n_oracle_safe}/{len(agents)} safe ({oracle_ceiling_pct:.1f}%)")
    print(f"Best Heuristic ({best_heur_name})                   : {best_heur_safe}/{len(agents)} safe ({100.0 * best_heur_safe / len(agents):.1f}%)")

    for mode in modes_to_run:
        sol = solved_results[mode]
        mean_t = (sol.total_travel_steps * exact_cfg.dt_s / sol.n_safe) if sol.n_safe > 0 else 0.0
        print(f"Exact Model (rounding={mode:<5})                 : {sol.n_safe}/{sol.n_agents} safe ({sol.pct_safe_model:.1f}%), travel={sol.total_travel_steps * exact_cfg.dt_s:.0f}s (mean {mean_t:.1f}s), solve={sol.solve_time_s:.2f}s")

    if "ceil" in solved_results and "floor" in solved_results:
        n_ceil = solved_results["ceil"].n_safe
        n_floor = solved_results["floor"].n_safe
        t_ceil = solved_results["ceil"].total_travel_steps * exact_cfg.dt_s
        t_floor = solved_results["floor"].total_travel_steps * exact_cfg.dt_s
        bracket_holds = (n_ceil <= best_heur_safe <= n_floor)
        print("-" * 75)
        print(f"Safe Agents Bracket [ceil, floor]             : [{n_ceil}, {n_floor}]")
        print(f"Travel Duration Bracket [floor, ceil]         : [{t_floor:.0f}s, {t_ceil:.0f}s]")
        print(f"Bracket Condition (n_ceil <= best_heur <= n_floor): {bracket_holds} ({n_ceil} <= {best_heur_safe} <= {n_floor})")
    print("=" * 75)


if __name__ == "__main__":
    main()

