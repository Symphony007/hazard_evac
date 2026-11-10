"""Experiment runner for Task 6: Solver ladder with coordinated CCRP routing.

Evaluates Static, TimeDependent, Selfish, Coordinated (Online), and
Coordinated (Offline) routers under increasing demand levels.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.config import (
    CongestionConfig,
    CoordinatedConfig,
    HazardConfig,
    ScenarioConfig,
    StudyAreaConfig,
    TrafficConfig,
)
from src.graph.loader import get_or_build_graph
from src.hazard.flood import FloodHazard
from src.metrics.metrics import per_agent_frame, run_metrics
from src.routing.base import Evacuee, Router, RoutingParams
from src.routing.coordinated import CoordinatedRouter
from src.routing.planners import (
    ReactiveRouter,
    SelfishRouter,
    StaticRouter,
    TimeDependentRouter,
)
from src.scenario import build_exits, exit_spread_report, generate_agents, shortest_baseline_length
from src.sim.multi_agent import simulate_population
from src.sim.single_agent import simulate_agent


def _json_serializable(obj: Any) -> Any:
    """Recursively coerce numpy / non-standard scalar types to standard Python primitives."""
    if isinstance(obj, dict):
        return {str(k): _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serializable(x) for x in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def instantiate_router(
    name: str,
    traffic_cfg: TrafficConfig,
    coord_cfg: CoordinatedConfig,
) -> Router:
    """Create a router instance by name with configured parameters."""
    if name == "static":
        return StaticRouter()
    elif name == "reactive":
        return ReactiveRouter()
    elif name == "time_dependent":
        return TimeDependentRouter()
    elif name == "selfish":
        return SelfishRouter(replan_interval_s=traffic_cfg.replan_interval_s)
    elif name == "coordinated_online":
        return CoordinatedRouter(
            planning_mode="online",
            flood_safety_margin_s=coord_cfg.flood_safety_margin_s,
            replans=bool(coord_cfg.replan_interval_s is not None),
            replan_interval_s=coord_cfg.replan_interval_s,
        )
    elif name == "coordinated_offline":
        return CoordinatedRouter(
            planning_mode="offline",
            flood_safety_margin_s=coord_cfg.flood_safety_margin_s,
            replans=bool(coord_cfg.replan_interval_s is not None),
            replan_interval_s=coord_cfg.replan_interval_s,
        )
    else:
        raise ValueError(f"Unknown router name: {name}")


def compute_oracle_ceiling(
    G: nx.MultiDiGraph,
    hazard: FloodHazard,
    agents: Sequence[Evacuee],
    exits: set[int] | Sequence[int],
    params: RoutingParams,
) -> float:
    """Compute the single-agent time-dependent oracle ceiling (% safe with unlimited capacity)."""
    if not agents:
        return 0.0
    router = TimeDependentRouter()
    n_safe = 0
    for a in agents:
        res = simulate_agent(
            G=G,
            hazard=hazard,
            router=router,
            evacuee=a,
            exits=exits,
            params=params,
        )
        if res.safe:
            n_safe += 1
    return float(100.0 * n_safe / len(agents))


def run_single_ladder(
    graph_cfg: StudyAreaConfig,
    hazard_cfg: HazardConfig,
    scenario_cfg: ScenarioConfig,
    traffic_cfg: TrafficConfig,
    cong_cfg: CongestionConfig,
    coord_cfg: CoordinatedConfig,
    demand: int,
    seed_index: int,
    G: nx.MultiDiGraph | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Execute a single multi-agent run across all configured ladder routers."""
    raw_dir = Path(cong_cfg.output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    token = f"demand{demand}_seed{seed_index}"
    csv_path = raw_dir / f"agents_{token}.csv"
    json_path = raw_dir / f"run_{token}.json"

    if not force and csv_path.exists() and json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if G is None:
        G = get_or_build_graph(graph_cfg)

    hazard_seed = cong_cfg.base_hazard_seed + seed_index
    scenario_seed = cong_cfg.base_scenario_seed + seed_index

    # 1. Setup hazard
    h_cfg = replace(
        hazard_cfg,
        spread_speed_m_per_min=float(cong_cfg.flood_speed_m_per_min),
        seed=int(hazard_seed),
        terrain_seed=int(hazard_cfg.terrain_seed),
    )
    hazard = FloodHazard(G, h_cfg)

    # 2. Select exits
    exits, exit_info = build_exits(
        G=G,
        hazard=hazard,
        scenario_cfg=scenario_cfg,
        center_lat=graph_cfg.center_lat,
        center_lon=graph_cfg.center_lon,
        horizon_min=h_cfg.horizon_min,
    )
    used_fallback_exits = exit_info.get("used_fallback_exits", False)
    n_fallback_exits = exit_info.get("n_fallback_exits", 0)
    n_exits_effective = exit_info.get("n_exits_effective", len(exits))

    spread_report = exit_spread_report(
        G=G,
        exits=exits,
        cfg=graph_cfg,
        center_lat=graph_cfg.center_lat,
        center_lon=graph_cfg.center_lon,
    )

    # 3. Sample nested agents: sample max_demand agents and take prefix
    max_demand = max(cong_cfg.demand_levels)
    rng = np.random.default_rng(scenario_seed)
    full_sample = generate_agents(
        G=G,
        hazard=hazard,
        exits=exits,
        scenario_cfg=replace(scenario_cfg, n_evacuees=max_demand),
        rng=rng,
    )
    agents = full_sample[:demand]
    departure_window_capped = getattr(full_sample, "departure_window_capped", False)

    # 4. Shortest static baseline length
    baseline_len = shortest_baseline_length(G, exits)
    routing_params = RoutingParams(
        vehicle_speed_scale=scenario_cfg.vehicle_speed_scale,
        sf_floor=scenario_cfg.sf_floor,
    )

    # 5. Compute single-agent oracle ceiling (% safe with infinite capacity)
    ceiling_pct = compute_oracle_ceiling(
        G=G,
        hazard=hazard,
        agents=agents,
        exits=exits,
        params=routing_params,
    )

    # 6. Simulate each router
    all_agent_frames = []
    router_metrics: dict[str, dict[str, Any]] = {}

    for r_name in cong_cfg.routers:
        router = instantiate_router(r_name, traffic_cfg, coord_cfg)
        pop_res = simulate_population(
            G=G,
            hazard=hazard,
            router=router,
            agents=agents,
            exits=exits,
            params=routing_params,
            traffic_cfg=traffic_cfg,
        )

        df_agent = per_agent_frame(pop_res.agent_results, baseline_len)
        df_agent["demand"] = int(demand)
        df_agent["seed_index"] = int(seed_index)
        all_agent_frames.append(df_agent)

        r_metrics = run_metrics(
            df_agent,
            edge_stats=pop_res.edge_stats,
            ceiling_pct=ceiling_pct,
        )
        router_metrics[router.name] = r_metrics

    # Save per-agent raw CSV
    combined_agents_df = pd.concat(all_agent_frames, ignore_index=True)
    combined_agents_df.to_csv(csv_path, index=False)

    run_record: dict[str, Any] = {
        "demand": int(demand),
        "seed_index": int(seed_index),
        "flood_speed_m_per_min": float(cong_cfg.flood_speed_m_per_min),
        "hazard_seed": hazard_seed,
        "scenario_seed": scenario_seed,
        "used_fallback_exits": bool(used_fallback_exits),
        "n_fallback_exits": int(n_fallback_exits),
        "n_exits_effective": int(n_exits_effective),
        "departure_window_capped": bool(departure_window_capped),
        "n_agents": len(agents),
        "ceiling_pct": float(ceiling_pct),
        "exit_spread": spread_report,
        "metrics": router_metrics,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_serializable(run_record), f, indent=2)

    return run_record


def run_ladder_experiment(
    graph_cfg: StudyAreaConfig,
    hazard_cfg: HazardConfig,
    scenario_cfg: ScenarioConfig,
    traffic_cfg: TrafficConfig,
    cong_cfg: CongestionConfig,
    coord_cfg: CoordinatedConfig,
    quick: bool = False,
    dry_run: bool = False,
    force: bool = False,
    workers: int | None = None,
    demand_override: list[int] | None = None,
    seed_override: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Execute the solver ladder experiment across demands and seeds."""
    if workers is None:
        workers = cong_cfg.workers

    if quick:
        demand_levels = [1000, 2000]
        seed_indices = [0, 1, 2]
    else:
        demand_levels = demand_override if demand_override is not None else cong_cfg.demand_levels
        seed_indices = seed_override if seed_override is not None else list(range(cong_cfg.n_seeds))

    # Preload road network graph once
    G = get_or_build_graph(graph_cfg)

    # --- Dry-run timing ---
    if dry_run:
        dry_demand = 3000 if 3000 in cong_cfg.demand_levels else max(cong_cfg.demand_levels)
        print("=" * 75)
        print(f"[dry-run] Benchmarking single run at demand={dry_demand}, seed_index=0")
        print(f"[dry-run] Routers: {cong_cfg.routers}")
        print("=" * 75)

        h_seed = cong_cfg.base_hazard_seed
        s_seed = cong_cfg.base_scenario_seed
        h_cfg = replace(
            hazard_cfg,
            spread_speed_m_per_min=float(cong_cfg.flood_speed_m_per_min),
            seed=int(h_seed),
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
        max_d = max(dry_demand, max(cong_cfg.demand_levels))
        rng = np.random.default_rng(s_seed)
        full_sample = generate_agents(
            G=G,
            hazard=hazard,
            exits=exits,
            scenario_cfg=replace(scenario_cfg, n_evacuees=max_d),
            rng=rng,
        )
        agents = full_sample[:dry_demand]
        routing_params = RoutingParams(
            vehicle_speed_scale=scenario_cfg.vehicle_speed_scale,
            sf_floor=scenario_cfg.sf_floor,
        )

        t0_ceil = time.perf_counter()
        ceil_val = compute_oracle_ceiling(G, hazard, agents, exits, routing_params)
        dt_ceil = time.perf_counter() - t0_ceil
        print(f"  Oracle ceiling ({len(agents)} agents): {dt_ceil:.2f} s (ceiling={ceil_val:.1f}%)")

        router_timings: dict[str, float] = {}
        for r_name in cong_cfg.routers:
            router = instantiate_router(r_name, traffic_cfg, coord_cfg)
            t0 = time.perf_counter()
            pop_res = simulate_population(
                G=G,
                hazard=hazard,
                router=router,
                agents=agents,
                exits=exits,
                params=routing_params,
                traffic_cfg=traffic_cfg,
            )
            dt = time.perf_counter() - t0
            n_safe = sum(1 for r in pop_res.agent_results if r.safe)
            pct = 100.0 * n_safe / len(agents)
            router_timings[r_name] = dt
            print(f"  Router {r_name:<20}: {dt:.2f} s (pct_safe={pct:.1f}%)")

        total_single_run_s = dt_ceil + sum(router_timings.values())
        total_tasks = len(demand_levels) * len(seed_indices)
        est_total_seq_s = total_single_run_s * total_tasks
        est_parallel_s = est_total_seq_s / max(1, workers)

        print("-" * 75)
        print(f"Total time for single run (demand {dry_demand}): {total_single_run_s:.2f} s")
        print(f"Estimated full experiment ({total_tasks} runs across {workers} workers):")
        print(f"  Sequential estimate : {est_total_seq_s:.1f} s ({est_total_seq_s / 60.0:.1f} min)")
        print(f"  Parallel estimate   : {est_parallel_s:.1f} s ({est_parallel_s / 60.0:.1f} min)")
        print("=" * 75)
        return []

    # --- Execution across tasks ---
    tasks = [(d, s) for d in demand_levels for s in seed_indices]
    tag = "QUICK RUN" if quick else "FULL RUN"
    print(f"[ladder] {tag}: {len(tasks)} runs (demands={demand_levels}, seeds={seed_indices}, workers={workers})")

    t_start = time.perf_counter()
    if workers > 1:
        print(f"[ladder] Launching {len(tasks)} tasks on {workers} workers...")
        results = Parallel(n_jobs=workers, verbose=10)(
            delayed(run_single_ladder)(
                graph_cfg=graph_cfg,
                hazard_cfg=hazard_cfg,
                scenario_cfg=scenario_cfg,
                traffic_cfg=traffic_cfg,
                cong_cfg=cong_cfg,
                coord_cfg=coord_cfg,
                demand=d,
                seed_index=s,
                G=G,
                force=force,
            )
            for d, s in tasks
        )
    else:
        results = [
            run_single_ladder(
                graph_cfg=graph_cfg,
                hazard_cfg=hazard_cfg,
                scenario_cfg=scenario_cfg,
                traffic_cfg=traffic_cfg,
                cong_cfg=cong_cfg,
                coord_cfg=coord_cfg,
                demand=d,
                seed_index=s,
                G=G,
                force=force,
            )
            for d, s in tasks
        ]

    elapsed = time.perf_counter() - t_start
    print(f"[ladder] Completed {len(results)} runs in {elapsed:.1f} s ({elapsed / 60.0:.2f} min)")
    return results

