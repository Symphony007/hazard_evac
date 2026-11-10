"""Multi-agent demand sweep experiment runner with point-queue congestion.

Runs multi-seed demand sweeps evaluating Static, Reactive, TimeDependent,
and Selfish routers under increasing traffic congestion.
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
    HazardConfig,
    ScenarioConfig,
    StudyAreaConfig,
    TrafficConfig,
)
from src.graph.loader import get_or_build_graph
from src.hazard.flood import FloodHazard
from src.metrics.metrics import per_agent_frame, run_metrics
from src.routing.base import RoutingParams
from src.routing.planners import (
    ReactiveRouter,
    SelfishRouter,
    StaticRouter,
    TimeDependentRouter,
)
from src.scenario import build_exits, exit_spread_report, generate_agents, shortest_baseline_length
from src.sim.multi_agent import simulate_population


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


def run_single_congestion(
    graph_cfg: StudyAreaConfig,
    hazard_cfg: HazardConfig,
    scenario_cfg: ScenarioConfig,
    traffic_cfg: TrafficConfig,
    cong_cfg: CongestionConfig,
    demand: int,
    seed_index: int,
    G: nx.MultiDiGraph | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Execute a single multi-agent run across all routers at a given demand level and seed.

    Parameters
    ----------
    graph_cfg : StudyAreaConfig
        Road network config.
    hazard_cfg : HazardConfig
        Hazard parameters.
    scenario_cfg : ScenarioConfig
        Scenario parameters.
    traffic_cfg : TrafficConfig
        Traffic dynamics config.
    cong_cfg : CongestionConfig
        Demand sweep parameters.
    demand : int
        Number of evacuee agents in this run.
    seed_index : int
        Seed index (0..n_seeds-1).
    G : nx.MultiDiGraph, optional
        Preloaded road network graph.
    force : bool
        If True, re-executes even if cached raw outputs exist.

    Returns
    -------
    dict[str, Any]
        Per-run summary containing metrics and diagnostics.
    """
    raw_dir = Path(cong_cfg.output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    token = f"demand{demand}_seed{seed_index}"
    csv_path = raw_dir / f"agents_{token}.csv"
    json_path = raw_dir / f"run_{token}.json"
    edge_csv_path = raw_dir / f"edges_{token}_td.csv"

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
    # Nested sample: the first N agents of the fixed larger sample
    agents = full_sample[:demand]
    departure_window_capped = getattr(full_sample, "departure_window_capped", False)

    # 4. Shortest static baseline length
    baseline_len = shortest_baseline_length(G, exits)

    # 5. Routers
    routers = [
        StaticRouter(),
        ReactiveRouter(),
        TimeDependentRouter(),
        SelfishRouter(replan_interval_s=traffic_cfg.replan_interval_s),
    ]
    routing_params = RoutingParams(
        vehicle_speed_scale=scenario_cfg.vehicle_speed_scale,
        sf_floor=scenario_cfg.sf_floor,
    )

    all_agent_frames = []
    router_metrics: dict[str, dict[str, Any]] = {}
    td_edge_stats = None

    for router in routers:
        pop_res = simulate_population(
            G=G,
            hazard=hazard,
            router=router,
            agents=agents,
            exits=exits,
            params=routing_params,
            traffic_cfg=traffic_cfg,
        )
        if router.name == "time_dependent":
            td_edge_stats = pop_res.edge_stats

        df_agent = per_agent_frame(pop_res.agent_results, baseline_len)
        df_agent["demand"] = int(demand)
        df_agent["seed_index"] = int(seed_index)
        all_agent_frames.append(df_agent)

        r_metrics = run_metrics(df_agent, edge_stats=pop_res.edge_stats)
        router_metrics[router.name] = r_metrics

    # Save per-agent raw CSV
    combined_agents_df = pd.concat(all_agent_frames, ignore_index=True)
    combined_agents_df.to_csv(csv_path, index=False)

    if td_edge_stats is not None:
        td_edge_stats.to_csv(edge_csv_path, index=False)

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
        "exit_spread": spread_report,
        "metrics": router_metrics,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_serializable(run_record), f, indent=2)

    return run_record


def run_congestion_experiment(
    graph_cfg: StudyAreaConfig,
    hazard_cfg: HazardConfig,
    scenario_cfg: ScenarioConfig,
    traffic_cfg: TrafficConfig,
    cong_cfg: CongestionConfig,
    quick: bool = False,
    dry_run: bool = False,
    force: bool = False,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    """Execute multi-seed demand sweep experimental evaluation.

    Parameters
    ----------
    graph_cfg : StudyAreaConfig
        Road network config.
    hazard_cfg : HazardConfig
        Base hazard parameters.
    scenario_cfg : ScenarioConfig
        Scenario parameters.
    traffic_cfg : TrafficConfig
        Traffic dynamics config.
    cong_cfg : CongestionConfig
        Demand sweep config.
    quick : bool
        If True, runs 2 demand levels (300, 1000) on 3 seeds (seeds 0..2).
    dry_run : bool
        If True, benchmarks a single run at demand 1000 and estimates total runtime.
    force : bool
        If True, re-executes even if cached outputs exist.
    workers : int, optional
        Parallel worker count override.
    """
    n_workers = workers if workers is not None else cong_cfg.workers

    if quick:
        demand_levels = [300, 1000]
        seeds = list(range(3))
        print(f"[congestion] QUICK RUN: {len(demand_levels) * len(seeds)} runs (demands={demand_levels}, seeds=0..2, workers={n_workers})")
    elif dry_run:
        demand_levels = [1000]
        seeds = [0]
        print("[congestion] DRY RUN: Benchmarking single run (demand=1000, seed=0)...")
    else:
        demand_levels = list(cong_cfg.demand_levels)
        seeds = list(range(cong_cfg.n_seeds))
        print(f"[congestion] FULL RUN: {len(demand_levels) * len(seeds)} runs (demands={demand_levels}, seeds=0..{cong_cfg.n_seeds - 1}, workers={n_workers})")

    G = get_or_build_graph(graph_cfg)

    tasks = [(d, s) for d in demand_levels for s in seeds]

    if dry_run:
        t0 = time.perf_counter()
        res = run_single_congestion(
            graph_cfg=graph_cfg,
            hazard_cfg=hazard_cfg,
            scenario_cfg=scenario_cfg,
            traffic_cfg=traffic_cfg,
            cong_cfg=cong_cfg,
            demand=demand_levels[0],
            seed_index=seeds[0],
            G=G,
            force=True,
        )
        elapsed = time.perf_counter() - t0
        total_planned_runs = len(cong_cfg.demand_levels) * cong_cfg.n_seeds
        est_serial_s = elapsed * total_planned_runs
        est_parallel_s = est_serial_s / max(1, n_workers)
        print("=" * 70)
        print("DRY RUN BENCHMARK TIMING")
        print("=" * 70)
        print(f"Single run (demand={demand_levels[0]}, 4 routers) : {elapsed:.2f} s")
        print(f"Total planned runs in full sweep    : {total_planned_runs}")
        print(f"Estimated full serial time           : {est_serial_s:.1f} s ({est_serial_s / 60:.1f} min)")
        print(f"Estimated full parallel time ({n_workers} workers): {est_parallel_s:.1f} s ({est_parallel_s / 60:.1f} min)")
        print("=" * 70)
        return [res]

    print(f"[congestion] Launching {len(tasks)} tasks on {n_workers} workers...")
    t0 = time.perf_counter()
    if n_workers == 1:
        records = [
            run_single_congestion(
                graph_cfg=graph_cfg,
                hazard_cfg=hazard_cfg,
                scenario_cfg=scenario_cfg,
                traffic_cfg=traffic_cfg,
                cong_cfg=cong_cfg,
                demand=d,
                seed_index=s,
                G=G,
                force=force,
            )
            for d, s in tasks
        ]
    else:
        records = Parallel(n_jobs=n_workers, verbose=10)(
            delayed(run_single_congestion)(
                graph_cfg=graph_cfg,
                hazard_cfg=hazard_cfg,
                scenario_cfg=scenario_cfg,
                traffic_cfg=traffic_cfg,
                cong_cfg=cong_cfg,
                demand=d,
                seed_index=s,
                G=None,  # Workers load from cache safely
                force=force,
            )
            for d, s in tasks
        )
    total_elapsed = time.perf_counter() - t0
    print(f"[congestion] Completed {len(records)} runs in {total_elapsed:.1f} s ({total_elapsed / 60:.2f} min)")

    return records

