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
    SelfishForecastRouter,
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
    elif name == "selfish_forecast":
        return SelfishForecastRouter(replan_interval_s=traffic_cfg.replan_interval_s)
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

    existing_record: dict[str, Any] | None = None
    existing_agents_df: pd.DataFrame | None = None

    if not force and json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_record = json.load(f)
        except Exception:
            existing_record = None

    if not force and csv_path.exists():
        try:
            existing_agents_df = pd.read_csv(csv_path)
        except Exception:
            existing_agents_df = None

    requested_routers = list(cong_cfg.routers)
    existing_metrics = existing_record.get("metrics", {}) if existing_record is not None else {}
    missing_routers = [r for r in requested_routers if r not in existing_metrics]

    if not force and not missing_routers and existing_record is not None and existing_agents_df is not None:
        return existing_record

    routers_to_run = requested_routers if (force or existing_record is None) else missing_routers

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
    if existing_record is not None and "ceiling_pct" in existing_record and existing_record["ceiling_pct"] is not None:
        ceiling_pct = float(existing_record["ceiling_pct"])
    else:
        ceiling_pct = compute_oracle_ceiling(
            G=G,
            hazard=hazard,
            agents=agents,
            exits=exits,
            params=routing_params,
        )

    # 6. Simulate only the missing / requested routers
    new_agent_frames = []
    new_metrics: dict[str, dict[str, Any]] = {}

    for r_name in routers_to_run:
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
        new_agent_frames.append(df_agent)

        r_metrics = run_metrics(
            df_agent,
            edge_stats=pop_res.edge_stats,
            ceiling_pct=ceiling_pct,
        )
        new_metrics[router.name] = r_metrics

    # 7. Merge with existing raw outputs
    if existing_agents_df is not None and not force:
        clean_df = existing_agents_df[~existing_agents_df["router_name"].isin(routers_to_run)]
        combined_agents_df = pd.concat([clean_df] + new_agent_frames, ignore_index=True)
    else:
        combined_agents_df = pd.concat(new_agent_frames, ignore_index=True)

    merged_metrics = dict(existing_metrics) if (existing_record and not force) else {}
    merged_metrics.update(new_metrics)

    if existing_record is not None and not force:
        run_record = dict(existing_record)
        run_record["metrics"] = merged_metrics
        run_record["ceiling_pct"] = float(ceiling_pct)
    else:
        run_record = {
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
            "metrics": merged_metrics,
        }

    combined_agents_df.to_csv(csv_path, index=False)
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
        dry_demand = 2000 if 2000 in demand_levels else (demand_levels[-1] if demand_levels else 2000)
        print("=" * 75)
        print(f"[dry-run] Benchmarking ONLY missing router 'selfish_forecast' at demand={dry_demand}, seed_index=0")
        print("=" * 75)

        raw_dir = Path(cong_cfg.output_dir) / "raw"
        missing_tasks = []
        for d in demand_levels:
            for s in seed_indices:
                jp = raw_dir / f"run_demand{d}_seed{s}.json"
                if not jp.exists():
                    missing_tasks.append((d, s))
                else:
                    try:
                        with open(jp, "r", encoding="utf-8") as f:
                            rec = json.load(f)
                        if "selfish_forecast" not in rec.get("metrics", {}):
                            missing_tasks.append((d, s))
                    except Exception:
                        missing_tasks.append((d, s))

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

        router = instantiate_router("selfish_forecast", traffic_cfg, coord_cfg)
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
        print(f"  Router selfish_forecast   : {dt:.2f} s (pct_safe={pct:.1f}%)")

        missing_count = len(missing_tasks)
        total_tasks = len(demand_levels) * len(seed_indices)
        est_total_seq_s = dt * missing_count
        est_parallel_s = est_total_seq_s / max(1, workers)

        print("-" * 75)
        print(f"Missing runs to fill in: {missing_count} (out of {total_tasks} configured runs)")
        print(f"Estimated time to fill all missing runs ({workers} workers):")
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

