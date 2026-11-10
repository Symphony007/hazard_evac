"""Multi-seed experiment runner with parallel execution and caching/resume support.

Design & Determinism
--------------------
- Run i (i = 0..n_seeds-1) at flood speed s uses:
    hazard_seed = base_hazard_seed + i
    scenario_seed = base_scenario_seed + i
- Terrain seed stays fixed by default (terrain is a static property of the area).
  Optional `vary_terrain` flag enables terrain randomization across seeds if needed.
- Paired comparison: All three routers (Static, Reactive, TimeDependent) evaluate on
  the EXACT SAME flood realization and the EXACT SAME agent sample within each run.
- Determinism: Each (speed, seed_index) task is completely self-contained and
  deterministic, yielding identical results regardless of worker count or execution order.
- Resume capability: Raw CSV and JSON outputs are saved per run. If both exist,
  execution is skipped unless `--force` is specified.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.config import ExperimentConfig, HazardConfig, ScenarioConfig, StudyAreaConfig
from src.graph.loader import get_or_build_graph
from src.hazard.flood import FloodHazard
from src.metrics.metrics import per_agent_frame, run_metrics
from src.routing.base import RoutingParams
from src.routing.planners import ReactiveRouter, StaticRouter, TimeDependentRouter
from src.scenario import build_exits, exit_spread_report, generate_agents, shortest_baseline_length
from src.sim.single_agent import simulate_agent


def _speed_str(speed: float) -> str:
    """Format flood speed into a clean filename token (e.g. 30, 75)."""
    return str(int(speed)) if float(speed).is_integer() else str(speed)


def _json_serializable(val: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val) if np.isfinite(val) else None
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, dict):
        return {k: _json_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_serializable(v) for v in val]
    return val


def run_single(
    graph_cfg: StudyAreaConfig,
    hazard_cfg: HazardConfig,
    scenario_cfg: ScenarioConfig,
    speed: float,
    seed_index: int,
    exp_cfg: ExperimentConfig,
    G: Any = None,
    force: bool = False,
) -> dict[str, Any]:
    """Execute a single paired (speed, seed_index) experiment across all routers.

    Parameters
    ----------
    graph_cfg : StudyAreaConfig
        Study area road network configuration.
    hazard_cfg : HazardConfig
        Base hazard parameters.
    scenario_cfg : ScenarioConfig
        Scenario parameters (evacuee count, departure window, speed scale).
    speed : float
        Flood spread speed in m/min for this run.
    seed_index : int
        Seed index (0..n_seeds-1).
    exp_cfg : ExperimentConfig
        Experiment configuration (seed offsets, output directory).
    G : nx.MultiDiGraph, optional
        Preloaded graph instance; if None, loaded from cache.
    force : bool
        If True, re-runs even if raw result files already exist.

    Returns
    -------
    dict
        Aggregated run record including metrics for each router and diagnostics.
    """
    raw_dir = Path(exp_cfg.output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    token = f"speed{_speed_str(speed)}_seed{seed_index}"
    csv_path = raw_dir / f"agents_{token}.csv"
    json_path = raw_dir / f"run_{token}.json"

    # RESUME support: check if already computed
    if not force and csv_path.exists() and json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            return cached_data
        except Exception:
            pass  # Corrupt file; recompute

    if G is None:
        G = get_or_build_graph(graph_cfg)

    # Paired seeds derivation
    hazard_seed = int(exp_cfg.base_hazard_seed + seed_index)
    scenario_seed = int(exp_cfg.base_scenario_seed + seed_index)
    terrain_seed = (
        int(hazard_cfg.terrain_seed + seed_index)
        if exp_cfg.vary_terrain
        else int(hazard_cfg.terrain_seed)
    )

    # 1. Instantiate hazard
    h_cfg = replace(
        hazard_cfg,
        spread_speed_m_per_min=float(speed),
        seed=hazard_seed,
        terrain_seed=terrain_seed,
    )
    hazard = FloodHazard(G, h_cfg)

    # 2. Select exits & diagnostics
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
    sectors_with_exit_count = len(exit_info.get("sectors_with_exit", []))
    sectors_dropped_count = len(exit_info.get("sectors_dropped", []))

    spread_report = exit_spread_report(
        G=G,
        exits=exits,
        cfg=graph_cfg,
        center_lat=graph_cfg.center_lat,
        center_lon=graph_cfg.center_lon,
    )

    # 3. Sample agents
    rng = np.random.default_rng(scenario_seed)
    agents = generate_agents(
        G=G,
        hazard=hazard,
        exits=exits,
        scenario_cfg=scenario_cfg,
        rng=rng,
    )
    departure_window_capped = getattr(agents, "departure_window_capped", False)

    # 4. Shortest static baseline length to nearest exit
    baseline_len = shortest_baseline_length(G, exits)

    # 5. Simulate each router under identical scenario
    routers = [
        StaticRouter(),
        ReactiveRouter(),
        TimeDependentRouter(),
    ]
    routing_params = RoutingParams(
        vehicle_speed_scale=scenario_cfg.vehicle_speed_scale,
        sf_floor=scenario_cfg.sf_floor,
    )

    all_agent_frames = []
    router_metrics: dict[str, dict[str, Any]] = {}

    for router in routers:
        results = [
            simulate_agent(G, hazard, router, agent, exits, routing_params)
            for agent in agents
        ]
        df_agent = per_agent_frame(results, baseline_len)
        df_agent["flood_speed_m_per_min"] = float(speed)
        df_agent["seed_index"] = int(seed_index)
        all_agent_frames.append(df_agent)

        r_metrics = run_metrics(df_agent)
        router_metrics[router.name] = r_metrics

    # Save per-agent raw CSV
    combined_agents_df = pd.concat(all_agent_frames, ignore_index=True)
    combined_agents_df.to_csv(csv_path, index=False)

    run_record: dict[str, Any] = {
        "speed": float(speed),
        "seed_index": int(seed_index),
        "hazard_seed": hazard_seed,
        "scenario_seed": scenario_seed,
        "terrain_seed": terrain_seed,
        "used_fallback_exits": bool(used_fallback_exits),
        "n_fallback_exits": int(n_fallback_exits),
        "n_exits_effective": int(n_exits_effective),
        "sectors_with_exit_count": int(sectors_with_exit_count),
        "sectors_dropped_count": int(sectors_dropped_count),
        "departure_window_capped": bool(departure_window_capped),
        "n_agents": len(agents),
        "exit_spread": spread_report,
        "metrics": router_metrics,
    }

    # Save run metrics JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_serializable(run_record), f, indent=2)

    return run_record


def run_experiment(
    graph_cfg: StudyAreaConfig,
    hazard_cfg: HazardConfig,
    scenario_cfg: ScenarioConfig,
    exp_cfg: ExperimentConfig,
    quick: bool = False,
    dry_run: bool = False,
    force: bool = False,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    """Run full, quick, or dry-run experimental evaluation across parameter sweep.

    Parameters
    ----------
    graph_cfg : StudyAreaConfig
        Road network config.
    hazard_cfg : HazardConfig
        Hazard simulation config.
    scenario_cfg : ScenarioConfig
        Evacuee scenario config.
    exp_cfg : ExperimentConfig
        Experiment sweep config.
    quick : bool
        If True, runs only 3 seeds on [30, 60, 90] m/min.
    dry_run : bool
        If True, runs 1 run (speed=60, seed=0), prints benchmark timing & ETA, and returns.
    force : bool
        If True, bypasses resume cache and recomputes all runs.
    workers : int, optional
        Number of parallel jobs (overrides exp_cfg.workers).

    Returns
    -------
    list[dict[str, Any]]
        List of all run records.
    """
    n_workers = workers if workers is not None else exp_cfg.workers
    output_dir = Path(exp_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Handle dry-run benchmark
    if dry_run:
        print("[experiment] DRY-RUN: benchmarking 1 run at speed=60 m/min, seed_index=0 ...")
        t0 = time.perf_counter()
        single_res = run_single(
            graph_cfg=graph_cfg,
            hazard_cfg=hazard_cfg,
            scenario_cfg=scenario_cfg,
            speed=60.0,
            seed_index=0,
            exp_cfg=exp_cfg,
            force=force,
        )
        elapsed = time.perf_counter() - t0

        # Calculate full experiment runs count
        total_full_runs = (
            len(exp_cfg.main_speeds_m_per_min) * exp_cfg.n_seeds
            + len(exp_cfg.stress_speeds_m_per_min) * exp_cfg.n_seeds_stress
        )
        # Estimate full duration with n_workers
        est_seconds = (total_full_runs * elapsed) / max(1, n_workers)
        est_min = est_seconds / 60.0

        print(f"[experiment] Dry-run wall-clock time: {elapsed:.2f} s")
        print(f"[experiment] Total runs in full experiment: {total_full_runs} ({len(exp_cfg.main_speeds_m_per_min)} main * {exp_cfg.n_seeds} + {len(exp_cfg.stress_speeds_m_per_min)} stress * {exp_cfg.n_seeds_stress})")
        print(f"[experiment] Estimated full-experiment time with {n_workers} worker(s): {est_min:.1f} min ({est_seconds:.0f} s)")
        return [single_res]

    # 2. Build task list
    if quick:
        speeds = [30.0, 60.0, 90.0]
        tasks = [(s, i, "main") for s in speeds for i in range(3)]
        print(f"[experiment] QUICK RUN: {len(tasks)} runs (speeds={speeds}, seeds=0..2, workers={n_workers})")
    else:
        main_tasks = [
            (s, i, "main")
            for s in exp_cfg.main_speeds_m_per_min
            for i in range(exp_cfg.n_seeds)
        ]
        stress_tasks = [
            (s, i, "stress")
            for s in exp_cfg.stress_speeds_m_per_min
            for i in range(exp_cfg.n_seeds_stress)
        ]
        tasks = main_tasks + stress_tasks
        print(f"[experiment] FULL RUN: {len(tasks)} runs ({len(main_tasks)} main + {len(stress_tasks)} stress, workers={n_workers})")

    # 3. Parallel execution
    # Pre-cache graph so workers can load GraphML cleanly
    get_or_build_graph(graph_cfg)

    t_start = time.perf_counter()

    if n_workers == 1:
        # Sequential execution
        records = []
        for idx, (spd, s_idx, _) in enumerate(tasks, start=1):
            t_run_0 = time.perf_counter()
            rec = run_single(
                graph_cfg=graph_cfg,
                hazard_cfg=hazard_cfg,
                scenario_cfg=scenario_cfg,
                speed=spd,
                seed_index=s_idx,
                exp_cfg=exp_cfg,
                force=force,
            )
            records.append(rec)
            elapsed = time.perf_counter() - t_start
            eta_sec = (elapsed / idx) * (len(tasks) - idx)
            print(f"[run {idx}/{len(tasks)}] speed={spd} seed={s_idx} ({time.perf_counter() - t_run_0:.1f}s) | ETA: {eta_sec/60:.1f} min")
    else:
        # Parallel execution via joblib
        print(f"[experiment] Launching {len(tasks)} tasks on {n_workers} workers...")
        records = Parallel(n_jobs=n_workers, verbose=10)(
            delayed(run_single)(
                graph_cfg,
                hazard_cfg,
                scenario_cfg,
                spd,
                s_idx,
                exp_cfg,
                force=force,
            )
            for (spd, s_idx, _) in tasks
        )

    total_time = time.perf_counter() - t_start
    print(f"[experiment] Completed {len(records)} runs in {total_time:.1f} s ({total_time/60:.2f} min)")

    return records

