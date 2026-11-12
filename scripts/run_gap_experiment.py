"""Optimality gap experiment comparing heuristics against exact time-expanded min-cost flow.

Evaluates single-agent oracle ceiling, exact ceil model, exact floor model,
replayed exact_plan in continuous-time point-queue simulator, and heuristics
(time_dependent, selfish, selfish_forecast, coordinated_online).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CoordinatedConfig,
    ExactConfig,
    GapConfig,
    HazardConfig,
    ScenarioConfig,
    StudyAreaConfig,
    TrafficConfig,
    load_config,
)
from src.exact.decompose import AgentPlan, decompose_flow
from src.exact.time_expanded import ExactModel, build_network, solve
from src.experiment.ladder_runner import compute_oracle_ceiling, instantiate_router
from src.graph.loader import get_or_build_graph
from src.hazard.flood import FloodHazard
from src.metrics.stats import mean_ci, paired_diff_ci
from src.routing.base import Evacuee, RoutingParams, make_routing_params
from src.routing.exact_plan import ExactPlanRouter
from src.scenario import build_exits, generate_agents
from src.sim.multi_agent import simulate_population


def _json_serializable(obj: Any) -> Any:
    """Coerce non-standard types to JSON serializable objects."""
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


def run_single_gap_instance(
    graph_cfg: StudyAreaConfig,
    hazard_cfg: HazardConfig,
    scenario_cfg: ScenarioConfig,
    traffic_cfg: TrafficConfig,
    coord_cfg: CoordinatedConfig,
    exact_cfg: ExactConfig,
    gap_cfg: GapConfig,
    demand: int,
    seed_index: int,
    G: nx.MultiDiGraph | None = None,
    force: bool = False,
    task6_raw_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute evaluation for one (demand, seed_index) instance."""
    raw_dir = Path(gap_cfg.output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    token = f"demand{demand}_seed{seed_index}"
    json_path = raw_dir / f"run_{token}.json"
    csv_path = raw_dir / f"agents_{token}.csv"

    if not force and json_path.exists() and csv_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            return record
        except Exception:
            pass

    if G is None:
        G = get_or_build_graph(graph_cfg)

    params = make_routing_params(scenario_cfg)

    hazard_seed = gap_cfg.base_hazard_seed + seed_index
    scenario_seed = gap_cfg.base_scenario_seed + seed_index

    # 1. Setup hazard
    h_cfg = replace(
        hazard_cfg,
        spread_speed_m_per_min=float(gap_cfg.flood_speed_m_per_min),
        seed=int(hazard_seed),
        terrain_seed=int(hazard_cfg.terrain_seed),
    )
    hazard = FloodHazard(G, h_cfg)

    # 2. Select exits
    exits, _ = build_exits(
        G=G,
        hazard=hazard,
        scenario_cfg=scenario_cfg,
        center_lat=graph_cfg.center_lat,
        center_lon=graph_cfg.center_lon,
        horizon_min=h_cfg.horizon_min,
    )

    # 3. Sample nested agents: sample max_demand agents and take prefix
    max_demand = max(gap_cfg.demand_levels)
    rng = np.random.default_rng(scenario_seed)
    full_sample = generate_agents(
        G=G,
        hazard=hazard,
        exits=exits,
        scenario_cfg=replace(scenario_cfg, n_evacuees=max_demand),
        rng=rng,
    )
    agents = full_sample[:demand]
    n_agents = len(agents)

    # 4. Check for cached task6 results (read-only)
    cached_task6_df: pd.DataFrame | None = None
    if task6_raw_dir is not None:
        task6_csv = task6_raw_dir / f"agents_{token}.csv"
        if task6_csv.exists():
            try:
                cached_task6_df = pd.read_csv(task6_csv)
            except Exception:
                cached_task6_df = None

    agent_records: list[dict[str, Any]] = []
    heuristic_metrics: dict[str, dict[str, Any]] = {}

    # 5. Evaluate heuristics
    for h_name in gap_cfg.heuristics:
        reused = False
        if cached_task6_df is not None and "router_name" in cached_task6_df.columns:
            h_df = cached_task6_df[cached_task6_df["router_name"] == h_name]
            if len(h_df) == n_agents:
                reused = True
                safe_count = int(h_df["safe"].sum())
                pct_safe = float(100.0 * safe_count / n_agents) if n_agents > 0 else 0.0
                heuristic_metrics[h_name] = {
                    "n_safe": safe_count,
                    "pct_safe": pct_safe,
                    "reused_task6": True,
                }
                for _, row in h_df.iterrows():
                    agent_records.append({
                        "agent_id": int(row["agent_id"]),
                        "router_name": h_name,
                        "origin": row.get("origin", ""),
                        "safe": bool(row["safe"]),
                        "trapped_reason": row.get("trapped_reason") if pd.notna(row.get("trapped_reason")) else None,
                        "depart_time_s": float(row["depart_time_s"]) if pd.notna(row.get("depart_time_s")) else 0.0,
                        "finish_time_s": float(row["finish_time_s"]) if pd.notna(row.get("finish_time_s")) else None,
                        "evac_time_s": float(row["evac_time_s"]) if pd.notna(row.get("evac_time_s")) else None,
                        "path_length_m": float(row["path_length_m"]) if pd.notna(row.get("path_length_m")) else 0.0,
                        "demand": demand,
                        "seed_index": seed_index,
                    })

        if not reused:
            router = instantiate_router(h_name, traffic_cfg, coord_cfg)
            pop_res = simulate_population(
                G=G,
                hazard=hazard,
                router=router,
                agents=agents,
                exits=exits,
                params=params,
                traffic_cfg=traffic_cfg,
            )
            safe_count = sum(1 for r in pop_res.agent_results if r.safe)
            pct_safe = float(100.0 * safe_count / n_agents) if n_agents > 0 else 0.0
            heuristic_metrics[h_name] = {
                "n_safe": safe_count,
                "pct_safe": pct_safe,
                "reused_task6": False,
            }
            for r in pop_res.agent_results:
                agent_records.append({
                    "agent_id": r.agent_id,
                    "router_name": h_name,
                    "origin": r.path_nodes[0] if r.path_nodes else "",
                    "safe": r.safe,
                    "trapped_reason": r.trapped_reason,
                    "depart_time_s": r.depart_time_s,
                    "finish_time_s": r.finish_time_s,
                    "evac_time_s": r.evac_time_s,
                    "path_length_m": r.path_length_m,
                    "demand": demand,
                    "seed_index": seed_index,
                })

    # 6. Single-agent Oracle Ceiling
    ceiling_pct = compute_oracle_ceiling(G, hazard, agents, exits, params)
    ceiling_n = int(round(ceiling_pct * n_agents / 100.0))

    # 7. Exact CEIL model (pessimistic / lower bound)
    cfg_ceil = replace(exact_cfg, rounding="ceil")
    t0_b_c = time.perf_counter()
    model_ceil = build_network(G, hazard, agents, exits, params, cfg_ceil, prune=True)
    t_b_ceil = time.perf_counter() - t0_b_c

    t0_s_c = time.perf_counter()
    res_ceil = solve(model_ceil, exact_cfg=cfg_ceil)
    t_s_ceil = time.perf_counter() - t0_s_c

    # Decompose flow into per-agent plans
    plans = decompose_flow(model_ceil, res_ceil, agents)

    # Replay decomposed plans in simulator
    replay_router = ExactPlanRouter(plans=plans)
    replay_res = simulate_population(
        G=G,
        hazard=hazard,
        router=replay_router,
        agents=agents,
        exits=exits,
        params=params,
        traffic_cfg=traffic_cfg,
    )
    exact_plan_realized_n = sum(1 for r in replay_res.agent_results if r.safe)
    exact_plan_realized_pct = float(100.0 * exact_plan_realized_n / n_agents) if n_agents > 0 else 0.0

    for r in replay_res.agent_results:
        agent_records.append({
            "agent_id": r.agent_id,
            "router_name": "exact_plan",
            "origin": r.path_nodes[0] if r.path_nodes else "",
            "safe": r.safe,
            "trapped_reason": r.trapped_reason,
            "depart_time_s": r.depart_time_s,
            "finish_time_s": r.finish_time_s,
            "evac_time_s": r.evac_time_s,
            "path_length_m": r.path_length_m,
            "demand": demand,
            "seed_index": seed_index,
        })

    # 8. Exact FLOOR model (optimistic / upper bound)
    cfg_floor = replace(exact_cfg, rounding="floor")
    t0_b_f = time.perf_counter()
    model_floor = build_network(G, hazard, agents, exits, params, cfg_floor, prune=True)
    t_b_floor = time.perf_counter() - t0_b_f

    t0_s_f = time.perf_counter()
    res_floor = solve(model_floor, exact_cfg=cfg_floor)
    t_s_floor = time.perf_counter() - t0_s_f

    n_safe_floor_model = res_floor.n_safe
    ub_n = min(ceiling_n, n_safe_floor_model)
    ub_pct = float(100.0 * ub_n / n_agents) if n_agents > 0 else 0.0

    # 9. Horizon checks
    horizon_warnings: list[str] = []
    horizon_s = exact_cfg.horizon_s
    for rec in agent_records:
        ft = rec.get("finish_time_s")
        if ft is not None and ft > horizon_s:
            horizon_warnings.append(
                f"Agent {rec['agent_id']} ({rec['router_name']}) finished at {ft:.1f}s > horizon {horizon_s}s"
            )

    # 10. Save raw outputs
    record = {
        "demand": demand,
        "seed_index": seed_index,
        "n_agents": n_agents,
        "ceiling_n": ceiling_n,
        "ceiling_pct": ceiling_pct,
        "n_safe_ceil_model": res_ceil.n_safe,
        "pct_safe_ceil_model": res_ceil.pct_safe_model,
        "n_safe_floor_model": res_floor.n_safe,
        "pct_safe_floor_model": res_floor.pct_safe_model,
        "ub_n": ub_n,
        "ub_pct": ub_pct,
        "exact_plan_realized_n": exact_plan_realized_n,
        "exact_plan_realized_pct": exact_plan_realized_pct,
        "build_time_ceil_s": t_b_ceil,
        "solve_time_ceil_s": t_s_ceil,
        "n_nodes_ceil": model_ceil.n_nodes,
        "n_arcs_ceil": model_ceil.n_arcs,
        "build_time_floor_s": t_b_floor,
        "solve_time_floor_s": t_s_floor,
        "n_nodes_floor": model_floor.n_nodes,
        "n_arcs_floor": model_floor.n_arcs,
        "heuristic_metrics": heuristic_metrics,
        "horizon_warnings": horizon_warnings,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_serializable(record), f, indent=2)

    agents_df = pd.DataFrame(agent_records)
    agents_df.to_csv(csv_path, index=False)

    return record


def run_dry_run_benchmark(
    graph_cfg: StudyAreaConfig,
    hazard_cfg: HazardConfig,
    scenario_cfg: ScenarioConfig,
    traffic_cfg: TrafficConfig,
    coord_cfg: CoordinatedConfig,
    exact_cfg: ExactConfig,
    gap_cfg: GapConfig,
    workers: int,
    demand: int = 2500,
    seed_index: int = 0,
) -> None:
    """Execute a dry-run on one instance and print stage timings and estimates."""
    print(f"\n{'='*70}\nDRY RUN BENCHMARK: Demand {demand}, Seed {seed_index}\n{'='*70}")

    t_start_total = time.perf_counter()
    G = get_or_build_graph(graph_cfg)
    hazard_seed = gap_cfg.base_hazard_seed + seed_index
    scenario_seed = gap_cfg.base_scenario_seed + seed_index

    h_cfg = replace(
        hazard_cfg,
        spread_speed_m_per_min=float(gap_cfg.flood_speed_m_per_min),
        seed=int(hazard_seed),
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

    rng = np.random.default_rng(scenario_seed)
    max_demand = max(max(gap_cfg.demand_levels), demand)
    full_sample = generate_agents(
        G=G,
        hazard=hazard,
        exits=exits,
        scenario_cfg=replace(scenario_cfg, n_evacuees=max_demand),
        rng=rng,
    )
    agents = full_sample[:demand]
    params = make_routing_params(scenario_cfg)

    # 1. Build Ceil
    t0 = time.perf_counter()
    cfg_ceil = replace(exact_cfg, rounding="ceil")
    model_ceil = build_network(G, hazard, agents, exits, params, cfg_ceil, prune=True)
    t_build_ceil = time.perf_counter() - t0
    print(f"1. Build Ceil Network:  {t_build_ceil:6.2f}s | Nodes: {model_ceil.n_nodes:,} | Arcs: {model_ceil.n_arcs:,}")

    # 2. Solve Ceil
    t0 = time.perf_counter()
    res_ceil = solve(model_ceil, exact_cfg=cfg_ceil)
    t_solve_ceil = time.perf_counter() - t0
    print(f"2. Solve Ceil Model:    {t_solve_ceil:6.2f}s | Safe: {res_ceil.n_safe}/{len(agents)} ({res_ceil.pct_safe_model:.1f}%)")

    # 3. Decompose Ceil Flow
    t0 = time.perf_counter()
    plans = decompose_flow(model_ceil, res_ceil, agents)
    t_decomp = time.perf_counter() - t0
    print(f"3. Decompose Flow:      {t_decomp:6.2f}s | Decomposed {len([p for p in plans.values() if p is not None])} safe plans")

    # 4. Replay in Simulator
    t0 = time.perf_counter()
    router = ExactPlanRouter(plans=plans)
    pop_res = simulate_population(G, hazard, router, agents, exits, params, traffic_cfg)
    t_replay = time.perf_counter() - t0
    replayed_safe = sum(1 for r in pop_res.agent_results if r.safe)
    print(f"4. Replay exact_plan:   {t_replay:6.2f}s | Realized safe: {replayed_safe}/{len(agents)} ({100.0*replayed_safe/len(agents):.1f}%)")

    # 5. Build & Solve Floor
    t0 = time.perf_counter()
    cfg_floor = replace(exact_cfg, rounding="floor")
    model_floor = build_network(G, hazard, agents, exits, params, cfg_floor, prune=True)
    t_build_floor = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_floor = solve(model_floor, exact_cfg=cfg_floor)
    t_solve_floor = time.perf_counter() - t0
    print(f"5. Build & Solve Floor: {t_build_floor + t_solve_floor:6.2f}s (Build: {t_build_floor:.2f}s, Solve: {t_solve_floor:.2f}s) | Safe: {res_floor.n_safe}")

    # 6. Heuristics
    t0 = time.perf_counter()
    for h_name in gap_cfg.heuristics:
        h_router = instantiate_router(h_name, traffic_cfg, coord_cfg)
        simulate_population(G, hazard, h_router, agents, exits, params, traffic_cfg)
    t_heuristics = time.perf_counter() - t0
    print(f"6. Heuristics ({len(gap_cfg.heuristics)}):   {t_heuristics:6.2f}s")

    t_instance_total = time.perf_counter() - t_start_total

    # Peak memory
    mem_info = "N/A"
    try:
        import psutil
        process = psutil.Process()
        mem_mb = process.memory_info().rss / (1024 * 1024)
        mem_info = f"{mem_mb:.1f} MB"
    except Exception:
        pass

    print(f"\nPeak Resident Memory:   {mem_info}")
    print(f"Total Instance Time:    {t_instance_total:.2f}s ({t_instance_total/60.0:.2f} min)")

    total_runs = len(gap_cfg.demand_levels) * gap_cfg.n_seeds
    estimated_total_s = (t_instance_total * total_runs) / max(1, workers)
    print(f"\nFull-Run Estimate ({total_runs} instances with {workers} workers):")
    print(f"  Approx. {estimated_total_s:.1f}s ({estimated_total_s / 60.0:.1f} min / {estimated_total_s / 3600.0:.2f} hours)")
    print(f"{'='*70}\n")


def aggregate_gap_results(
    output_dir: Path,
    demand_levels: list[int],
    heuristics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Aggregate per-run JSON and CSV results into summary.csv and paired_gaps.csv."""
    raw_dir = output_dir / "raw"
    records: list[dict[str, Any]] = []
    agent_dfs: list[pd.DataFrame] = []

    for j_file in sorted(raw_dir.glob("run_demand*_seed*.json")):
        try:
            with open(j_file, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception:
            pass

    for c_file in sorted(raw_dir.glob("agents_demand*_seed*.csv")):
        try:
            agent_dfs.append(pd.read_csv(c_file))
        except Exception:
            pass

    if not records or not agent_dfs:
        raise ValueError(f"No run records found in {raw_dir}")

    all_agents_df = pd.concat(agent_dfs, ignore_index=True)
    run_df = pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # Summary Table
    # -----------------------------------------------------------------------
    summary_rows: list[dict[str, Any]] = []
    horizon_warnings: list[str] = []

    for r in records:
        for w in r.get("horizon_warnings", []):
            if w not in horizon_warnings:
                horizon_warnings.append(w)

    for d in demand_levels:
        sub_run = run_df[run_df["demand"] == d]
        if sub_run.empty:
            continue

        n_seeds_agg = len(sub_run)
        row: dict[str, Any] = {
            "demand": d,
            "n_runs": n_seeds_agg,
        }

        # Model & Replay Bounds
        m_ceil_ub, ci_l, ci_h = mean_ci(sub_run["ceiling_pct"].values)
        row["ceiling_mean"] = m_ceil_ub
        row["ceiling_ci_low"] = ci_l
        row["ceiling_ci_high"] = ci_h

        m_ub, ci_l, ci_h = mean_ci(sub_run["ub_pct"].values)
        row["ub_mean"] = m_ub
        row["ub_ci_low"] = ci_l
        row["ub_ci_high"] = ci_h

        m_floor, ci_l, ci_h = mean_ci(sub_run["pct_safe_floor_model"].values)
        row["floor_model_mean"] = m_floor
        row["floor_model_ci_low"] = ci_l
        row["floor_model_ci_high"] = ci_h

        m_ceil, ci_l, ci_h = mean_ci(sub_run["pct_safe_ceil_model"].values)
        row["ceil_model_mean"] = m_ceil
        row["ceil_model_ci_low"] = ci_l
        row["ceil_model_ci_high"] = ci_h

        m_real, ci_l, ci_h = mean_ci(sub_run["exact_plan_realized_pct"].values)
        row["exact_plan_realized_mean"] = m_real
        row["exact_plan_realized_ci_low"] = ci_l
        row["exact_plan_realized_ci_high"] = ci_h

        # Heuristic survival & Gaps
        for h_name in heuristics:
            h_pcts: list[float] = []
            gap_ubs: list[float] = []
            gap_lbs: list[float] = []

            for _, rec in sub_run.iterrows():
                hm = rec.get("heuristic_metrics", {}).get(h_name, {})
                pct = hm.get("pct_safe", np.nan)
                h_pcts.append(pct)
                ub = rec.get("ub_pct", np.nan)
                lb = rec.get("exact_plan_realized_pct", np.nan)
                gap_ubs.append(ub - pct)
                gap_lbs.append(lb - pct)

            m_h, l_h, h_h = mean_ci(h_pcts)
            row[f"{h_name}_pct_mean"] = m_h
            row[f"{h_name}_pct_ci_low"] = l_h
            row[f"{h_name}_pct_ci_high"] = h_h

            m_gub, l_gub, h_gub = mean_ci(gap_ubs)
            row[f"{h_name}_gap_ub_mean"] = m_gub
            row[f"{h_name}_gap_ub_ci_low"] = l_gub
            row[f"{h_name}_gap_ub_ci_high"] = h_gub

            m_glb, l_glb, h_glb = mean_ci(gap_lbs)
            row[f"{h_name}_gap_lb_mean"] = m_glb
            row[f"{h_name}_gap_lb_ci_low"] = l_glb
            row[f"{h_name}_gap_lb_ci_high"] = h_glb

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # -----------------------------------------------------------------------
    # Paired Differences on Common-Safe Evacuees
    # -----------------------------------------------------------------------
    paired_rows: list[dict[str, Any]] = []

    for d in demand_levels:
        d_agents = all_agents_df[all_agents_df["demand"] == d]
        if d_agents.empty:
            continue

        exact_agents = d_agents[d_agents["router_name"] == "exact_plan"]
        exact_safe_map = {
            (r["seed_index"], r["agent_id"]): r["evac_time_s"]
            for _, r in exact_agents[exact_agents["safe"] == True].iterrows()
        }

        for h_name in heuristics:
            h_agents = d_agents[d_agents["router_name"] == h_name]
            h_safe_map = {
                (r["seed_index"], r["agent_id"]): r["evac_time_s"]
                for _, r in h_agents[h_agents["safe"] == True].iterrows()
            }

            # Intersection: safe under both
            common_keys = sorted(list(set(exact_safe_map.keys()) & set(h_safe_map.keys())))
            n_common = len(common_keys)

            if n_common > 0:
                h_times_s = np.array([h_safe_map[k] for k in common_keys], dtype=np.float64)
                ex_times_s = np.array([exact_safe_map[k] for k in common_keys], dtype=np.float64)

                # Convert to minutes for paired difference analysis
                diff_res = paired_diff_ci(h_times_s / 60.0, ex_times_s / 60.0)
                mean_ratio = float(np.mean(h_times_s) / np.mean(ex_times_s)) if np.mean(ex_times_s) > 0 else np.nan

                paired_rows.append({
                    "demand": d,
                    "heuristic": h_name,
                    "n_common_safe": n_common,
                    "mean_h_evac_s": float(np.mean(h_times_s)),
                    "mean_exact_evac_s": float(np.mean(ex_times_s)),
                    "diff_mean_min": diff_res.mean_diff,
                    "diff_ci_low_min": diff_res.ci_low,
                    "diff_ci_high_min": diff_res.ci_high,
                    "ratio_h_to_exact": mean_ratio,
                    "p_ttest": diff_res.p_ttest,
                    "p_wilcoxon": diff_res.p_wilcoxon,
                })
            else:
                paired_rows.append({
                    "demand": d,
                    "heuristic": h_name,
                    "n_common_safe": 0,
                    "mean_h_evac_s": np.nan,
                    "mean_exact_evac_s": np.nan,
                    "diff_mean_min": np.nan,
                    "diff_ci_low_min": np.nan,
                    "diff_ci_high_min": np.nan,
                    "ratio_h_to_exact": np.nan,
                    "p_ttest": np.nan,
                    "p_wilcoxon": np.nan,
                })

    paired_df = pd.DataFrame(paired_rows)

    return summary_df, paired_df, horizon_warnings


def generate_gap_figures(
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    heuristics: list[str],
    output_dir: Path,
) -> None:
    """Generate gap_survival.png and gap_evac_time.png."""
    demands = summary_df["demand"].values

    # 1. Figure: gap_survival.png
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Shaded band [ceil_model, ub]
    ceil_vals = summary_df["ceil_model_mean"].values
    ub_vals = summary_df["ub_mean"].values
    ax.fill_between(
        demands,
        ceil_vals,
        ub_vals,
        color="gray",
        alpha=0.25,
        label="Exact Model Bracket [Ceil, UB]",
    )

    # Dashed line for ceiling
    ax.plot(
        demands,
        summary_df["ceiling_mean"].values,
        linestyle="--",
        color="black",
        linewidth=1.8,
        label="Oracle Ceiling (Infinite Cap)",
    )

    # Exact plan realized
    ax.plot(
        demands,
        summary_df["exact_plan_realized_mean"].values,
        marker="o",
        color="darkblue",
        linewidth=2.2,
        label="Exact Plan Replayed (Achievable LB)",
    )

    # Heuristics
    colors = {"time_dependent": "#1f77b4", "selfish": "#ff7f0e", "selfish_forecast": "#2ca02c", "coordinated_online": "#d62728"}
    for h_name in heuristics:
        col = f"{h_name}_pct_mean"
        if col in summary_df.columns:
            ax.plot(
                demands,
                summary_df[col].values,
                marker="s",
                linestyle="-.",
                linewidth=1.5,
                color=colors.get(h_name, None),
                label=f"{h_name}",
            )

    ax.set_xlabel("Evacuation Demand (Vehicles)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Survival Rate (% Safe)", fontsize=11, fontweight="bold")
    ax.set_title("Optimality Gap: Survival Rates vs Demand", fontsize=13, fontweight="bold")
    ax.set_xticks(demands)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()

    fig_survival_path = output_dir / "gap_survival.png"
    plt.savefig(fig_survival_path, dpi=150, facecolor="white")
    plt.close()

    # 2. Figure: gap_evac_time.png
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # For each router on common safe set
    # Exact plan baseline mean evac time (min)
    if not paired_df.empty:
        # Group by demand
        for h_name in heuristics:
            sub = paired_df[paired_df["heuristic"] == h_name]
            if not sub.empty:
                dem = sub["demand"].values
                diff_m = sub["diff_mean_min"].values
                ci_l = sub["diff_ci_low_min"].values
                ci_h = sub["diff_ci_high_min"].values
                err_low = diff_m - ci_l
                err_high = ci_h - diff_m
                ax.errorbar(
                    dem,
                    diff_m,
                    yerr=[err_low, err_high],
                    marker="o",
                    capsize=4,
                    linewidth=1.8,
                    label=f"{h_name} - exact_plan (min)",
                    color=colors.get(h_name, None),
                )

        ax.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xlabel("Evacuation Demand (Vehicles)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Paired Evacuation Time Difference (Minutes)", fontsize=11, fontweight="bold")
        ax.set_title("Evacuation Delay Relative to Exact Plan (Common-Safe Evacuees)", fontsize=13, fontweight="bold")
        ax.set_xticks(demands)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", framealpha=0.9)
        plt.tight_layout()

        fig_evac_path = output_dir / "gap_evac_time.png"
        plt.savefig(fig_evac_path, dpi=150, facecolor="white")
        plt.close()


def print_formatted_tables(
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    heuristics: list[str],
    horizon_warnings: list[str],
) -> None:
    """Print readable summary tables to terminal."""
    print("\n" + "=" * 90)
    print("TASK 7b: OPTIMALITY GAP EXPERIMENT SUMMARY")
    print("=" * 90)

    # 1. Survival Table
    print("\nSURVIVAL RATES (% SAFE) OVER SEEDS (MEAN [95% CI]):")
    print("-" * 90)
    hdr = f"{'Demand':>7} | {'Runs':>4} | {'Ceiling':>14} | {'UB':>14} | {'Floor Model':>14} | {'Ceil Model':>14} | {'Exact Replay':>14}"
    print(hdr)
    print("-" * 90)
    for _, row in summary_df.iterrows():
        d = int(row["demand"])
        runs = int(row["n_runs"])
        c_str = f"{row['ceiling_mean']:.1f}%"
        ub_str = f"{row['ub_mean']:.1f}%"
        f_str = f"{row['floor_model_mean']:.1f}%"
        ce_str = f"{row['ceil_model_mean']:.1f}%"
        ex_str = f"{row['exact_plan_realized_mean']:.1f}%"
        print(f"{d:7d} | {runs:4d} | {c_str:>14} | {ub_str:>14} | {f_str:>14} | {ce_str:>14} | {ex_str:>14}")

    # 2. Heuristic Survival & Gaps
    print("\nHEURISTIC SURVIVAL RATES AND GAPS (GAP_UB = UB - H, GAP_LB = REPLAY - H):")
    print("-" * 90)
    hdr_h = f"{'Demand':>7} | {'Heuristic':<20} | {'Pct Safe (Mean [CI])':<24} | {'Gap UB (pp)':<16} | {'Gap LB (pp)':<16}"
    print(hdr_h)
    print("-" * 90)
    for _, row in summary_df.iterrows():
        d = int(row["demand"])
        for h in heuristics:
            m_pct = row.get(f"{h}_pct_mean", np.nan)
            l_pct = row.get(f"{h}_pct_ci_low", np.nan)
            h_pct = row.get(f"{h}_pct_ci_high", np.nan)
            pct_str = f"{m_pct:.2f} [{l_pct:.2f}, {h_pct:.2f}]"

            m_gub = row.get(f"{h}_gap_ub_mean", np.nan)
            m_glb = row.get(f"{h}_gap_lb_mean", np.nan)
            print(f"{d:7d} | {h:<20} | {pct_str:<24} | {m_gub:+7.2f} pp        | {m_glb:+7.2f} pp")
        print("-" * 90)

    # 3. Paired Evacuation Differences
    print("\nCOMMON-SAFE EVACUATION TIME PAIRED DIFFERENCES (HEURISTIC - EXACT_PLAN):")
    print("-" * 90)
    hdr_p = f"{'Demand':>7} | {'Heuristic':<20} | {'N Safe':>7} | {'Diff Min [95% CI]':<25} | {'Ratio':>6} | {'p (t-test)':>10}"
    print(hdr_p)
    print("-" * 90)
    for _, row in paired_df.iterrows():
        d = int(row["demand"])
        h = row["heuristic"]
        n = int(row["n_common_safe"])
        diff_m = row["diff_mean_min"]
        ci_l = row["diff_ci_low_min"]
        ci_h = row["diff_ci_high_min"]
        diff_str = f"{diff_m:+.2f} [{ci_l:+.2f}, {ci_h:+.2f}]" if np.isfinite(diff_m) else "N/A"
        ratio_str = f"{row['ratio_h_to_exact']:.2f}x" if np.isfinite(row["ratio_h_to_exact"]) else "N/A"
        p_str = f"{row['p_ttest']:.4e}" if np.isfinite(row["p_ttest"]) else "N/A"
        print(f"{d:7d} | {h:<20} | {n:7d} | {diff_str:<25} | {ratio_str:>6} | {p_str:>10}")
    print("=" * 90)

    # 4. Warnings
    if horizon_warnings:
        print(f"\n[HORIZON WARNINGS] Found {len(horizon_warnings)} finish times exceeding horizon:")
        for w in horizon_warnings[:10]:
            print(f"  - {w}")
        if len(horizon_warnings) > 10:
            print(f"  ... and {len(horizon_warnings) - 10} more.")
    else:
        print("\n[HORIZON WARNINGS] None: all safe agents finished well within horizon_s.")
    print("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 7b: Optimality Gap Experiment.")
    parser.add_argument("--graph-config", type=str, default="configs/study_area.yaml")
    parser.add_argument("--hazard-config", type=str, default="configs/hazard.yaml")
    parser.add_argument("--scenario-config", type=str, default="configs/scenario.yaml")
    parser.add_argument("--traffic-config", type=str, default="configs/traffic.yaml")
    parser.add_argument("--coordinated-config", type=str, default="configs/coordinated.yaml")
    parser.add_argument("--exact-config", type=str, default="configs/exact_gap.yaml")
    parser.add_argument("--gap-config", type=str, default="configs/gap.yaml")
    parser.add_argument("--quick", action="store_true", help="Quick run: demand 2000, seeds 0-1.")
    parser.add_argument("--dry-run", action="store_true", help="Dry run: benchmark demand and seed.")
    parser.add_argument("--dry-run-demand", type=int, default=2500, help="Demand level for dry-run benchmark.")
    parser.add_argument("--dry-run-seed", type=int, default=0, help="Seed index for dry-run benchmark.")
    parser.add_argument("--force", action="store_true", help="Force recomputation.")
    parser.add_argument("--workers", type=int, default=None, help="Worker count override.")
    parser.add_argument("--analyze-only", action="store_true", help="Skip runs; analyze existing raw outputs.")
    args = parser.parse_args()

    # Load configs
    graph_cfg = load_config(PROJECT_ROOT / args.graph_config, StudyAreaConfig)
    hazard_cfg = load_config(PROJECT_ROOT / args.hazard_config, HazardConfig)
    scenario_cfg = load_config(PROJECT_ROOT / args.scenario_config, ScenarioConfig)
    traffic_cfg = load_config(PROJECT_ROOT / args.traffic_config, TrafficConfig)
    coord_cfg = load_config(PROJECT_ROOT / args.coordinated_config, CoordinatedConfig)
    exact_cfg = load_config(PROJECT_ROOT / args.exact_config, ExactConfig)
    gap_cfg = load_config(PROJECT_ROOT / args.gap_config, GapConfig)

    workers = args.workers if args.workers is not None else gap_cfg.workers

    if args.quick:
        gap_cfg = replace(gap_cfg, demand_levels=[2000], n_seeds=2)

    output_dir = PROJECT_ROOT / gap_cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    task6_raw_dir = PROJECT_ROOT / "outputs" / "task6" / "raw"

    # Dry-run execution
    if args.dry_run:
        run_dry_run_benchmark(
            graph_cfg=graph_cfg,
            hazard_cfg=hazard_cfg,
            scenario_cfg=scenario_cfg,
            traffic_cfg=traffic_cfg,
            coord_cfg=coord_cfg,
            exact_cfg=exact_cfg,
            gap_cfg=gap_cfg,
            workers=workers,
            demand=args.dry_run_demand,
            seed_index=args.dry_run_seed,
        )
        return

    # Simulation execution
    if not args.analyze_only:
        tasks: list[tuple[int, int]] = []
        for d in gap_cfg.demand_levels:
            for s in range(gap_cfg.n_seeds):
                tasks.append((d, s))

        print(f"Executing {len(tasks)} gap experiment runs with {workers} workers...")
        if workers > 1:
            Parallel(n_jobs=workers)(
                delayed(run_single_gap_instance)(
                    graph_cfg=graph_cfg,
                    hazard_cfg=hazard_cfg,
                    scenario_cfg=scenario_cfg,
                    traffic_cfg=traffic_cfg,
                    coord_cfg=coord_cfg,
                    exact_cfg=exact_cfg,
                    gap_cfg=gap_cfg,
                    demand=d,
                    seed_index=s,
                    force=args.force,
                    task6_raw_dir=task6_raw_dir,
                )
                for d, s in tasks
            )
        else:
            G = get_or_build_graph(graph_cfg)
            for d, s in tasks:
                run_single_gap_instance(
                    graph_cfg=graph_cfg,
                    hazard_cfg=hazard_cfg,
                    scenario_cfg=scenario_cfg,
                    traffic_cfg=traffic_cfg,
                    coord_cfg=coord_cfg,
                    exact_cfg=exact_cfg,
                    gap_cfg=gap_cfg,
                    demand=d,
                    seed_index=s,
                    G=G,
                    force=args.force,
                    task6_raw_dir=task6_raw_dir,
                )

    # Aggregation & Figures
    summary_df, paired_df, horizon_warnings = aggregate_gap_results(
        output_dir=output_dir,
        demand_levels=gap_cfg.demand_levels,
        heuristics=gap_cfg.heuristics,
    )

    summary_df.to_csv(output_dir / "summary.csv", index=False)
    paired_df.to_csv(output_dir / "paired_gaps.csv", index=False)

    generate_gap_figures(
        summary_df=summary_df,
        paired_df=paired_df,
        heuristics=gap_cfg.heuristics,
        output_dir=output_dir,
    )

    print_formatted_tables(
        summary_df=summary_df,
        paired_df=paired_df,
        heuristics=gap_cfg.heuristics,
        horizon_warnings=horizon_warnings,
    )


if __name__ == "__main__":
    main()
