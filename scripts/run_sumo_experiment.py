#!/usr/bin/env python
"""SUMO validation experiment for Task 8b.

Replays each router's routes inside SUMO with flood closures (open-loop)
and compares survival, delay, and spillback metrics against point-queue results.

OPEN-LOOP: Routes are fixed (computed by the point-queue router at planning time);
routers do NOT react to SUMO's real-time traffic.  State this limitation clearly.

Usage
-----
  python scripts/run_sumo_experiment.py                # full run
  python scripts/run_sumo_experiment.py --quick        # demand 1000, seed 0, all routers
  python scripts/run_sumo_experiment.py --dry-run      # timing estimate at demand 2000
  python scripts/run_sumo_experiment.py --analyze-only # aggregate existing results only
  python scripts/run_sumo_experiment.py --quick --unmatched  # use net_unmatched.net.xml
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CoordinatedConfig,
    HazardConfig,
    ScenarioConfig,
    StudyAreaConfig,
    SumoConfig,
    SumoExperimentConfig,
    TrafficConfig,
    load_config,
)
from src.experiment.ladder_runner import instantiate_router
from src.graph.loader import get_or_build_graph
from src.hazard.flood import FloodHazard
from src.routing.base import make_routing_params
from src.scenario import build_exits, generate_agents
from src.sim.multi_agent import simulate_population
from src.sumo.network import find_sumo_binary
from src.sumo.replay import VehicleOutcome, build_flood_time_map, replay_routes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or (isinstance(obj, float) and math.isnan(obj)):
        return None
    return obj


def _ci95(values: list[float]) -> tuple[float, float]:
    """Return (mean, half-width of 95% CI) using t-distribution."""
    from scipy import stats
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], float("nan")
    m = float(np.mean(values))
    se = float(stats.sem(values))
    hw = float(stats.t.ppf(0.975, df=len(values) - 1) * se)
    return m, hw


def _safe_mean(vals: list[float]) -> float:
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


def _safe_p90(vals: list[float]) -> float:
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    return float(np.percentile(clean, 90)) if clean else float("nan")


# ---------------------------------------------------------------------------
# Single (demand, seed, router) run
# ---------------------------------------------------------------------------

def run_pq_and_replay(
    demand: int,
    seed_index: int,
    router_name: str,
    graph_cfg: StudyAreaConfig,
    hazard_cfg: HazardConfig,
    scenario_cfg: ScenarioConfig,
    traffic_cfg: TrafficConfig,
    coord_cfg: CoordinatedConfig,
    sumo_cfg: SumoConfig,
    exp_cfg: SumoExperimentConfig,
    G,
    out_dir: Path,
    routes_dir: Path,
    sumo_dir: Path,
    net_path: Path,
    vtype_path: Path,
    edge_map_g2s: dict[str, str],
    edge_map_s2g: dict[str, list],
    force: bool = False,
    task6_raw_dir: Path | None = None,
) -> dict[str, Any]:
    """Run point-queue sim + SUMO replay for one (demand, seed, router) tuple.

    Returns a flat metrics dict suitable for DataFrame construction.
    """
    token = f"demand{demand}_seed{seed_index}_{router_name}"
    result_path = out_dir / "raw" / f"result_{token}.json"
    routes_pkl = routes_dir / f"routes_{token}.pkl"

    if not force and result_path.exists():
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # --- Setup hazard/exits/agents ---
    hazard_seed = exp_cfg.base_hazard_seed + seed_index
    scenario_seed = exp_cfg.base_scenario_seed + seed_index

    h_cfg = replace(
        hazard_cfg,
        spread_speed_m_per_min=float(exp_cfg.flood_speed_m_per_min),
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
    exits_set = set(exits)

    rng = np.random.default_rng(scenario_seed)
    full_sample = generate_agents(
        G=G,
        hazard=hazard,
        exits=exits,
        scenario_cfg=replace(scenario_cfg, n_evacuees=max(exp_cfg.demand_levels)),
        rng=rng,
    )
    agents = full_sample[:demand]

    params = make_routing_params(scenario_cfg)

    # --- Point-queue simulation (with route recording) ---
    t0 = time.perf_counter()
    router = instantiate_router(router_name, traffic_cfg, coord_cfg)
    pq_res = simulate_population(
        G=G,
        hazard=hazard,
        router=router,
        agents=agents,
        exits=exits,
        params=params,
        traffic_cfg=traffic_cfg,
    )
    pq_wall = time.perf_counter() - t0

    pq_safe_results = {r.agent_id: r for r in pq_res.agent_results if r.safe}
    pq_safe_count = len(pq_safe_results)
    pq_safe_pct = 100.0 * pq_safe_count / len(agents) if agents else 0.0

    # --- Save routes pickle ---
    routes_data = {r.agent_id: r.route_edges for r in pq_res.agent_results}
    routes_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(routes_pkl, "wb") as f:
        pickle.dump(routes_data, f)

    # --- Check vs ladder task6 raw (optional) ---
    pq_mismatch_warning = None
    if task6_raw_dir is not None:
        ladder_path = task6_raw_dir / f"agents_demand{demand}_seed{seed_index}.csv"
        if ladder_path.exists():
            try:
                ladder_df = pd.read_csv(ladder_path)
                sub = ladder_df[ladder_df["router_name"] == router_name]
                if len(sub) > 0:
                    ladder_safe = int((sub["safe"] == True).sum())  # noqa: E712
                    if ladder_safe != pq_safe_count:
                        pq_mismatch_warning = (
                            f"WARNING: PQ safe count mismatch for {router_name} "
                            f"d={demand} s={seed_index}: "
                            f"task6={ladder_safe}, now={pq_safe_count}"
                        )
                        print(pq_mismatch_warning)
            except Exception:
                pass

    # --- SUMO Replay ---
    # Build flood time map from hazard + edge_map
    flood_times = build_flood_time_map(hazard, {sid: uvk for sid, uvk in edge_map_s2g.items()})

    # Collect agents with routes for SUMO (safe + trapped with routes)
    agents_with_routes = []
    at_exit_agents = set()
    no_route_agents = set()

    agents_map = {a.id: a for a in agents}
    for r in pq_res.agent_results:
        if r.route_edges:
            agents_with_routes.append((agents_map[r.agent_id], r.route_edges))
        elif r.agent_id in exits_set:
            # Agent started at exit
            at_exit_agents.add(r.agent_id)
        else:
            no_route_agents.add(r.agent_id)

    t1 = time.perf_counter()
    # Unique label per run to allow parallel TraCI connections
    label = f"replay_{demand}_{seed_index}_{router_name}"[:63]  # TraCI label length limit
    outcomes = replay_routes(
        net_path=net_path,
        vtype_path=vtype_path,
        agents_with_routes=agents_with_routes,
        edge_map_g2s=edge_map_g2s,
        flood_times_by_sumo_edge=flood_times,
        step_length_s=sumo_cfg.step_length_s,
        time_to_teleport=sumo_cfg.time_to_teleport,
        horizon_s=exp_cfg.horizon_s,
        sumo_label=label,
    )
    sumo_wall = time.perf_counter() - t1

    # --- Compute SUMO metrics ---
    sumo_arrived = {aid: oc for aid, oc in outcomes.items() if oc.status == "ARRIVED"}
    sumo_trapped = {aid: oc for aid, oc in outcomes.items() if oc.status == "TRAPPED"}
    sumo_unfinished = {aid: oc for aid, oc in outcomes.items() if oc.status == "UNFINISHED"}
    sumo_never_inserted = {aid: oc for aid, oc in outcomes.items() if oc.status == "NEVER_INSERTED"}

    # Agents already at exit at departure count as safe in SUMO too
    total_sumo_safe = len(sumo_arrived) + len(at_exit_agents)
    sumo_safe_pct = 100.0 * total_sumo_safe / len(agents) if agents else 0.0
    sumo_trapped_pct = 100.0 * len(sumo_trapped) / len(agents) if agents else 0.0
    sumo_unfinished_pct = 100.0 * len(sumo_unfinished) / len(agents) if agents else 0.0
    sumo_never_inserted_pct = 100.0 * len(sumo_never_inserted) / len(agents) if agents else 0.0

    # Travel times (evac = arrival - planned_depart)
    sumo_evac_times = []
    sumo_waiting_times = []
    sumo_time_losses = []
    sumo_depart_delays = []
    n_waiting_gt60 = 0

    for oc in sumo_arrived.values():
        if oc.actual_arrival_s is not None and oc.planned_depart_s is not None:
            sumo_evac_times.append(oc.actual_arrival_s - oc.planned_depart_s)
        if oc.waiting_time_s is not None:
            sumo_waiting_times.append(oc.waiting_time_s)
            if oc.waiting_time_s > 60.0:
                n_waiting_gt60 += 1
        if oc.time_loss_s is not None:
            sumo_time_losses.append(oc.time_loss_s)
        if oc.depart_delay_s is not None:
            sumo_depart_delays.append(oc.depart_delay_s)

    # PQ times for common-safe agents
    common_ids = sorted(set(sumo_arrived.keys()) & set(pq_safe_results.keys()))
    pq_evac_common = [pq_safe_results[aid].evac_time_s for aid in common_ids]
    sumo_evac_common = [
        sumo_arrived[aid].actual_arrival_s - sumo_arrived[aid].planned_depart_s
        for aid in common_ids
        if sumo_arrived[aid].actual_arrival_s is not None
           and sumo_arrived[aid].planned_depart_s is not None
    ]

    # --- Outcome accounting (Requirement 3) ---
    n_arrived = len(sumo_arrived)
    n_trapped = len(sumo_trapped)
    n_unfinished = len(sumo_unfinished)
    n_never_inserted = len(sumo_never_inserted)
    n_at_exit = len(at_exit_agents)
    n_no_route = len(no_route_agents)
    n_vehicles_not_created = n_at_exit + n_no_route
    total_accounted = (
        n_arrived
        + n_trapped
        + n_unfinished
        + n_never_inserted
        + n_vehicles_not_created
    )
    assert total_accounted == len(agents), (
        f"Outcome accounting failed for {router_name} (demand={demand}, seed={seed_index}): "
        f"arrived={n_arrived} + trapped={n_trapped} + unfinished={n_unfinished} + "
        f"never_inserted={n_never_inserted} + vehicles_not_created={n_vehicles_not_created} "
        f"(at_exit={n_at_exit}, no_route={n_no_route}) == {total_accounted} != {len(agents)}"
    )
    print(
        f"  [accounting] {router_name} d={demand} s={seed_index}: "
        f"arrived={n_arrived} + trapped={n_trapped} + unfinished={n_unfinished} + "
        f"never_inserted={n_never_inserted} + not_created={n_vehicles_not_created} "
        f"(at_exit={n_at_exit}, no_route={n_no_route}) == {total_accounted} == n_agents ({len(agents)})"
    )

    # --- Agreement sets and statistics (Requirement 4) ---
    sumo_safe_set = set(sumo_arrived.keys()) | at_exit_agents
    pq_safe_set = set(pq_safe_results.keys())
    all_agent_ids = {a.id for a in agents}

    both_safe_ids = pq_safe_set & sumo_safe_set
    only_pq_ids = pq_safe_set - sumo_safe_set
    only_sumo_ids = sumo_safe_set - pq_safe_set
    neither_ids = all_agent_ids - (pq_safe_set | sumo_safe_set)

    agent_edge_len = {r.agent_id: len(r.route_edges) for r in pq_res.agent_results}

    both_departs = [agents_map[aid].depart_time_s for aid in both_safe_ids]
    both_edges = [agent_edge_len[aid] for aid in both_safe_ids]

    only_pq_departs = [agents_map[aid].depart_time_s for aid in only_pq_ids]
    only_pq_edges = [agent_edge_len[aid] for aid in only_pq_ids]

    only_sumo_departs = [agents_map[aid].depart_time_s for aid in only_sumo_ids]
    only_sumo_edges = [agent_edge_len[aid] for aid in only_sumo_ids]

    record: dict[str, Any] = {
        "demand": demand,
        "seed_index": seed_index,
        "router_name": router_name,
        "n_agents": len(agents),
        "pq_safe_pct": pq_safe_pct,
        "pq_safe_count": pq_safe_count,
        "sumo_safe_pct": sumo_safe_pct,
        "sumo_safe_count": total_sumo_safe,
        "sumo_trapped_pct": sumo_trapped_pct,
        "sumo_unfinished_pct": sumo_unfinished_pct,
        "sumo_never_inserted_pct": sumo_never_inserted_pct,
        "sumo_mean_evac_s": _safe_mean(sumo_evac_times),
        "sumo_p90_evac_s": _safe_p90(sumo_evac_times),
        "pq_mean_evac_s": _safe_mean([r.evac_time_s for r in pq_safe_results.values()]),
        "pq_p90_evac_s": _safe_p90([r.evac_time_s for r in pq_safe_results.values()]),
        "sumo_mean_evac_common_s": _safe_mean(sumo_evac_common),
        "sumo_p90_evac_common_s": _safe_p90(sumo_evac_common),
        "pq_mean_evac_common_s": _safe_mean(pq_evac_common),
        "pq_p90_evac_common_s": _safe_p90(pq_evac_common),
        "sumo_mean_waiting_s": _safe_mean(sumo_waiting_times),
        "sumo_mean_time_loss_s": _safe_mean(sumo_time_losses),
        "sumo_mean_depart_delay_s": _safe_mean(sumo_depart_delays),
        "n_waiting_gt60s": n_waiting_gt60,
        "sumo_wall_s": sumo_wall,
        "pq_wall_s": pq_wall,
        "pq_mismatch_warning": pq_mismatch_warning,
        "n_never_inserted": len(sumo_never_inserted),
        "n_common_safe": len(common_ids),
        "agree_both_safe": len(both_safe_ids),
        "agree_only_pq_safe": len(only_pq_ids),
        "agree_only_sumo_safe": len(only_sumo_ids),
        "agree_neither_safe": len(neither_ids),
        "agree_both_mean_depart_s": _safe_mean(both_departs),
        "agree_both_mean_route_edges": _safe_mean(both_edges),
        "agree_only_pq_mean_depart_s": _safe_mean(only_pq_departs),
        "agree_only_pq_mean_route_edges": _safe_mean(only_pq_edges),
        "agree_only_sumo_mean_depart_s": _safe_mean(only_sumo_departs),
        "agree_only_sumo_mean_route_edges": _safe_mean(only_sumo_edges),
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(record), f, indent=2)

    return record


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def aggregate_results(
    records: list[dict[str, Any]],
    out_dir: Path,
    routers: list[str],
    demand_levels: list[int],
) -> None:
    """Compute summary CSV, paired diffs CSV, and figures."""
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "all_raw.csv", index=False)

    summary_rows = []
    for demand in demand_levels:
        for router in routers:
            sub = df[(df["demand"] == demand) & (df["router_name"] == router)]
            if len(sub) == 0:
                continue
            row: dict[str, Any] = {"demand": demand, "router_name": router, "n_seeds": len(sub)}
            for col in [
                "pq_safe_pct", "sumo_safe_pct", "sumo_trapped_pct",
                "sumo_unfinished_pct", "sumo_never_inserted_pct",
                "sumo_mean_evac_s", "sumo_p90_evac_s",
                "pq_mean_evac_s", "pq_p90_evac_s",
                "sumo_mean_evac_common_s", "sumo_p90_evac_common_s",
                "pq_mean_evac_common_s", "pq_p90_evac_common_s",
                "sumo_mean_waiting_s", "sumo_mean_time_loss_s",
                "sumo_mean_depart_delay_s",
                "n_waiting_gt60s",
                "agree_both_safe", "agree_only_pq_safe",
                "agree_only_sumo_safe", "agree_neither_safe",
                "agree_both_mean_depart_s", "agree_both_mean_route_edges",
                "agree_only_pq_mean_depart_s", "agree_only_pq_mean_route_edges",
                "agree_only_sumo_mean_depart_s", "agree_only_sumo_mean_route_edges",
            ]:
                if col in sub.columns:
                    vals = sub[col].dropna().tolist()
                    m, hw = _ci95(vals)
                    row[f"{col}_mean"] = m
                    row[f"{col}_ci95"] = hw
            if "sumo_safe_pct_mean" in row and "pq_safe_pct_mean" in row:
                row["safe_diff_pct_mean"] = row["sumo_safe_pct_mean"] - row["pq_safe_pct_mean"]
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n[SUMO experiment] Summary CSV -> {out_dir / 'summary.csv'}")

    # --- Printed Summary Table (Requirement 5) ---
    print("\n" + "=" * 135)
    print("SUMO VALIDATION EXPERIMENT: PERFORMANCE SUMMARY (MEAN OVER SEEDS)")
    print("=" * 135)
    summary_cols = [
        "demand", "router_name",
        "pq_safe_pct_mean", "sumo_safe_pct_mean",
        "sumo_trapped_pct_mean", "sumo_unfinished_pct_mean", "sumo_never_inserted_pct_mean",
        "sumo_mean_evac_common_s_mean", "sumo_p90_evac_common_s_mean",
        "pq_mean_evac_common_s_mean", "pq_p90_evac_common_s_mean",
        "sumo_mean_waiting_s_mean", "sumo_mean_time_loss_s_mean",
        "sumo_mean_depart_delay_s_mean", "n_waiting_gt60s_mean",
    ]
    valid_cols = [c for c in summary_cols if c in summary_df.columns]
    rename_map = {
        "demand": "Demand",
        "router_name": "Router",
        "pq_safe_pct_mean": "PQ Safe%",
        "sumo_safe_pct_mean": "SUMO Safe%",
        "sumo_trapped_pct_mean": "Trapped%",
        "sumo_unfinished_pct_mean": "Unfin%",
        "sumo_never_inserted_pct_mean": "NeverIns%",
        "sumo_mean_evac_common_s_mean": "SUMO Evac(s)",
        "sumo_p90_evac_common_s_mean": "SUMO p90(s)",
        "pq_mean_evac_common_s_mean": "PQ Evac(s)",
        "pq_p90_evac_common_s_mean": "PQ p90(s)",
        "sumo_mean_waiting_s_mean": "Wait(s)",
        "sumo_mean_time_loss_s_mean": "Loss(s)",
        "sumo_mean_depart_delay_s_mean": "DepDelay(s)",
        "n_waiting_gt60s_mean": "Wait>60s",
    }
    disp_df = summary_df[valid_cols].rename(columns=rename_map)
    print(disp_df.to_string(index=False))

    # --- Survival Differences (sumo_safe_pct - pq_safe_pct) ---
    print("\n" + "=" * 80)
    print("SURVIVAL DIFFERENCE PER ROUTER (SUMO - PQ in percentage points)")
    print("=" * 80)
    for demand in demand_levels:
        sub = summary_df[summary_df["demand"] == demand]
        for _, r in sub.iterrows():
            diff = r["sumo_safe_pct_mean"] - r["pq_safe_pct_mean"]
            print(f"  Demand {int(demand):4d} | {r['router_name']:<20}: {diff:+.2f} pp "
                  f"(SUMO {r['sumo_safe_pct_mean']:.1f}% vs PQ {r['pq_safe_pct_mean']:.1f}%)")

    # --- Agreement Table (Requirement 4) ---
    print("\n" + "=" * 135)
    print("AGREEMENT TABLE: POINT-QUEUE VS SUMO SURVIVAL & ROUTE ATTRIBUTES")
    print("=" * 135)
    agree_cols = [
        "demand", "router_name",
        "agree_both_safe_mean", "agree_only_pq_safe_mean",
        "agree_only_sumo_safe_mean", "agree_neither_safe_mean",
        "agree_both_mean_depart_s_mean", "agree_both_mean_route_edges_mean",
        "agree_only_pq_mean_depart_s_mean", "agree_only_pq_mean_route_edges_mean",
        "agree_only_sumo_mean_depart_s_mean", "agree_only_sumo_mean_route_edges_mean",
    ]
    valid_agree_cols = [c for c in agree_cols if c in summary_df.columns]
    rename_agree_map = {
        "demand": "Demand",
        "router_name": "Router",
        "agree_both_safe_mean": "Both Safe",
        "agree_only_pq_safe_mean": "Only PQ",
        "agree_only_sumo_safe_mean": "Only SUMO",
        "agree_neither_safe_mean": "Neither",
        "agree_both_mean_depart_s_mean": "Both Dep(s)",
        "agree_both_mean_route_edges_mean": "Both Edges",
        "agree_only_pq_mean_depart_s_mean": "PQ-only Dep(s)",
        "agree_only_pq_mean_route_edges_mean": "PQ-only Edges",
        "agree_only_sumo_mean_depart_s_mean": "SUMO-only Dep(s)",
        "agree_only_sumo_mean_route_edges_mean": "SUMO-only Edges",
    }
    disp_agree = summary_df[valid_agree_cols].rename(columns=rename_agree_map)
    print(disp_agree.to_string(index=False))

    # Paired differences
    paired_rows = []
    comparison_pairs = [
        ("selfish_forecast", "coordinated_online"),
        ("time_dependent", "selfish_forecast"),
        ("coordinated_online", "selfish"),
        ("selfish_forecast", "static"),
    ]
    for demand in demand_levels:
        for r_a, r_b in comparison_pairs:
            sub_a = df[(df["demand"] == demand) & (df["router_name"] == r_a)].set_index("seed_index")
            sub_b = df[(df["demand"] == demand) & (df["router_name"] == r_b)].set_index("seed_index")
            common_seeds = sorted(set(sub_a.index) & set(sub_b.index))
            if not common_seeds:
                continue
            for col in ["sumo_safe_pct", "sumo_mean_waiting_s", "sumo_mean_time_loss_s"]:
                if col not in sub_a.columns or col not in sub_b.columns:
                    continue
                diffs = []
                for s in common_seeds:
                    va = sub_a.loc[s, col] if s in sub_a.index else float("nan")
                    vb = sub_b.loc[s, col] if s in sub_b.index else float("nan")
                    if not (math.isnan(float(va)) or math.isnan(float(vb))):
                        diffs.append(float(va) - float(vb))
                if diffs:
                    m, hw = _ci95(diffs)
                    paired_rows.append({
                        "demand": demand, "router_A": r_a, "router_B": r_b,
                        "metric": col, "mean_diff_A_minus_B": m, "ci95_hw": hw,
                        "n_seeds": len(diffs),
                    })
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(out_dir / "paired_diffs.csv", index=False)
    print(f"\n[SUMO experiment] Paired diffs -> {out_dir / 'paired_diffs.csv'}")
    print(paired_df[paired_df["metric"] == "sumo_safe_pct"].to_string(index=False))

    # Router ranking comparison
    print("\n[SUMO experiment] Router ranking by pct_safe:")
    for demand in demand_levels:
        sub = summary_df[summary_df["demand"] == demand]
        if len(sub) == 0:
            continue
        pq_rank = sub.sort_values("pq_safe_pct_mean", ascending=False)["router_name"].tolist()
        sumo_rank = sub.sort_values("sumo_safe_pct_mean", ascending=False)["router_name"].tolist()
        print(f"  Demand {demand}: PQ rank: {pq_rank} | SUMO rank: {sumo_rank}")

    # Figure 1: sumo_vs_pq_safe.png
    _plot_safe_comparison(summary_df, demand_levels, routers, out_dir)
    # Figure 2: sumo_delay.png
    _plot_delay(summary_df, demand_levels, routers, out_dir)


def _plot_safe_comparison(df: pd.DataFrame, demands: list[int], routers: list[str], out_dir: Path) -> None:
    """Grouped bars: PQ vs SUMO pct_safe per router per demand."""
    n_routers = len(routers)
    n_demands = len(demands)
    fig, axes = plt.subplots(1, n_demands, figsize=(5 * n_demands, 5), sharey=True)
    if n_demands == 1:
        axes = [axes]

    colors_pq = "#4472C4"
    colors_sumo = "#ED7D31"

    for ax, demand in zip(axes, demands):
        sub = df[df["demand"] == demand]
        x = np.arange(n_routers)
        width = 0.35
        pq_means = []
        pq_errs = []
        sumo_means = []
        sumo_errs = []
        for r in routers:
            row = sub[sub["router_name"] == r]
            pq_means.append(float(row["pq_safe_pct_mean"].iloc[0]) if len(row) > 0 else float("nan"))
            pq_errs.append(float(row["pq_safe_pct_ci95"].iloc[0]) if len(row) > 0 else 0.0)
            sumo_means.append(float(row["sumo_safe_pct_mean"].iloc[0]) if len(row) > 0 else float("nan"))
            sumo_errs.append(float(row["sumo_safe_pct_ci95"].iloc[0]) if len(row) > 0 else 0.0)

        ax.bar(x - width / 2, pq_means, width, yerr=pq_errs, label="Point-Queue", color=colors_pq, capsize=4)
        ax.bar(x + width / 2, sumo_means, width, yerr=sumo_errs, label="SUMO", color=colors_sumo, capsize=4)
        ax.set_title(f"Demand {demand}")
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace("_", "\n") for r in routers], fontsize=8)
        ax.set_ylabel("% Safe")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)

    fig.suptitle("PQ vs SUMO pct_safe by Router and Demand", fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_dir / "sumo_vs_pq_safe.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SUMO experiment] Figure -> {out_dir / 'sumo_vs_pq_safe.png'}")


def _plot_delay(df: pd.DataFrame, demands: list[int], routers: list[str], out_dir: Path) -> None:
    """Mean waiting time and time loss per router per demand."""
    n_demands = len(demands)
    fig, axes = plt.subplots(1, n_demands, figsize=(5 * n_demands, 5), sharey=False)
    if n_demands == 1:
        axes = [axes]

    for ax, demand in zip(axes, demands):
        sub = df[df["demand"] == demand]
        x = np.arange(len(routers))
        width = 0.35
        waiting = []
        time_loss = []
        for r in routers:
            row = sub[sub["router_name"] == r]
            waiting.append(float(row["sumo_mean_waiting_s_mean"].iloc[0]) if len(row) > 0 else float("nan"))
            time_loss.append(float(row["sumo_mean_time_loss_s_mean"].iloc[0]) if len(row) > 0 else float("nan"))

        ax.bar(x - width / 2, waiting, width, label="Mean Waiting (s)", color="#70AD47", capsize=4)
        ax.bar(x + width / 2, time_loss, width, label="Mean Time Loss (s)", color="#FFC000", capsize=4)
        ax.set_title(f"Demand {demand}")
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace("_", "\n") for r in routers], fontsize=8)
        ax.set_ylabel("Seconds")
        ax.legend(fontsize=8)

    fig.suptitle("SUMO Mean Waiting Time and Time Loss by Router and Demand", fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_dir / "sumo_delay.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SUMO experiment] Figure -> {out_dir / 'sumo_delay.png'}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SUMO validation experiment (Task 8b).")
    parser.add_argument("--graph-config", default="configs/study_area.yaml")
    parser.add_argument("--hazard-config", default="configs/hazard.yaml")
    parser.add_argument("--scenario-config", default="configs/scenario.yaml")
    parser.add_argument("--traffic-config", default="configs/traffic.yaml")
    parser.add_argument("--coord-config", default="configs/coordinated.yaml")
    parser.add_argument("--sumo-config", default="configs/sumo.yaml")
    parser.add_argument("--exp-config", default="configs/sumo_experiment.yaml")
    parser.add_argument("--quick", action="store_true", help="demand 1000, seed 0, all routers")
    parser.add_argument("--dry-run", action="store_true", help="Estimate run time at demand 2000")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--analyze-only", action="store_true", help="Aggregate existing results only")
    parser.add_argument("--unmatched", action="store_true", help="Use net_unmatched.net.xml")
    args = parser.parse_args()

    graph_cfg = load_config(PROJECT_ROOT / args.graph_config, StudyAreaConfig)
    hazard_cfg = load_config(PROJECT_ROOT / args.hazard_config, HazardConfig)
    scenario_cfg = load_config(PROJECT_ROOT / args.scenario_config, ScenarioConfig)
    traffic_cfg = load_config(PROJECT_ROOT / args.traffic_config, TrafficConfig)
    sumo_cfg = load_config(PROJECT_ROOT / args.sumo_config, SumoConfig)
    exp_cfg = load_config(PROJECT_ROOT / args.exp_config, SumoExperimentConfig)

    # CoordinatedConfig (may not exist in all setups)
    coord_path = PROJECT_ROOT / args.coord_config
    if coord_path.exists():
        coord_cfg = load_config(coord_path, CoordinatedConfig)
    else:
        coord_cfg = CoordinatedConfig()

    workers = args.workers if args.workers is not None else exp_cfg.workers
    out_dir = PROJECT_ROOT / exp_cfg.output_dir
    routes_dir = out_dir / "routes"
    sumo_dir = PROJECT_ROOT / sumo_cfg.sumo_dir

    # Network paths
    if args.unmatched:
        net_path = sumo_dir / "net_unmatched.net.xml"
        if not net_path.exists():
            net_path = sumo_dir / "net.net.xml"
            print(f"[warn] net_unmatched.net.xml not found, using net.net.xml")
    else:
        net_path = sumo_dir / "net.net.xml"

    vtype_path = sumo_dir / "vtype.add.xml"
    map_file = sumo_dir / "edge_map.json"

    # Validate SUMO environment
    try:
        find_sumo_binary("sumo")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if not net_path.exists():
        print(f"ERROR: SUMO network not found at {net_path}. Run export_sumo_network.py first.")
        sys.exit(1)

    with open(map_file, "r", encoding="utf-8") as f:
        edge_map_data = json.load(f)
    edge_map_g2s: dict[str, str] = edge_map_data["graph_to_sumo"]
    edge_map_s2g: dict[str, list] = edge_map_data["sumo_to_graph"]

    out_dir.mkdir(parents=True, exist_ok=True)
    routes_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    G = get_or_build_graph(graph_cfg)

    # Task6 raw for sanity check
    task6_raw = PROJECT_ROOT / "outputs" / "task6" / "raw"
    task6_raw_dir = task6_raw if task6_raw.exists() else None

    # --- Quick run ---
    if args.quick:
        demand_levels = [1000]
        seed_indices = [0]
        routers = exp_cfg.routers
        print(f"[SUMO experiment] QUICK RUN: demand={demand_levels}, seeds={seed_indices}, routers={routers}")
    # --- Dry run ---
    elif args.dry_run:
        print("[SUMO experiment] DRY RUN: timing one (demand=2000, seed=0, time_dependent)...")
        t0 = time.perf_counter()
        rec = run_pq_and_replay(
            demand=2000, seed_index=0, router_name="time_dependent",
            graph_cfg=graph_cfg, hazard_cfg=hazard_cfg, scenario_cfg=scenario_cfg,
            traffic_cfg=traffic_cfg, coord_cfg=coord_cfg, sumo_cfg=sumo_cfg,
            exp_cfg=exp_cfg, G=G, out_dir=out_dir, routes_dir=routes_dir,
            sumo_dir=sumo_dir, net_path=net_path, vtype_path=vtype_path,
            edge_map_g2s=edge_map_g2s, edge_map_s2g=edge_map_s2g,
            force=True,
            task6_raw_dir=task6_raw_dir,
        )
        dt = time.perf_counter() - t0
        n_tasks = len(exp_cfg.demand_levels) * exp_cfg.n_seeds * len(exp_cfg.routers)
        print(f"[DRY RUN] One (demand=2000, seed=0, time_dependent): {dt:.1f}s")
        print(f"[DRY RUN] PQ safe: {rec['pq_safe_pct']:.1f}% | SUMO safe: {rec['sumo_safe_pct']:.1f}%")
        print(f"[DRY RUN] Estimated total ({n_tasks} tasks, {workers} workers): {dt * n_tasks / workers:.0f}s = {dt * n_tasks / workers / 60:.1f} min")
        return
    # --- Analyze only ---
    elif args.analyze_only:
        records = []
        for f in (out_dir / "raw").glob("result_*.json"):
            try:
                with open(f) as fh:
                    records.append(json.load(fh))
            except Exception:
                pass
        if not records:
            print("[SUMO experiment] No results found to analyze.")
            return
        aggregate_results(records, out_dir, exp_cfg.routers, exp_cfg.demand_levels)
        return
    else:
        demand_levels = exp_cfg.demand_levels
        seed_indices = list(range(exp_cfg.n_seeds))
        routers = exp_cfg.routers

    # --- Full / quick run ---
    tasks = [
        (demand, seed_idx, router)
        for demand in demand_levels
        for seed_idx in seed_indices
        for router in routers
    ]
    print(f"[SUMO experiment] {len(tasks)} tasks | {workers} workers | net: {net_path.name}")

    t_start = time.perf_counter()
    if workers > 1:
        results = Parallel(n_jobs=workers, verbose=5)(
            delayed(run_pq_and_replay)(
                demand=d, seed_index=s, router_name=r,
                graph_cfg=graph_cfg, hazard_cfg=hazard_cfg,
                scenario_cfg=scenario_cfg, traffic_cfg=traffic_cfg,
                coord_cfg=coord_cfg, sumo_cfg=sumo_cfg, exp_cfg=exp_cfg,
                G=G, out_dir=out_dir, routes_dir=routes_dir,
                sumo_dir=sumo_dir, net_path=net_path, vtype_path=vtype_path,
                edge_map_g2s=edge_map_g2s, edge_map_s2g=edge_map_s2g,
                force=args.force, task6_raw_dir=task6_raw_dir,
            )
            for d, s, r in tasks
        )
    else:
        results = []
        for d, s, r in tasks:
            rec = run_pq_and_replay(
                demand=d, seed_index=s, router_name=r,
                graph_cfg=graph_cfg, hazard_cfg=hazard_cfg,
                scenario_cfg=scenario_cfg, traffic_cfg=traffic_cfg,
                coord_cfg=coord_cfg, sumo_cfg=sumo_cfg, exp_cfg=exp_cfg,
                G=G, out_dir=out_dir, routes_dir=routes_dir,
                sumo_dir=sumo_dir, net_path=net_path, vtype_path=vtype_path,
                edge_map_g2s=edge_map_g2s, edge_map_s2g=edge_map_s2g,
                force=args.force, task6_raw_dir=task6_raw_dir,
            )
            results.append(rec)
            print(f"  [{d}, s{s}, {r}] PQ={rec['pq_safe_pct']:.1f}% SUMO={rec['sumo_safe_pct']:.1f}%")

    elapsed = time.perf_counter() - t_start
    print(f"[SUMO experiment] Completed {len(results)} runs in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Warn about never-inserted or mismatches
    for rec in results:
        if rec.get("n_never_inserted", 0) > 0:
            print(f"  WARNING: {rec['n_never_inserted']} never-inserted vehicles in "
                  f"demand={rec['demand']} seed={rec['seed_index']} router={rec['router_name']}")
        if rec.get("pq_mismatch_warning"):
            print(f"  {rec['pq_mismatch_warning']}")

    aggregate_results(results, out_dir, routers, demand_levels)


if __name__ == "__main__":
    main()

