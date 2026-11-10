#!/usr/bin/env python
"""CLI runner and analysis for Task 6: Solver ladder experiment.

Evaluates Static, TimeDependent, Selfish, Coordinated (Online), and
Coordinated (Offline) routing across demand levels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CongestionConfig,
    CoordinatedConfig,
    HazardConfig,
    ScenarioConfig,
    StudyAreaConfig,
    TrafficConfig,
    load_config,
)
from src.diagnostics.exit_capacity import exit_capacity_report, save_exit_capacity_csv  # noqa: E402
from src.experiment.ladder_runner import run_ladder_experiment  # noqa: E402
from src.graph.loader import get_or_build_graph  # noqa: E402
from src.hazard.flood import FloodHazard  # noqa: E402
from src.metrics.stats import mean_ci, paired_diff_ci  # noqa: E402
from src.scenario import build_exits  # noqa: E402

ROUTER_COLORS = {
    "static": "#7f7f7f",
    "time_dependent": "#1f77b4",
    "selfish": "#ff7f0e",
    "coordinated_online": "#2ca02c",
    "coordinated_offline": "#9467bd",
}

ROUTER_MARKERS = {
    "static": "o",
    "time_dependent": "s",
    "selfish": "^",
    "coordinated_online": "D",
    "coordinated_offline": "*",
}


def load_all_json_records(output_dir: Path) -> list[dict[str, Any]]:
    """Load all raw JSON run records from output_dir / 'raw'."""
    raw_dir = output_dir / "raw"
    records = []
    if not raw_dir.exists():
        return records
    for path in sorted(raw_dir.glob("run_demand*_seed*.json")):
        with open(path, "r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def build_summary_dataframe(
    records: list[dict[str, Any]],
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Aggregate per-run metrics across seeds into mean and 95% CI."""
    flat_rows = []
    for r in records:
        demand = int(r["demand"])
        seed_idx = int(r["seed_index"])
        ceiling_pct = float(r.get("ceiling_pct", np.nan))
        for r_name, metrics in r.get("metrics", {}).items():
            row = {
                "demand": demand,
                "seed_index": seed_idx,
                "router": r_name,
                "ceiling_pct": ceiling_pct,
            }
            for k, v in metrics.items():
                row[k] = v
            flat_rows.append(row)

    if not flat_rows:
        return pd.DataFrame()

    raw_df = pd.DataFrame(flat_rows)
    metrics_to_agg = [
        "pct_safe",
        "ceiling_pct",
        "congestion_loss_pct",
        "mean_evac_min",
        "p90_evac_min",
        "last_evac_min",
        "mean_queue_delay_s",
        "max_congestion",
        "n_edges_over_storage",
        "n_trapped_no_route",
        "n_trapped_en_route",
        "mean_compute_ms",
    ]

    summary_rows = []
    for (demand, router), group in raw_df.groupby(["demand", "router"], sort=False):
        s_row: dict[str, Any] = {
            "demand": int(demand),
            "router": str(router),
            "n_seeds": len(group),
        }
        for m in metrics_to_agg:
            vals = group[m].dropna().values if m in group else np.array([])
            m_mean, c_low, c_high = mean_ci(vals, level=ci_level)
            s_row[f"{m}_mean"] = m_mean
            s_row[f"{m}_ci_low"] = c_low
            s_row[f"{m}_ci_high"] = c_high

        summary_rows.append(s_row)

    res = pd.DataFrame(summary_rows)
    # Sort demands numerically, routers predictably
    router_order = ["static", "time_dependent", "selfish", "coordinated_online", "coordinated_offline"]
    res["r_rank"] = res["router"].map(lambda x: router_order.index(x) if x in router_order else 99)
    res = res.sort_values(["demand", "r_rank"]).drop(columns=["r_rank"])
    return res


def build_paired_diffs(
    records: list[dict[str, Any]],
    comparisons: list[tuple[str, str]],
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Compute paired differences across seeds per demand level."""
    rows = []
    demands = sorted(list({r["demand"] for r in records}))

    for d in demands:
        d_records = [r for r in records if r["demand"] == d]
        for name_a, name_b in comparisons:
            for metric in ["pct_safe", "mean_evac_min", "mean_queue_delay_s"]:
                vals_a = []
                vals_b = []
                for rec in sorted(d_records, key=lambda x: x["seed_index"]):
                    m_a = rec.get("metrics", {}).get(name_a, {})
                    m_b = rec.get("metrics", {}).get(name_b, {})
                    if metric in m_a and metric in m_b and m_a[metric] is not None and m_b[metric] is not None:
                        vals_a.append(float(m_a[metric]))
                        vals_b.append(float(m_b[metric]))

                if len(vals_a) >= 2:
                    res = paired_diff_ci(vals_a, vals_b, level=ci_level)
                    rows.append(
                        {
                            "demand": d,
                            "metric": metric,
                            "comparison": f"{name_a} - {name_b}",
                            "router_a": name_a,
                            "router_b": name_b,
                            "n_pairs": res.n,
                            "mean_diff": res.mean_diff,
                            "ci_low": res.ci_low,
                            "ci_high": res.ci_high,
                            "p_ttest": res.p_ttest,
                            "p_wilcoxon": res.p_wilcoxon,
                        }
                    )

    return pd.DataFrame(rows)


def build_poa_table(
    records: list[dict[str, Any]],
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Compute Price of Anarchy (mean_evac_min(selfish) / mean_evac_min(router))."""
    rows = []
    demands = sorted(list({r["demand"] for r in records}))
    target_routers = ["static", "time_dependent", "coordinated_online", "coordinated_offline"]

    for d in demands:
        d_records = [r for r in records if r["demand"] == d]
        for r_name in target_routers:
            poa_vals = []
            for rec in sorted(d_records, key=lambda x: x["seed_index"]):
                m_self = rec.get("metrics", {}).get("selfish", {})
                m_r = rec.get("metrics", {}).get(r_name, {})
                t_self = m_self.get("mean_evac_min")
                t_r = m_r.get("mean_evac_min")
                if t_self and t_r and t_r > 0:
                    poa_vals.append(float(t_self) / float(t_r))

            if len(poa_vals) >= 2:
                m_val, c_low, c_high = mean_ci(poa_vals, level=ci_level)
                rows.append(
                    {
                        "demand": d,
                        "router": r_name,
                        "n_seeds": len(poa_vals),
                        "poa_mean": m_val,
                        "poa_ci_low": c_low,
                        "poa_ci_high": c_high,
                    }
                )

    return pd.DataFrame(rows)


def plot_ladder_figures(
    summary_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Generate the three required 150 dpi white-background figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    demands = sorted(summary_df["demand"].unique())
    routers = [r for r in ROUTER_COLORS if r in summary_df["router"].values]

    # 1. ladder_pct_safe.png
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150, facecolor="white")
    ax.set_facecolor("white")

    # Oracle ceiling band
    ceil_group = summary_df.groupby("demand")["ceiling_pct_mean"].first()
    ax.plot(
        demands,
        ceil_group.values,
        linestyle="--",
        color="#333333",
        linewidth=1.8,
        label="Oracle Ceiling (single-agent TD)",
    )
    all_pct_values = []
    for r in routers:
        sub = summary_df[summary_df["router"] == r].sort_values("demand")
        if sub.empty:
            continue
        xs = sub["demand"].values
        ys = sub["pct_safe_mean"].values
        y_low = sub["pct_safe_ci_low"].values
        y_high = sub["pct_safe_ci_high"].values
        all_pct_values.extend(ys)
        all_pct_values.extend(y_low)
        all_pct_values.extend(y_high)

        color = ROUTER_COLORS.get(r, "#333333")
        marker = ROUTER_MARKERS.get(r, "o")
        ax.plot(xs, ys, label=r, color=color, marker=marker, linewidth=1.8, markersize=6)
        if np.all(np.isfinite(y_low)) and np.all(np.isfinite(y_high)):
            ax.fill_between(xs, y_low, y_high, color=color, alpha=0.15)

    # Zoom Y-axis to data range (do not start at 0)
    all_pct_values.extend(ceil_group.values)
    valid_pcts = [v for v in all_pct_values if np.isfinite(v)]
    if valid_pcts:
        y_min = max(0.0, min(valid_pcts) - 2.5)
        y_max = min(100.0, max(valid_pcts) + 2.5)
        ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Demand (Number of Evacuees)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Evacuees Reaching Exit (% Safe)", fontsize=11, fontweight="bold")
    ax.set_title("Solver Ladder: Evacuation Survival Rate vs. Demand", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "ladder_pct_safe.png", dpi=150, facecolor="white")
    plt.close(fig)

    # 2. ladder_delay_evac.png (2 panels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150, facecolor="white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    for r in routers:
        sub = summary_df[summary_df["router"] == r].sort_values("demand")
        if sub.empty:
            continue
        xs = sub["demand"].values
        color = ROUTER_COLORS.get(r, "#333333")
        marker = ROUTER_MARKERS.get(r, "o")

        # Mean queuing delay
        q_ys = sub["mean_queue_delay_s_mean"].values
        q_low = sub["mean_queue_delay_s_ci_low"].values
        q_high = sub["mean_queue_delay_s_ci_high"].values
        ax1.plot(xs, q_ys, label=r, color=color, marker=marker, linewidth=1.8, markersize=5)
        if np.all(np.isfinite(q_low)) and np.all(np.isfinite(q_high)):
            ax1.fill_between(xs, q_low, q_high, color=color, alpha=0.15)

        # Mean evacuation time
        e_ys = sub["mean_evac_min_mean"].values
        e_low = sub["mean_evac_min_ci_low"].values
        e_high = sub["mean_evac_min_ci_high"].values
        ax2.plot(xs, e_ys, label=r, color=color, marker=marker, linewidth=1.8, markersize=5)
        if np.all(np.isfinite(e_low)) and np.all(np.isfinite(e_high)):
            ax2.fill_between(xs, e_low, e_high, color=color, alpha=0.15)

    ax1.set_xlabel("Demand", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Queuing Delay (s)", fontsize=11, fontweight="bold")
    ax1.set_title("Bottleneck Queuing Delay vs. Demand", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)

    ax2.set_xlabel("Demand", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Evacuation Duration (min)", fontsize=11, fontweight="bold")
    ax2.set_title("Mean Evacuation Duration vs. Demand", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "ladder_delay_evac.png", dpi=150, facecolor="white")
    plt.close(fig)

    # 3. ladder_loss.png (Congestion loss percentage per router vs demand)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150, facecolor="white")
    ax.set_facecolor("white")

    n_demands = len(demands)
    n_routers = len(routers)
    width = 0.8 / max(1, n_routers)
    x_indices = np.arange(n_demands)

    for i, r in enumerate(routers):
        sub = summary_df[summary_df["router"] == r].sort_values("demand")
        if sub.empty:
            continue
        ys = sub["congestion_loss_pct_mean"].values
        y_low = sub["congestion_loss_pct_ci_low"].values
        y_high = sub["congestion_loss_pct_ci_high"].values
        errs = np.zeros((2, len(ys)))
        if np.all(np.isfinite(y_low)) and np.all(np.isfinite(y_high)):
            errs[0, :] = np.clip(ys - y_low, 0, None)
            errs[1, :] = np.clip(y_high - ys, 0, None)

        offsets = x_indices + (i - n_routers / 2.0 + 0.5) * width
        color = ROUTER_COLORS.get(r, "#333333")
        ax.bar(
            offsets,
            ys,
            width=width,
            yerr=errs if np.all(np.isfinite(y_low)) else None,
            capsize=3,
            label=r,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )

    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(d) for d in demands], fontweight="bold")
    ax.set_xlabel("Demand (Number of Evacuees)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Congestion Loss (Oracle Ceiling - pct_safe, pp)", fontsize=11, fontweight="bold")
    ax.set_title("Evacuation Loss Attributable to Congestion", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "ladder_loss.png", dpi=150, facecolor="white")
    plt.close(fig)


def print_tables(
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    poa_df: pd.DataFrame,
    records: list[dict[str, Any]],
) -> None:
    """Print formatted terminal tables."""
    print("\n" + "=" * 105)
    print("TASK 6 SUMMARY: SOLVER LADDER METRICS ACROSS DEMAND LEVELS")
    print("=" * 105)
    header = (
        f"{'Demand':<8} {'Router':<22} {'pct_safe (95% CI)':<22} "
        f"{'Loss (pp)':<12} {'Evac min':<10} {'Queue (s)':<12} {'Max Cong':<10} {'Over-stor':<10}"
    )
    print(header)
    print("-" * 105)

    for _, row in summary_df.iterrows():
        d = int(row["demand"])
        r = str(row["router"])
        pct = f"{row['pct_safe_mean']:.1f} [{row['pct_safe_ci_low']:.1f}, {row['pct_safe_ci_high']:.1f}]"
        loss = f"{row['congestion_loss_pct_mean']:.2f}"
        evac = f"{row['mean_evac_min_mean']:.2f}"
        q_d = f"{row['mean_queue_delay_s_mean']:.1f}s"
        mc = f"{row['max_congestion_mean']:.1f}x"
        over = f"{row['n_edges_over_storage_mean']:.1f}"
        print(f"{d:<8} {r:<22} {pct:<22} {loss:<12} {evac:<10} {q_d:<12} {mc:<10} {over:<10}")

    print("=" * 105)

    if not paired_df.empty:
        print("\n" + "=" * 95)
        print("PAIRED DIFFERENCES ACROSS ROUTERS")
        print("=" * 95)
        print(f"{'Demand':<8} {'Metric':<20} {'Comparison':<35} {'Mean Diff (95% CI)':<22} {'p (t-test)':<10}")
        print("-" * 95)
        for _, p_row in paired_df.iterrows():
            d = int(p_row["demand"])
            met = str(p_row["metric"])
            comp = str(p_row["comparison"])
            diff = f"{p_row['mean_diff']:+.2f} [{p_row['ci_low']:.2f}, {p_row['ci_high']:.2f}]"
            p_val = f"{p_row['p_ttest']:.4f}"
            print(f"{d:<8} {met:<20} {comp:<35} {diff:<22} {p_val:<10}")
        print("=" * 95)

    if not poa_df.empty:
        print("\n" + "=" * 70)
        print("PRICE OF ANARCHY (Evacuation Duration vs Selfish)")
        print("PoA = mean_evac_min(selfish) / mean_evac_min(router)")
        print("=" * 70)
        print(f"{'Demand':<8} {'Router':<22} {'PoA Mean':<12} {'95% CI':<20}")
        print("-" * 70)
        for _, r_row in poa_df.iterrows():
            d = int(r_row["demand"])
            r = str(r_row["router"])
            poa_m = f"{r_row['poa_mean']:.3f}"
            poa_ci = f"[{r_row['poa_ci_low']:.3f}, {r_row['poa_ci_high']:.3f}]"
            print(f"{d:<8} {r:<22} {poa_m:<12} {poa_ci:<20}")
        print("=" * 70)

    # Trapped edge_closed_en_route diagnosis
    print("\n" + "=" * 70)
    print("EN-ROUTE HAZARD TRAPPING DIAGNOSTICS (edge_closed_en_route)")
    print("=" * 70)
    en_route_counts: dict[str, int] = {}
    total_sim_agents: dict[str, int] = {}
    for rec in records:
        for r_name, m in rec.get("metrics", {}).items():
            en_route_counts[r_name] = en_route_counts.get(r_name, 0) + int(m.get("n_trapped_en_route", 0))
            total_sim_agents[r_name] = total_sim_agents.get(r_name, 0) + int(m.get("n_agents", 0))

    for r_name in sorted(en_route_counts):
        cnt = en_route_counts[r_name]
        tot = total_sim_agents.get(r_name, 1)
        pct = 100.0 * cnt / tot if tot else 0.0
        print(f"  {r_name:<22}: {cnt:4d} trapped en-route ({pct:.2f}% of {tot} agents)")
    print("=" * 70)

    # Fallback exit check
    fallback_runs = [r for r in records if r.get("used_fallback_exits", False)]
    print("\n" + "=" * 70)
    print(f"EXIT DIAGNOSTICS: {len(fallback_runs)}/{len(records)} runs used fallback exits")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run solver ladder multi-agent experiment.")
    parser.add_argument("--graph-config", type=str, default="configs/study_area.yaml")
    parser.add_argument("--hazard-config", type=str, default="configs/hazard.yaml")
    parser.add_argument("--scenario-config", type=str, default="configs/scenario.yaml")
    parser.add_argument("--traffic-config", type=str, default="configs/traffic.yaml")
    parser.add_argument("--congestion-config", type=str, default="configs/congestion2.yaml")
    parser.add_argument("--coordinated-config", type=str, default="configs/coordinated.yaml")
    parser.add_argument("--quick", action="store_true", help="Run demands [1000, 2000] and 3 seeds")
    parser.add_argument("--dry-run", action="store_true", help="Benchmark single run and estimate total time")
    parser.add_argument("--force", action="store_true", help="Re-run even if cached outputs exist")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument("--analyze-only", action="store_true", help="Skip simulation and re-generate plots/tables")
    parser.add_argument("--demand-levels", type=str, default=None, help="Comma-separated demand levels override")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed indices override")
    args = parser.parse_args()

    graph_cfg = load_config(args.graph_config, StudyAreaConfig)
    hazard_cfg = load_config(args.hazard_config, HazardConfig)
    scenario_cfg = load_config(args.scenario_config, ScenarioConfig)
    traffic_cfg = load_config(args.traffic_config, TrafficConfig)
    cong_cfg = load_config(args.congestion_config, CongestionConfig)
    coord_cfg = load_config(args.coordinated_config, CoordinatedConfig)

    demand_override = [int(x.strip()) for x in args.demand_levels.split(",")] if args.demand_levels else None
    seed_override = [int(x.strip()) for x in args.seeds.split(",")] if args.seeds else None

    out_dir = Path(cong_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        run_ladder_experiment(
            graph_cfg=graph_cfg,
            hazard_cfg=hazard_cfg,
            scenario_cfg=scenario_cfg,
            traffic_cfg=traffic_cfg,
            cong_cfg=cong_cfg,
            coord_cfg=coord_cfg,
            dry_run=True,
            workers=args.workers or cong_cfg.workers,
        )
        return

    if not args.analyze_only:
        run_ladder_experiment(
            graph_cfg=graph_cfg,
            hazard_cfg=hazard_cfg,
            scenario_cfg=scenario_cfg,
            traffic_cfg=traffic_cfg,
            cong_cfg=cong_cfg,
            coord_cfg=coord_cfg,
            quick=args.quick,
            force=args.force,
            workers=args.workers,
            demand_override=demand_override,
            seed_override=seed_override,
        )

    # Analysis & Report
    records = load_all_json_records(out_dir)
    if not records:
        print("[ladder] No run records found in", out_dir / "raw")
        return

    summary_df = build_summary_dataframe(records, ci_level=cong_cfg.ci_level)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary metrics saved to {summary_path}")

    # Paired differences
    comparisons = [
        ("coordinated_online", "selfish"),
        ("coordinated_online", "time_dependent"),
        ("coordinated_online", "static"),
        ("coordinated_offline", "coordinated_online"),
        ("selfish", "time_dependent"),
    ]
    paired_df = build_paired_diffs(records, comparisons, ci_level=cong_cfg.ci_level)
    paired_path = out_dir / "paired_diffs.csv"
    paired_df.to_csv(paired_path, index=False)
    print(f"Paired differences saved to {paired_path}")

    # Price of anarchy
    poa_df = build_poa_table(records, ci_level=cong_cfg.ci_level)
    poa_path = out_dir / "poa.csv"
    poa_df.to_csv(poa_path, index=False)
    print(f"Price of anarchy saved to {poa_path}")

    # Exit capacity report on example run
    G = get_or_build_graph(graph_cfg)
    ex_rec = next((r for r in records if r["seed_index"] == cong_cfg.example_seed_index), records[0])
    h_seed = ex_rec["hazard_seed"]
    h_cfg = HazardConfig(
        spread_speed_m_per_min=float(cong_cfg.flood_speed_m_per_min),
        seed=int(h_seed),
        terrain_seed=int(hazard_cfg.terrain_seed),
        horizon_min=hazard_cfg.horizon_min,
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
    demands = sorted(summary_df["demand"].unique())
    ceil_pct = float(ex_rec.get("ceiling_pct", 92.0))
    exit_cap_report = exit_capacity_report(G, exits, demands, ceil_pct)
    exit_cap_path = out_dir / "exit_capacity.csv"
    save_exit_capacity_csv(exit_cap_report, exit_cap_path)
    print(f"Exit capacity report saved to {exit_cap_path}")

    # Figures
    plot_ladder_figures(summary_df, out_dir)
    print(f"Figures saved to {out_dir} (ladder_pct_safe.png, ladder_delay_evac.png, ladder_loss.png)")

    # Print tables and diagnostics
    print_tables(summary_df, paired_df, poa_df, records)

    # Print Exit Capacity numbers
    print("\n" + "=" * 70)
    print("EXIT CAPACITY & THROUGHPUT LOWER BOUNDS (Example Seed)")
    print(f"Total Exit Flow Capacity: {exit_cap_report['total_exit_capacity_veh_per_min']:.1f} veh/min across {exit_cap_report['n_exits']} exits")
    print(f"Oracle Ceiling: {exit_cap_report['ceiling_pct']:.1f}%")
    print(f"Note: {exit_cap_report['note']}")
    print("-" * 70)
    print(f"{'Demand':<10} {'Savable Agents':<18} {'T_min Throughput Bound (min)':<30}")
    for d_val, b in exit_cap_report["demand_bounds"].items():
        print(f"{d_val:<10} {b['n_savable']:<18.1f} {b['t_min_minutes']:<30.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

