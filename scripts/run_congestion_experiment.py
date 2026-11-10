#!/usr/bin/env python
"""CLI script: run multi-agent demand sweep experiments with road capacities.

Usage
-----
    python scripts/run_congestion_experiment.py --graph-config configs/study_area.yaml \\
        --hazard-config configs/hazard.yaml --scenario-config configs/scenario.yaml \\
        --traffic-config configs/traffic.yaml --congestion-config configs/congestion.yaml --quick
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CongestionConfig,
    HazardConfig,
    ScenarioConfig,
    StudyAreaConfig,
    TrafficConfig,
    load_config,
)
from src.experiment.congestion_runner import run_congestion_experiment
from src.graph.loader import get_or_build_graph
from src.hazard.flood import FloodHazard
from src.metrics.stats import mean_ci, paired_diff_ci
from src.scenario import build_exits


# ---------------------------------------------------------------------------
# Visualisation Helpers
# ---------------------------------------------------------------------------

def _edge_lines(G):
    """Return coordinate sequences and edge keys for plotting."""
    lines = []
    edge_keys = []
    for u, v, k, data in G.edges(keys=True, data=True):
        if "geometry" in data:
            coords = list(data["geometry"].coords)
        else:
            coords = [
                (G.nodes[u]["x"], G.nodes[u]["y"]),
                (G.nodes[v]["x"], G.nodes[v]["y"]),
            ]
        lines.append(coords)
        edge_keys.append((u, v, k))
    return lines, edge_keys


def _graph_extent(G, margin_frac: float = 0.02):
    xs = [d["x"] for _, d in G.nodes(data=True)]
    ys = [d["y"] for _, d in G.nodes(data=True)]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    mx = (x_max - x_min) * margin_frac
    my = (y_max - y_min) * margin_frac
    return x_min - mx, x_max + mx, y_min - my, y_max + my


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------

def plot_demand_sweep(
    summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot pct_safe and mean_queue_delay_s vs demand across all routers."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), facecolor="white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    styles = {
        "static": {
            "color": "#d9534f",
            "marker": "o",
            "linestyle": "-",
            "label": "Static Dijkstra (snapshot)",
        },
        "reactive": {
            "color": "#f0ad4e",
            "marker": "s",
            "linestyle": "--",
            "label": "Reactive Re-routing",
        },
        "time_dependent": {
            "color": "#5cb85c",
            "marker": "^",
            "linestyle": "-.",
            "label": "Time-Dependent (forecast)",
        },
        "selfish": {
            "color": "#337ab7",
            "marker": "D",
            "linestyle": ":",
            "label": "Selfish (live congestion)",
        },
    }

    demands = sorted(summary_df["demand"].unique())

    # Panel 1: pct_safe vs demand
    for router_name, style in styles.items():
        subset = summary_df[summary_df["router"] == router_name].sort_values("demand")
        if subset.empty:
            continue

        dems = subset["demand"].values
        means = subset["pct_safe_mean"].values
        lows = subset["pct_safe_ci_low"].values
        highs = subset["pct_safe_ci_high"].values

        lows_clean = np.clip(np.where(np.isfinite(lows), lows, means), 0.0, 100.0)
        highs_clean = np.clip(np.where(np.isfinite(highs), highs, means), 0.0, 100.0)
        means_clean = np.clip(means, 0.0, 100.0)

        ax1.plot(
            dems,
            means_clean,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=6,
            zorder=3,
            label=style["label"],
        )
        ax1.fill_between(
            dems,
            lows_clean,
            highs_clean,
            color=style["color"],
            alpha=0.18,
            zorder=2,
        )

    ax1.set_xlabel("Evacuation Demand (Number of Evacuees)", fontsize=11)
    ax1.set_ylabel("% Evacuees Safe (pct_safe)", fontsize=11)
    ax1.set_title("Evacuation Success vs Demand Under Road Capacity (Mean ± 95% CI)", fontsize=12)
    ax1.set_ylim(-2, 102)
    ax1.set_xticks(demands)
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.legend(frameon=True, loc="lower left", fontsize=9.5)

    # Panel 2: mean_queue_delay_s vs demand
    for router_name, style in styles.items():
        subset = summary_df[summary_df["router"] == router_name].sort_values("demand")
        if subset.empty:
            continue

        dems = subset["demand"].values
        means = subset["mean_queue_delay_s_mean"].values
        lows = subset["mean_queue_delay_s_ci_low"].values
        highs = subset["mean_queue_delay_s_ci_high"].values

        lows_clean = np.maximum(np.where(np.isfinite(lows), lows, means), 0.0)
        highs_clean = np.maximum(np.where(np.isfinite(highs), highs, means), 0.0)

        ax2.plot(
            dems,
            means,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=6,
            zorder=3,
            label=style["label"],
        )
        ax2.fill_between(
            dems,
            lows_clean,
            highs_clean,
            color=style["color"],
            alpha=0.18,
            zorder=2,
        )

    ax2.set_xlabel("Evacuation Demand (Number of Evacuees)", fontsize=11)
    ax2.set_ylabel("Mean Queuing Delay per Agent (seconds)", fontsize=11)
    ax2.set_title("Congestion Delay vs Evacuation Demand (Mean ± 95% CI)", fontsize=12)
    ax2.set_xticks(demands)
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.legend(frameon=True, loc="upper left", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Demand sweep plot saved to {out_path}")


def plot_congestion_map(
    G: nx.MultiDiGraph,
    edge_stats_df: pd.DataFrame,
    exits: list[int],
    sources: list[int],
    demand: int,
    seed_idx: int,
    speed: float,
    out_path: Path,
) -> None:
    """Draw network map colored and scaled by edge total queue delay."""
    lines, edge_keys = _edge_lines(G)
    extent = _graph_extent(G)

    # Map edge keys to total_queue_delay_s
    delay_dict = {}
    for _, row in edge_stats_df.iterrows():
        e = (int(row["u"]), int(row["v"]), int(row["k"]))
        delay_dict[e] = float(row["total_queue_delay_s"])

    max_delay = max(delay_dict.values()) if delay_dict else 1.0

    colors = []
    widths = []

    cmap = plt.cm.plasma

    for e in edge_keys:
        d = delay_dict.get(e, 0.0)
        if d <= 0.1:
            colors.append("#d0d0d0")
            widths.append(0.6)
        else:
            # Log scaling from 0 to 1
            norm_val = math.log10(1.0 + d) / math.log10(1.0 + max(max_delay, 10.0))
            colors.append(cmap(norm_val))
            widths.append(1.0 + 3.5 * norm_val)

    fig, ax = plt.subplots(figsize=(11, 11), facecolor="white")
    ax.set_facecolor("white")

    lc = LineCollection(lines, colors=colors, linewidths=widths, zorder=2)
    ax.add_collection(lc)

    # Exits
    exit_xs = [G.nodes[n]["x"] for n in exits if n in G.nodes]
    exit_ys = [G.nodes[n]["y"] for n in exits if n in G.nodes]
    ax.scatter(
        exit_xs,
        exit_ys,
        s=140,
        c="#2ca02c",
        marker="^",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label=f"Perimeter Exits (N={len(exit_xs)})",
    )

    # Flood sources
    src_xs = [G.nodes[n]["x"] for n in sources if n in G.nodes]
    src_ys = [G.nodes[n]["y"] for n in sources if n in G.nodes]
    ax.scatter(
        src_xs,
        src_ys,
        s=150,
        c="#1f77b4",
        marker="o",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label=f"Flood Sources (N={len(src_xs)})",
    )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        f"Bottleneck Queuing Congestion Map\n"
        f"Demand = {demand} agents | Seed Index = {seed_idx} | Flood Speed = {speed:.0f} m/min | Time-Dependent Router",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(frameon=True, loc="upper right", fontsize=9.5)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_delay))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.03, pad=0.03)
    cbar.set_label("Total Edge Queuing Delay (seconds, log-scaled width)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Congestion map saved to {out_path}")


# ---------------------------------------------------------------------------
# Aggregation & Analysis
# ---------------------------------------------------------------------------

def aggregate_congestion_results(
    raw_records: list[dict[str, Any]],
    cong_cfg: CongestionConfig,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate per-run JSONs into summary, paired difference, and diagnostic frames."""
    # 1. Diagnostics DataFrame
    diag_rows = []
    for r in raw_records:
        spread = r.get("exit_spread", {})
        diag_rows.append(
            {
                "demand": int(r["demand"]),
                "seed_index": int(r["seed_index"]),
                "flood_speed_m_per_min": float(r["flood_speed_m_per_min"]),
                "hazard_seed": int(r["hazard_seed"]),
                "scenario_seed": int(r["scenario_seed"]),
                "used_fallback_exits": bool(r["used_fallback_exits"]),
                "n_fallback_exits": int(r["n_fallback_exits"]),
                "n_exits_effective": int(r.get("n_exits_effective", len(r.get("exit_spread", {})))),
                "departure_window_capped": bool(r["departure_window_capped"]),
                "n_agents": int(r["n_agents"]),
                "pairwise_mean_dist_m": float(spread.get("pairwise_mean_dist_m", np.nan)),
                "pairwise_min_dist_m": float(spread.get("pairwise_min_dist_m", np.nan)),
                "angular_coverage_deg": float(spread.get("angular_coverage_deg", np.nan)),
                "n_distinct_sectors": int(spread.get("n_distinct_sectors", 0)),
            }
        )
    diag_df = pd.DataFrame(diag_rows).sort_values(["demand", "seed_index"])
    diag_df.to_csv(out_dir / "diagnostics.csv", index=False)

    # 2. Summary DataFrame per (demand, router)
    demands = sorted({r["demand"] for r in raw_records})
    routers = ["static", "reactive", "time_dependent", "selfish"]
    metric_keys = [
        "pct_safe",
        "mean_evac_min",
        "mean_queue_delay_s",
        "max_congestion",
        "n_edges_over_storage",
        "n_trapped_no_route",
        "n_trapped_en_route",
    ]

    summary_rows = []
    for d in demands:
        subset = [r for r in raw_records if r["demand"] == d]
        for router in routers:
            row: dict[str, Any] = {
                "demand": d,
                "router": router,
                "n_runs": len(subset),
            }
            for m in metric_keys:
                vals = [
                    r["metrics"][router][m]
                    for r in subset
                    if router in r["metrics"] and r["metrics"][router][m] is not None
                ]
                m_mean, c_low, c_high = mean_ci(vals, level=cong_cfg.ci_level)
                row[f"{m}_mean"] = m_mean
                row[f"{m}_ci_low"] = c_low
                row[f"{m}_ci_high"] = c_high
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    # 3. Paired Differences per demand in pct_safe
    comparisons = [
        ("selfish - static", "selfish", "static"),
        ("time_dependent - static", "time_dependent", "static"),
        ("time_dependent - selfish", "time_dependent", "selfish"),
        ("reactive - static", "reactive", "static"),
    ]

    paired_rows = []
    for d in demands:
        subset = [r for r in raw_records if r["demand"] == d]
        for comp_name, r_a, r_b in comparisons:
            a_vals = [
                r["metrics"][r_a]["pct_safe"]
                for r in subset
                if r_a in r["metrics"] and r["metrics"][r_a]["pct_safe"] is not None
            ]
            b_vals = [
                r["metrics"][r_b]["pct_safe"]
                for r in subset
                if r_b in r["metrics"] and r["metrics"][r_b]["pct_safe"] is not None
            ]
            diff_res = paired_diff_ci(a_vals, b_vals, level=cong_cfg.ci_level)
            paired_rows.append(
                {
                    "demand": d,
                    "comparison": comp_name,
                    "mean_diff": diff_res.mean_diff,
                    "ci_low": diff_res.ci_low,
                    "ci_high": diff_res.ci_high,
                    "n": diff_res.n,
                    "p_ttest": diff_res.p_ttest,
                    "p_wilcoxon": diff_res.p_wilcoxon,
                }
            )

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(out_dir / "paired_diffs.csv", index=False)

    return summary_df, paired_df, diag_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent demand sweep evaluation under point-queue congestion.")
    parser.add_argument("--graph-config", type=str, default="configs/study_area.yaml")
    parser.add_argument("--hazard-config", type=str, default="configs/hazard.yaml")
    parser.add_argument("--scenario-config", type=str, default="configs/scenario.yaml")
    parser.add_argument("--traffic-config", type=str, default="configs/traffic.yaml")
    parser.add_argument("--congestion-config", type=str, default="configs/congestion.yaml")
    parser.add_argument("--quick", action="store_true", help="Run 2 demand levels (300, 1000) on 3 seeds.")
    parser.add_argument("--dry-run", action="store_true", help="Run demand 1000 single seed to estimate time.")
    parser.add_argument("--force", action="store_true", help="Force rerun ignoring cached files.")
    parser.add_argument("--workers", type=int, default=None, help="Worker count override.")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing raw records.")
    args = parser.parse_args()

    graph_cfg = load_config(args.graph_config, StudyAreaConfig)
    hazard_cfg = load_config(args.hazard_config, HazardConfig)
    scenario_cfg = load_config(args.scenario_config, ScenarioConfig)
    traffic_cfg = load_config(args.traffic_config, TrafficConfig)
    cong_cfg = load_config(args.congestion_config, CongestionConfig)

    out_dir = Path(cong_cfg.output_dir)
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        records = run_congestion_experiment(
            graph_cfg=graph_cfg,
            hazard_cfg=hazard_cfg,
            scenario_cfg=scenario_cfg,
            traffic_cfg=traffic_cfg,
            cong_cfg=cong_cfg,
            quick=args.quick,
            dry_run=args.dry_run,
            force=args.force,
            workers=args.workers,
        )
        if args.dry_run:
            return
    else:
        print("[congestion] Loading cached raw records...")
        records = []
        for path in sorted(raw_dir.glob("run_*.json")):
            with open(path, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        print(f"[congestion] Loaded {len(records)} records.")

    if not records:
        print("ERROR: No run records found.")
        return

    # 1. Aggregation
    summary_df, paired_df, diag_df = aggregate_congestion_results(records, cong_cfg, out_dir)

    # 2. Print Summary Tables
    print("\n" + "=" * 105)
    print("TASK 5 SUMMARY: MULTI-AGENT METRICS ACROSS DEMAND LEVELS")
    print("=" * 105)
    print(
        f"{'Demand':<8} {'Router':<16} {'pct_safe (95% CI)':<24} "
        f"{'Evac min':<10} {'Queue delay (s)':<18} {'Max Congest':<14} {'Over-storage'}"
    )
    print("-" * 105)
    for _, row in summary_df.iterrows():
        pct_ci = f"{row['pct_safe_mean']:.1f} [{row['pct_safe_ci_low']:.1f}, {row['pct_safe_ci_high']:.1f}]"
        evac_m = f"{row['mean_evac_min_mean']:.2f}" if np.isfinite(row['mean_evac_min_mean']) else "N/A"
        q_del = f"{row['mean_queue_delay_s_mean']:.1f}s" if np.isfinite(row['mean_queue_delay_s_mean']) else "N/A"
        cong = f"{row['max_congestion_mean']:.1f}x" if np.isfinite(row['max_congestion_mean']) else "N/A"
        over_s = f"{row['n_edges_over_storage_mean']:.1f}" if np.isfinite(row['n_edges_over_storage_mean']) else "N/A"
        print(
            f"{int(row['demand']):<8} "
            f"{row['router']:<16} "
            f"{pct_ci:<24} "
            f"{evac_m:<10} "
            f"{q_del:<18} "
            f"{cong:<14} "
            f"{over_s}"
        )
    print("=" * 105)

    print("\n" + "=" * 95)
    print("PAIRED DIFFERENCES IN pct_safe (PERCENTAGE POINTS)")
    print("=" * 95)
    print(
        f"{'Demand':<8} {'Comparison':<28} {'Mean Diff':<12} {'95% CI':<20} "
        f"{'p (t-test)':<14} {'p (wilcoxon)'}"
    )
    print("-" * 95)
    for _, row in paired_df.iterrows():
        ci_str = f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]" if np.isfinite(row['ci_low']) else "N/A"
        pt_str = f"{row['p_ttest']:.4f}" if np.isfinite(row['p_ttest']) else "N/A"
        pw_str = f"{row['p_wilcoxon']:.4f}" if np.isfinite(row['p_wilcoxon']) else "N/A"
        print(
            f"{int(row['demand']):<8} "
            f"{row['comparison']:<28} "
            f"{row['mean_diff']:<+12.2f} "
            f"{ci_str:<20} "
            f"{pt_str:<14} "
            f"{pw_str}"
        )
    print("=" * 95)

    # 3. Print Fallback Diagnostics Note
    fallback_runs = diag_df[diag_df["used_fallback_exits"].astype(bool)]
    print("\n" + "=" * 95)
    print("EXIT DIAGNOSTICS & FALLBACK SUMMARY")
    print("=" * 95)
    print(f"Total runs: {len(diag_df)}")
    print(f"Runs using fallback exits: {len(fallback_runs)}")
    if not fallback_runs.empty:
        for d in sorted(fallback_runs["demand"].unique()):
            sub = fallback_runs[fallback_runs["demand"] == d]
            print(f"  - Demand {d}: {len(sub)} runs used fallback exits (mean {sub['n_fallback_exits'].mean():.1f} fallback exits)")
    else:
        print("  - All runs successfully used unflooded sector-based exits without fallback.")
    print("=" * 95)

    # 4. Generate Figures
    plot_demand_sweep(summary_df, out_dir / "demand_sweep.png")

    # 5. Bottlenecks & Congestion Map
    print("\nExtracting bottleneck edges and generating congestion map...")
    G = get_or_build_graph(graph_cfg)
    ex_demand = cong_cfg.example_demand
    ex_seed = cong_cfg.example_seed_index

    # Check if example run exists
    ex_edge_csv = raw_dir / f"edges_demand{ex_demand}_seed{ex_seed}_td.csv"
    if ex_edge_csv.exists():
        edge_stats_df = pd.read_csv(ex_edge_csv)
    else:
        # If not run yet (e.g. quick mode with different demand), run or load first available
        avail_edges = list(raw_dir.glob("edges_demand*_seed*_td.csv"))
        if avail_edges:
            ex_edge_csv = avail_edges[-1]
            edge_stats_df = pd.read_csv(ex_edge_csv)
            # parse demand from filename
            tok = ex_edge_csv.stem.split("_")
            ex_demand = int(tok[1].replace("demand", ""))
            ex_seed = int(tok[2].replace("seed", ""))
        else:
            edge_stats_df = pd.DataFrame()

    if not edge_stats_df.empty:
        # Top 15 bottleneck edges by total_queue_delay_s
        top_15 = edge_stats_df.sort_values("total_queue_delay_s", ascending=False).head(15)
        top_15_clean = top_15[
            [
                "u",
                "v",
                "k",
                "highway",
                "lanes_n",
                "flow_capacity",
                "total_queue_delay_s",
                "n_vehicles",
                "peak_occupancy",
                "storage_capacity",
            ]
        ]
        bottlenecks_path = out_dir / "bottlenecks.csv"
        top_15_clean.to_csv(bottlenecks_path, index=False)
        print(f"Top 15 bottleneck edges saved to {bottlenecks_path}")
        print("\nTop 5 Bottleneck Edges by Queuing Delay:")
        for _, b_row in top_15_clean.head(5).iterrows():
            print(
                f"  Edge ({int(b_row['u'])}->{int(b_row['v'])}, k={int(b_row['k'])}): "
                f"hwy={b_row['highway']}, lanes={int(b_row['lanes_n'])}, "
                f"delay={b_row['total_queue_delay_s']:.1f}s, n_veh={int(b_row['n_vehicles'])}, "
                f"peak/cap={int(b_row['peak_occupancy'])}/{int(b_row['storage_capacity'])}"
            )

        # Generate Congestion Map
        h_cfg = replace(
            hazard_cfg,
            spread_speed_m_per_min=float(cong_cfg.flood_speed_m_per_min),
            seed=int(cong_cfg.base_hazard_seed + ex_seed),
            terrain_seed=int(hazard_cfg.terrain_seed),
        )
        hazard = FloodHazard(G, h_cfg)
        exits, _ = build_exits(
            G,
            hazard,
            scenario_cfg,
            center_lat=graph_cfg.center_lat,
            center_lon=graph_cfg.center_lon,
            horizon_min=h_cfg.horizon_min,
        )
        sources = getattr(hazard, "sources", [])
        plot_congestion_map(
            G=G,
            edge_stats_df=edge_stats_df,
            exits=list(exits),
            sources=list(sources),
            demand=ex_demand,
            seed_idx=ex_seed,
            speed=cong_cfg.flood_speed_m_per_min,
            out_path=out_dir / "congestion_map.png",
        )


if __name__ == "__main__":
    main()
