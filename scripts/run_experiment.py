#!/usr/bin/env python
"""CLI multi-seed experiment runner and statistical analyzer.

Usage
-----
    # Dry run benchmark (1 run, speed=60, seed=0)
    python scripts/run_experiment.py --dry-run

    # Quick test (3 seeds, 3 speeds: 30, 60, 90)
    python scripts/run_experiment.py --quick

    # Full experiment
    python scripts/run_experiment.py

    # Re-analyze existing outputs without re-running simulations
    python scripts/run_experiment.py --analyze-only
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig, HazardConfig, ScenarioConfig, StudyAreaConfig, load_config
from src.experiment.runner import _speed_str, run_experiment, run_single
from src.graph.loader import get_or_build_graph
from src.hazard.flood import FloodHazard
from src.metrics.stats import mean_ci, paired_diff_ci
from src.routing.base import RoutingParams
from src.routing.planners import StaticRouter, TimeDependentRouter
from src.scenario import build_exits, exit_spread_report, generate_agents, shortest_baseline_length
from src.sim.single_agent import simulate_agent


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


def _route_coords(G, route_edges):
    """Extract coordinates sequence for a planned route."""
    coords = []
    for u, v, k in route_edges:
        data = G.edges[u, v, k]
        if "geometry" in data:
            c = list(data["geometry"].coords)
        else:
            c = [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
        if not coords:
            coords.extend(c)
        else:
            coords.extend(c[1:])
    return coords


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

def plot_sweep_ci(
    summary_df: pd.DataFrame,
    exp_cfg: ExperimentConfig,
    out_path: Path,
):
    """Plot pct_safe vs flood speed with shaded 95% CI bands and stress zone."""
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.set_facecolor("white")

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
    }

    all_speeds = sorted(summary_df["speed"].unique())
    stress_speeds = sorted(exp_cfg.stress_speeds_m_per_min)

    # Shaded stress region if stress speeds are present
    stress_present = [s for s in all_speeds if s in stress_speeds]
    if stress_present:
        stress_min = min(stress_present)
        # Extend shaded span slightly before stress_min
        prev_speeds = [s for s in all_speeds if s < stress_min]
        x_start = (max(prev_speeds) + stress_min) / 2.0 if prev_speeds else (stress_min - 10)
        x_end = max(all_speeds) + 15
        ax.axvspan(
            x_start,
            x_end,
            color="#eeeeee",
            alpha=0.7,
            zorder=0,
            label="Stress Regime (boundary exits flood)",
        )

    for router_name, style in styles.items():
        subset = summary_df[summary_df["router"] == router_name].sort_values("speed")
        if subset.empty:
            continue

        spds = subset["speed"].values
        means = subset["pct_safe_mean"].values
        lows = subset["pct_safe_ci_low"].values
        highs = subset["pct_safe_ci_high"].values

        # If CIs are NaN (e.g. single run), fallback to mean
        lows_clean = np.where(np.isfinite(lows), lows, means)
        highs_clean = np.where(np.isfinite(highs), highs, means)

        # Clip CI display bands and means to [0, 100] for display only
        lows_clean = np.clip(lows_clean, 0.0, 100.0)
        highs_clean = np.clip(highs_clean, 0.0, 100.0)
        means_clean = np.clip(means, 0.0, 100.0)

        ax.plot(
            spds,
            means_clean,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=6,
            zorder=3,
            label=style["label"],
        )
        ax.fill_between(
            spds,
            lows_clean,
            highs_clean,
            color=style["color"],
            alpha=0.18,
            zorder=2,
        )

    ax.set_xlabel("Flood Spread Speed (m/min)", fontsize=11)
    ax.set_ylabel("% Evacuees Safe (pct_safe)", fontsize=11)
    ax.set_title("Evacuation Success vs Hazard Spread Speed (Mean ± 95% CI)", fontsize=12)
    ax.set_ylim(-2, 102)
    ax.set_xticks(all_speeds)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(frameon=True, loc="lower left", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sweep CI plot saved to {out_path}")


def plot_gain_over_static(
    paired_df: pd.DataFrame,
    out_path: Path,
):
    """Plot paired percentage point difference in pct_safe over Static."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    ax.set_facecolor("white")

    pct_pairs = paired_df[paired_df["metric"] == "pct_safe"].copy()

    comparisons = {
        "reactive - static": {
            "color": "#f0ad4e",
            "marker": "s",
            "label": "Reactive − Static",
        },
        "time_dependent - static": {
            "color": "#5cb85c",
            "marker": "^",
            "label": "Time-Dependent − Static",
        },
    }

    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.2, zorder=1)

    for comp_name, style in comparisons.items():
        subset = pct_pairs[pct_pairs["comparison"] == comp_name].sort_values("speed")
        if subset.empty:
            continue

        spds = subset["speed"].values
        means = subset["mean_diff"].values
        lows = subset["ci_low"].values
        highs = subset["ci_high"].values

        # Asymmetric error bars clipped to valid display bounds [-100, 100]
        lows_disp = np.clip(lows, -100.0, 100.0)
        highs_disp = np.clip(highs, -100.0, 100.0)
        yerr_lower = np.where(np.isfinite(lows_disp), means - lows_disp, 0.0)
        yerr_upper = np.where(np.isfinite(highs_disp), highs_disp - means, 0.0)
        # Ensure non-negative error bar extents for display
        yerr_lower = np.maximum(yerr_lower, 0.0)
        yerr_upper = np.maximum(yerr_upper, 0.0)
        yerr = [yerr_lower, yerr_upper]

        ax.errorbar(
            spds,
            means,
            yerr=yerr,
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
            capsize=4,
            markersize=6,
            zorder=3,
            label=style["label"],
        )

    all_speeds = sorted(pct_pairs["speed"].unique())
    ax.set_xlabel("Flood Spread Speed (m/min)", fontsize=11)
    ax.set_ylabel("Gain in pct_safe (percentage points)", fontsize=11)
    ax.set_title("Safety Gain Over Static Dijkstra (Mean ± 95% CI)", fontsize=12)
    ax.set_xticks(all_speeds)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(frameon=True, loc="upper right", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gain plot saved to {out_path}")


def plot_failure_example(
    G,
    hazard: FloodHazard,
    agent,
    t_trap: float,
    trap_node: int | str | None,
    static_plan,
    td_plan,
    exits: Sequence[int],
    spread_info: dict,
    out_path: Path,
):
    """Draw dark-mode road map comparing static route vs time-dependent route."""
    lines, edge_keys = _edge_lines(G)
    extent = _graph_extent(G)

    fig, ax = plt.subplots(figsize=(12, 12), facecolor="#141414")
    ax.set_facecolor("#141414")

    # 1. Base network colored by hazard state at t_trap
    colors = []
    for u, v, k in edge_keys:
        sf = hazard.edge_speed_factor(u, v, k, t_trap)
        if sf <= 0.0:
            colors.append("#ff3333")  # closed
        elif sf < 1.0:
            colors.append("#ff9933")  # degraded
        else:
            colors.append("#383838")  # open

    lc = LineCollection(lines, colors=colors, linewidths=0.6, zorder=1)
    ax.add_collection(lc)

    # 2. Time-dependent route (drawn FIRST, green solid)
    if td_plan and td_plan.edges:
        td_coords = _route_coords(G, td_plan.edges)
        if td_coords:
            xs, ys = zip(*td_coords)
            ax.plot(
                xs,
                ys,
                linestyle="-",
                color="#00ff66",
                linewidth=3.0,
                zorder=3,
                label="Time-Dependent Route (Safe)",
            )

    # 3. Static route (drawn ON TOP with thicker light underlay so overlaps remain visible)
    if static_plan and static_plan.edges:
        s_coords = _route_coords(G, static_plan.edges)
        if s_coords:
            xs, ys = zip(*s_coords)
            # Thicker white underlay
            ax.plot(
                xs,
                ys,
                linestyle="-",
                color="#ffffff",
                linewidth=4.5,
                alpha=0.6,
                zorder=4,
            )
            # Dashed red line
            ax.plot(
                xs,
                ys,
                linestyle="--",
                color="#ff2222",
                linewidth=2.5,
                zorder=5,
                label="Static Route (Trapped)",
            )

    # 4. Markers
    # Flood sources
    for s in hazard.sources:
        ax.plot(
            G.nodes[s]["x"],
            G.nodes[s]["y"],
            "*",
            color="cyan",
            markersize=14,
            zorder=6,
            label="Flood Source" if s == hazard.sources[0] else "",
        )

    # Exits
    for e in exits:
        ax.plot(
            G.nodes[e]["x"],
            G.nodes[e]["y"],
            "^",
            color="#66ff66",
            markersize=10,
            zorder=6,
            label="Safe Exit" if e == exits[0] else "",
        )

    # Agent Origin
    ax.plot(
        G.nodes[agent.origin]["x"],
        G.nodes[agent.origin]["y"],
        "o",
        color="white",
        markersize=9,
        markeredgecolor="black",
        zorder=7,
        label="Agent Origin",
    )

    # Point of trapping (large red X)
    if trap_node is not None and trap_node in G.nodes:
        tx, ty = G.nodes[trap_node]["x"], G.nodes[trap_node]["y"]
        ax.plot(
            tx,
            ty,
            "x",
            color="#ff0000",
            markersize=15,
            markeredgewidth=3.5,
            zorder=8,
            label=f"Trapped at t = {t_trap:.1f}s",
        )
        ax.annotate(
            f"Trapped at t={t_trap:.1f}s\n({t_trap/60:.1f} min)",
            xy=(tx, ty),
            xytext=(tx + 0.001, ty + 0.001),
            color="#ffcccc",
            fontsize=9,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#330000", ec="#ff3333", alpha=0.8),
            zorder=9,
        )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")

    # Titles & Subtitles
    sec_covered = spread_info.get("n_distinct_sectors", 0)
    cov_deg = spread_info.get("angular_coverage_deg", 0.0)
    ax.set_title(
        f"Static Routing Failure: Agent {agent.id} (Depart t={agent.depart_time_s:.1f}s)\n"
        f"Exit Spread: {sec_covered}/8 sectors covered, angular coverage = {cov_deg:.1f}°",
        color="white",
        fontsize=12,
        pad=10,
    )
    ax.tick_params(colors="white", labelsize=6)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        loc="upper right",
        facecolor="#222222",
        edgecolor="#555555",
        labelcolor="white",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="#141414", bbox_inches="tight")
    plt.close(fig)
    print(f"Failure example saved to {out_path}")


# ---------------------------------------------------------------------------
# Aggregation & Analysis
# ---------------------------------------------------------------------------

def aggregate_and_report(
    raw_records: list[dict[str, Any]],
    exp_cfg: ExperimentConfig,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate raw experiment JSON records and write summary CSVs."""
    stress_speeds = set(exp_cfg.stress_speeds_m_per_min)

    # 1. Diagnostics DataFrame
    diag_rows = []
    for r in raw_records:
        spread = r.get("exit_spread", {})
        diag_rows.append(
            {
                "speed": float(r["speed"]),
                "seed_index": int(r["seed_index"]),
                "hazard_seed": int(r["hazard_seed"]),
                "scenario_seed": int(r["scenario_seed"]),
                "used_fallback_exits": bool(r["used_fallback_exits"]),
                "n_fallback_exits": int(r["n_fallback_exits"]),
                "n_exits_effective": int(r.get("n_exits_effective", len(r.get("exit_spread", {})))),
                "sectors_with_exit_count": int(r.get("sectors_with_exit_count", 0)),
                "sectors_dropped_count": int(r.get("sectors_dropped_count", 0)),
                "departure_window_capped": bool(r["departure_window_capped"]),
                "n_agents": int(r["n_agents"]),
                "pairwise_mean_dist_m": float(spread.get("pairwise_mean_dist_m", np.nan)),
                "pairwise_min_dist_m": float(spread.get("pairwise_min_dist_m", np.nan)),
                "angular_coverage_deg": float(spread.get("angular_coverage_deg", np.nan)),
                "largest_gap_deg": float(spread.get("largest_gap_deg", np.nan)),
                "n_distinct_sectors": int(spread.get("n_distinct_sectors", 0)),
            }
        )
    diag_df = pd.DataFrame(diag_rows).sort_values(["speed", "seed_index"])
    diag_df.to_csv(out_dir / "diagnostics.csv", index=False)

    # 2. Summary DataFrame per (speed, router)
    speeds = sorted({r["speed"] for r in raw_records})
    routers = ["static", "reactive", "time_dependent"]
    metric_keys = [
        "pct_safe",
        "mean_evac_min",
        "p90_evac_min",
        "last_evac_min",
        "mean_path_overhead",
        "mean_compute_ms",
        "n_trapped_no_route",
        "n_trapped_en_route",
        "n_agents",
    ]

    summary_rows = []
    for s in speeds:
        regime = "stress" if s in stress_speeds else "main"
        subset_records = [r for r in raw_records if r["speed"] == s]
        for router in routers:
            row: dict[str, Any] = {
                "speed": s,
                "router": router,
                "regime": regime,
                "n_runs": len(subset_records),
            }
            for m in metric_keys:
                vals = [
                    r["metrics"][router][m]
                    for r in subset_records
                    if router in r["metrics"] and r["metrics"][router][m] is not None
                ]
                m_mean, c_low, c_high = mean_ci(vals, level=exp_cfg.ci_level)
                row[f"{m}_mean"] = m_mean
                row[f"{m}_ci_low"] = c_low
                row[f"{m}_ci_high"] = c_high
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    # 3. Paired Differences per speed
    comparisons = [
        ("reactive - static", "reactive", "static"),
        ("time_dependent - static", "time_dependent", "static"),
        ("time_dependent - reactive", "time_dependent", "reactive"),
    ]
    test_metrics = ["pct_safe", "mean_evac_min", "mean_compute_ms"]

    paired_rows = []
    for s in speeds:
        subset = [r for r in raw_records if r["speed"] == s]
        for comp_name, r_a, r_b in comparisons:
            for m in test_metrics:
                a_vals = [
                    r["metrics"][r_a][m]
                    for r in subset
                    if r_a in r["metrics"] and r["metrics"][r_a][m] is not None
                ]
                b_vals = [
                    r["metrics"][r_b][m]
                    for r in subset
                    if r_b in r["metrics"] and r["metrics"][r_b][m] is not None
                ]
                diff_res = paired_diff_ci(a_vals, b_vals, level=exp_cfg.ci_level)
                paired_rows.append(
                    {
                        "speed": s,
                        "comparison": comp_name,
                        "metric": m,
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


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed evacuation experiment runner.")
    parser.add_argument("--graph-config", type=str, default="configs/study_area.yaml")
    parser.add_argument("--hazard-config", type=str, default="configs/hazard.yaml")
    parser.add_argument("--scenario-config", type=str, default="configs/scenario.yaml")
    parser.add_argument("--experiment-config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--quick", action="store_true", help="Quick run (3 seeds, 3 speeds: 30, 60, 90).")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 trial (speed=60, seed=0) and print timing ETA.")
    parser.add_argument("--force", action="store_true", help="Force recomputation, ignoring resume cache.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes.")
    parser.add_argument("--analyze-only", action="store_true", help="Skip simulation, re-aggregate existing raw data.")
    args = parser.parse_args()

    graph_cfg = load_config(args.graph_config, StudyAreaConfig)
    hazard_cfg = load_config(args.hazard_config, HazardConfig)
    scenario_cfg = load_config(args.scenario_config, ScenarioConfig)
    exp_cfg = load_config(args.experiment_config, ExperimentConfig)

    out_dir = Path(exp_cfg.output_dir)
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run simulation or handle dry-run
    if not args.analyze_only:
        records = run_experiment(
            graph_cfg=graph_cfg,
            hazard_cfg=hazard_cfg,
            scenario_cfg=scenario_cfg,
            exp_cfg=exp_cfg,
            quick=args.quick,
            dry_run=args.dry_run,
            force=args.force,
            workers=args.workers,
        )
        if args.dry_run:
            return
    else:
        print("[experiment] Analyze-only mode: loading raw JSON records from disk...")
        records = []
        for path in sorted(raw_dir.glob("run_*.json")):
            with open(path, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        print(f"[experiment] Loaded {len(records)} raw run records.")

    if not records:
        print("ERROR: No run records found.")
        return

    # 2. Aggregation
    summary_df, paired_df, diag_df = aggregate_and_report(records, exp_cfg, out_dir)

    # 3. Print Results Tables
    print("\n" + "=" * 95)
    print("TASK 4 SUMMARY: MULTI-SEED METRICS ACROSS FLOOD SPEEDS")
    print("=" * 95)
    print(
        f"{'Speed':<8} {'Regime':<8} {'Router':<16} {'pct_safe (95% CI)':<24} "
        f"{'Evac min (95% CI)':<22} {'ms/agent':<12} {'Runs'}"
    )
    print("-" * 95)
    for _, row in summary_df.iterrows():
        pct_ci = f"{row['pct_safe_mean']:.1f} [{row['pct_safe_ci_low']:.1f}, {row['pct_safe_ci_high']:.1f}]"
        evac_mean = row["mean_evac_min_mean"]
        if np.isfinite(evac_mean):
            evac_ci = f"{evac_mean:.2f} [{row['mean_evac_min_ci_low']:.2f}, {row['mean_evac_min_ci_high']:.2f}]"
        else:
            evac_ci = "N/A"
        print(
            f"{row['speed']:<8.0f} "
            f"{row['regime']:<8} "
            f"{row['router']:<16} "
            f"{pct_ci:<24} "
            f"{evac_ci:<22} "
            f"{row['mean_compute_ms_mean']:<12.1f} "
            f"{int(row['n_runs'])}"
        )
    print("=" * 95)

    print("\n" + "=" * 95)
    print("PAIRED DIFFERENCES IN pct_safe (PERCENTAGE POINTS)")
    print("=" * 95)
    print(
        f"{'Speed':<8} {'Comparison':<28} {'Mean Diff':<12} {'95% CI':<20} "
        f"{'p (t-test)':<14} {'p (wilcoxon)':<14} {'N'}"
    )
    print("-" * 95)
    pct_diffs = paired_df[paired_df["metric"] == "pct_safe"]
    for _, row in pct_diffs.iterrows():
        ci_str = f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]" if np.isfinite(row['ci_low']) else "N/A"
        pt_str = f"{row['p_ttest']:.4f}" if np.isfinite(row['p_ttest']) else "N/A"
        pw_str = f"{row['p_wilcoxon']:.4f}" if np.isfinite(row['p_wilcoxon']) else "N/A"
        print(
            f"{row['speed']:<8.0f} "
            f"{row['comparison']:<28} "
            f"{row['mean_diff']:<+12.2f} "
            f"{ci_str:<20} "
            f"{pt_str:<14} "
            f"{pw_str:<14} "
            f"{int(row['n'])}"
        )
    print("=" * 95)

    # 4. Print Diagnostics List
    fallback_runs = diag_df[diag_df["used_fallback_exits"].astype(bool)]
    capped_runs = diag_df[diag_df["departure_window_capped"].astype(bool)]
    print("\n" + "=" * 95)
    print("DIAGNOSTICS & ANOMALY SUMMARY")
    print("=" * 95)
    print(f"Total runs analyzed: {len(diag_df)}")
    print(f"Runs using fallback exits: {len(fallback_runs)}")
    if not fallback_runs.empty:
        fb_speeds = [float(s) for s in sorted(fallback_runs["speed"].unique())]
        print(f"  Fallback exits occurred at speeds: {fb_speeds}")
        for s in fb_speeds:
            sub = fallback_runs[fallback_runs["speed"] == s]
            print(f"    - Speed {s:.0f} m/min: {len(sub)} runs (mean {sub['n_fallback_exits'].mean():.1f} fallback exits)")

    print(f"Runs with departure window capped: {len(capped_runs)}")
    if not capped_runs.empty:
        cap_speeds = [float(s) for s in sorted(capped_runs["speed"].unique())]
        print(f"  Departure window capping occurred at speeds: {cap_speeds}")
        for s in cap_speeds:
            sub = capped_runs[capped_runs["speed"] == s]
            print(f"    - Speed {s:.0f} m/min: {len(sub)} runs")

    # Exit spread metrics
    print("\nMean Exit Spread Metrics Across Speeds:")
    for s in sorted(diag_df["speed"].unique()):
        sub = diag_df[diag_df["speed"] == s]
        print(
            f"  Speed {s:3.0f} m/min: "
            f"exits = {sub['n_exits_effective'].mean():.1f}, "
            f"sectors = {sub['n_distinct_sectors'].mean():.1f}/8, "
            f"coverage = {sub['angular_coverage_deg'].mean():.1f}°, "
            f"pairwise min dist = {sub['pairwise_min_dist_m'].mean():.0f}m"
        )
    print("=" * 95)

    # 5. Generate Figures
    plot_sweep_ci(summary_df, exp_cfg, out_dir / "sweep_ci.png")
    plot_gain_over_static(paired_df, out_dir / "gain_over_static.png")

    # 6. Generate Failure Example Figure
    print("\nGenerating failure example figure...")
    G = get_or_build_graph(graph_cfg)
    ex_speed = exp_cfg.example_speed_m_per_min
    ex_seed_idx = exp_cfg.example_seed_index

    # Load raw CSV for example run
    ex_token = f"speed{_speed_str(ex_speed)}_seed{ex_seed_idx}"
    ex_csv = raw_dir / f"agents_{ex_token}.csv"
    ex_json = raw_dir / f"run_{ex_token}.json"

    if ex_csv.exists() and ex_json.exists():
        with open(ex_json, "r", encoding="utf-8") as f:
            ex_run_data = json.load(f)
        df_ex = pd.read_csv(ex_csv)

        # Match static and time_dependent agents
        static_df = df_ex[df_ex["router_name"] == "static"].set_index("agent_id")
        td_df = df_ex[df_ex["router_name"] == "time_dependent"].set_index("agent_id")

        chosen_agent_id = None
        for aid in sorted(static_df.index):
            if (not static_df.loc[aid, "safe"]) and td_df.loc[aid, "safe"]:
                chosen_agent_id = aid
                break

        # Reconstruct scenario to get coordinates and planned paths
        h_cfg = replace(
            hazard_cfg,
            spread_speed_m_per_min=float(ex_speed),
            seed=int(exp_cfg.base_hazard_seed + ex_seed_idx),
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
        rng = np.random.default_rng(int(exp_cfg.base_scenario_seed + ex_seed_idx))
        agents = generate_agents(G, hazard, exits, scenario_cfg, rng)
        params = RoutingParams(
            vehicle_speed_scale=scenario_cfg.vehicle_speed_scale,
            sf_floor=scenario_cfg.sf_floor,
        )

        if chosen_agent_id is not None:
            agent = next(a for a in agents if a.id == chosen_agent_id)
            s_row = static_df.loc[chosen_agent_id]
            t_trap = float(s_row["trap_time_s"]) if np.isfinite(s_row["trap_time_s"]) else float(s_row["depart_time_s"])

            # Trace static agent to find trap node
            static_res = simulate_agent(G, hazard, StaticRouter(), agent, exits, params)
            trap_node = static_res.path_nodes[-1] if static_res.path_nodes else agent.origin

            static_plan = StaticRouter().plan(G, hazard, agent, agent.depart_time_s, agent.origin, exits, params)
            td_plan = TimeDependentRouter().plan(G, hazard, agent, agent.depart_time_s, agent.origin, exits, params)

            plot_failure_example(
                G=G,
                hazard=hazard,
                agent=agent,
                t_trap=t_trap,
                trap_node=trap_node,
                static_plan=static_plan,
                td_plan=td_plan,
                exits=exits,
                spread_info=ex_run_data.get("exit_spread", {}),
                out_path=out_dir / "failure_example.png",
            )
        else:
            print(f"[warning] No agent at speed {ex_speed} was trapped under static but safe under time-dependent.")
    else:
        print(f"[warning] Example files {ex_csv} not found; skipping failure example plot.")


if __name__ == "__main__":
    main()
