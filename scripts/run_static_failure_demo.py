#!/usr/bin/env python
"""CLI experiment: Static routing failure vs Reactive and Time-Dependent routing.

Usage
-----
    python scripts/run_static_failure_demo.py \
        --graph-config configs/study_area.yaml \
        --hazard-config configs/hazard.yaml \
        --scenario-config configs/scenario.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HazardConfig, ScenarioConfig, StudyAreaConfig, load_config
from src.graph.loader import get_or_build_graph
from src.hazard.flood import FloodHazard
from src.routing.base import RoutingParams
from src.routing.planners import ReactiveRouter, StaticRouter, TimeDependentRouter
from src.scenario import build_exits, generate_agents, shortest_baseline_length
from src.sim.single_agent import simulate_agent


# ---------------------------------------------------------------------------
# Visualisation Helpers
# ---------------------------------------------------------------------------

def _edge_lines(G):
    """Return a list of coordinate sequences and edge keys for plotting."""
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
# Plotting functions
# ---------------------------------------------------------------------------

def plot_failure_sweep(sweep_summary: pd.DataFrame, out_path: Path):
    """Plot pct_safe vs flood speed for the three routing methods."""
    fig, ax = plt.subplots(figsize=(8, 5))

    styles = {
        "static": {"color": "#e74c3c", "marker": "o", "label": "Static Dijkstra (snapshot)"},
        "reactive": {"color": "#e67e22", "marker": "s", "label": "Reactive Re-routing"},
        "time_dependent": {"color": "#2ecc71", "marker": "^", "label": "Time-Dependent Dijkstra (forecast)"},
    }

    speeds = sorted(sweep_summary["flood_speed_m_per_min"].unique())

    for router_name, style in styles.items():
        subset = sweep_summary[sweep_summary["router"] == router_name].sort_values(
            "flood_speed_m_per_min"
        )
        ax.plot(
            subset["flood_speed_m_per_min"],
            subset["pct_safe"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
            markersize=7,
            label=style["label"],
        )

    ax.set_xlabel("Flood Spread Speed (m/min)", fontsize=11)
    ax.set_ylabel("% Evacuees Safe (pct_safe)", fontsize=11)
    ax.set_title("Evacuation Success Under Increasing Hazard Spread Speed", fontsize=12)
    ax.set_ylim(-5, 105)
    ax.set_xticks(speeds)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sweep plot saved to {out_path}")


def plot_failure_example(
    G,
    hazard: FloodHazard,
    agent,
    t_trap: float,
    static_plan,
    td_plan,
    exits: Sequence[int],
    out_path: Path,
):
    """Draw road network and route comparison at the moment of failure."""
    lines, edge_keys = _edge_lines(G)
    extent = _graph_extent(G)

    fig, ax = plt.subplots(figsize=(12, 12), facecolor="#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    # 1. Base network colored by hazard state at t_trap
    colors = []
    for u, v, k in edge_keys:
        sf = hazard.edge_speed_factor(u, v, k, t_trap)
        if sf <= 0.0:
            colors.append("#ff3333")  # closed
        elif sf < 1.0:
            colors.append("#ff9933")  # degraded
        else:
            colors.append("#444444")  # open

    lc = LineCollection(lines, colors=colors, linewidths=0.6, zorder=1)
    ax.add_collection(lc)

    # 2. Static route (red dashed)
    if static_plan and static_plan.edges:
        s_coords = _route_coords(G, static_plan.edges)
        if s_coords:
            xs, ys = zip(*s_coords)
            ax.plot(
                xs,
                ys,
                linestyle="--",
                color="#ff4444",
                linewidth=2.5,
                zorder=4,
                label="Static Route (Trapped)",
            )

    # 3. Time-dependent route (green solid)
    if td_plan and td_plan.edges:
        td_coords = _route_coords(G, td_plan.edges)
        if td_coords:
            xs, ys = zip(*td_coords)
            ax.plot(
                xs,
                ys,
                linestyle="-",
                color="#00ff66",
                linewidth=2.5,
                zorder=5,
                label="Time-Dependent Route (Safe)",
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
            color="lime",
            markersize=10,
            zorder=6,
            label="Exit" if e == exits[0] else "",
        )

    # Origin
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

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_title(
        f"Static Routing Breakdown: Agent {agent.id} Trapped at t = {t_trap:.1f} s ({t_trap/60:.1f} min)",
        color="white",
        fontsize=13,
        pad=10,
    )
    ax.tick_params(colors="white", labelsize=6)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        loc="upper right",
        facecolor="#2a2a2a",
        edgecolor="#555555",
        labelcolor="white",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="#1a1a1a", bbox_inches="tight")
    plt.close(fig)
    print(f"Failure example saved to {out_path}")


# ---------------------------------------------------------------------------
# Main Experiment Loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Static routing failure experiment.")
    parser.add_argument(
        "--graph-config", type=str, default="configs/study_area.yaml"
    )
    parser.add_argument(
        "--hazard-config", type=str, default="configs/hazard.yaml"
    )
    parser.add_argument(
        "--scenario-config", type=str, default="configs/scenario.yaml"
    )
    args = parser.parse_args()

    graph_cfg = load_config(args.graph_config, StudyAreaConfig)
    base_hazard_cfg = load_config(args.hazard_config, HazardConfig)
    scenario_cfg = load_config(args.scenario_config, ScenarioConfig)

    print("Loading graph …")
    G = get_or_build_graph(graph_cfg)

    routers = [
        StaticRouter(),
        ReactiveRouter(),
        TimeDependentRouter(),
    ]
    routing_params = RoutingParams(
        vehicle_speed_scale=scenario_cfg.vehicle_speed_scale,
        sf_floor=scenario_cfg.sf_floor,
    )

    all_results: list[dict] = []
    summary_rows: list[dict] = []

    # Store experiment data per speed to choose an example failure
    speed_hazards: dict[float, FloodHazard] = {}
    speed_exits: dict[float, list[int]] = {}
    speed_agents: dict[float, list] = {}
    speed_agent_results: dict[float, dict[str, list]] = {}

    print(
        f"\nStarting flood speed sweep over {scenario_cfg.flood_speed_sweep_m_per_min} m/min …\n"
    )

    for speed in scenario_cfg.flood_speed_sweep_m_per_min:
        print(f"--- Flood spread speed: {speed} m/min ---")
        # Build modified hazard configuration
        h_cfg = replace(base_hazard_cfg, spread_speed_m_per_min=float(speed))
        hazard = FloodHazard(G, h_cfg)
        speed_hazards[speed] = hazard

        # Rebuild exits per flood speed (unflooded boundary nodes depend on flood speed)
        exits, _ = build_exits(
            G,
            hazard,
            scenario_cfg,
            center_lat=graph_cfg.center_lat,
            center_lon=graph_cfg.center_lon,
            horizon_min=h_cfg.horizon_min,
        )
        speed_exits[speed] = exits

        # Sample agents
        rng = np.random.default_rng(scenario_cfg.scenario_seed)
        agents = generate_agents(G, hazard, exits, scenario_cfg, rng)
        speed_agents[speed] = agents

        # Precompute static baseline shortest lengths
        baseline_lengths = shortest_baseline_length(G, exits)

        speed_agent_results[speed] = {}

        for router in routers:
            results_for_router = []
            for agent in agents:
                res = simulate_agent(G, hazard, router, agent, exits, routing_params)
                results_for_router.append(res)

                base_len = baseline_lengths.get(agent.origin, np.nan)
                overhead = (
                    res.path_length_m / base_len
                    if (res.safe and base_len > 0)
                    else np.nan
                )

                all_results.append(
                    {
                        "flood_speed_m_per_min": speed,
                        "router": router.name,
                        "agent_id": agent.id,
                        "origin": agent.origin,
                        "depart_time_s": agent.depart_time_s,
                        "safe": res.safe,
                        "trapped_reason": res.trapped_reason or "none",
                        "finish_time_s": res.finish_time_s,
                        "evac_time_s": res.evac_time_s,
                        "path_length_m": res.path_length_m,
                        "baseline_length_m": base_len,
                        "path_overhead": overhead,
                        "n_replans": res.n_replans,
                        "compute_ms": res.compute_ms,
                        "trap_time_s": res.trap_time_s,
                    }
                )

            speed_agent_results[speed][router.name] = results_for_router

            # Aggregate statistics
            n_total = len(agents)
            n_safe = sum(1 for r in results_for_router if r.safe)
            pct_safe = 100.0 * n_safe / n_total

            safe_evac_times_min = [
                r.evac_time_s / 60.0 for r in results_for_router if r.safe and r.evac_time_s is not None
            ]
            mean_evac_min = float(np.mean(safe_evac_times_min)) if safe_evac_times_min else np.nan

            safe_overheads = [
                r.path_length_m / baseline_lengths[agent.origin]
                for r, agent in zip(results_for_router, agents)
                if r.safe and baseline_lengths.get(agent.origin, 0) > 0
            ]
            mean_overhead = float(np.mean(safe_overheads)) if safe_overheads else np.nan

            mean_compute_ms = float(np.mean([r.compute_ms for r in results_for_router]))
            trapped_no_route = sum(1 for r in results_for_router if r.trapped_reason == "no_route")
            trapped_en_route = sum(
                1 for r in results_for_router if r.trapped_reason == "edge_closed_en_route"
            )

            summary_rows.append(
                {
                    "flood_speed_m_per_min": speed,
                    "router": router.name,
                    "pct_safe": pct_safe,
                    "mean_evac_min": mean_evac_min,
                    "mean_overhead": mean_overhead,
                    "mean_compute_ms": mean_compute_ms,
                    "trapped_no_route": trapped_no_route,
                    "trapped_en_route": trapped_en_route,
                }
            )

    # -----------------------------------------------------------------------
    # Print Results Table
    # -----------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 90)
    print("TASK 3 RESULTS: SOLVER LADDER COMPARISON ACROSS FLOOD SPEEDS")
    print("=" * 90)
    print(
        f"{'Speed (m/min)':<14} {'Router':<16} {'pct_safe':<10} {'Evac (min)':<12} "
        f"{'Overhead':<10} {'Compute (ms)':<14} {'Trapped (no/en-route)'}"
    )
    print("-" * 90)
    for _, row in summary_df.iterrows():
        evac_str = f"{row['mean_evac_min']:.2f}" if not np.isnan(row['mean_evac_min']) else "N/A"
        over_str = f"{row['mean_overhead']:.3f}" if not np.isnan(row['mean_overhead']) else "N/A"
        trapped_str = f"{row['trapped_no_route']} / {row['trapped_en_route']}"
        print(
            f"{row['flood_speed_m_per_min']:<14.0f} "
            f"{row['router']:<16} "
            f"{row['pct_safe']:<10.1f} "
            f"{evac_str:<12} "
            f"{over_str:<10} "
            f"{row['mean_compute_ms']:<14.2f} "
            f"{trapped_str}"
        )
    print("=" * 90)

    # Save CSV
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "task3_results.csv"
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"\nDetailed agent results saved to {csv_path}")

    # Plot sweep curve
    plot_failure_sweep(summary_df, out_dir / "failure_sweep.png")

    # -----------------------------------------------------------------------
    # Find Failure Example
    # -----------------------------------------------------------------------
    # Try example_flood_speed first, else try sweep speeds descending
    speeds_to_try = [scenario_cfg.example_flood_speed_m_per_min] + sorted(
        scenario_cfg.flood_speed_sweep_m_per_min, reverse=True
    )
    # Deduplicate while preserving order
    seen = set()
    speeds_order = [s for s in speeds_to_try if not (s in seen or seen.add(s))]

    example_found = False
    chosen_speed = None
    chosen_agent = None
    chosen_static_res = None
    chosen_td_res = None

    for s in speeds_order:
        static_res_list = speed_agent_results[s]["static"]
        td_res_list = speed_agent_results[s]["time_dependent"]
        agents_list = speed_agents[s]

        for ag, s_res, td_res in zip(agents_list, static_res_list, td_res_list):
            if (not s_res.safe) and td_res.safe:
                example_found = True
                chosen_speed = s
                chosen_agent = ag
                chosen_static_res = s_res
                chosen_td_res = td_res
                break
        if example_found:
            break

    if not example_found:
        print("\nWARNING: No agent found that was trapped under static but safe under time-dependent.")
    else:
        print(
            f"\nChosen Failure Example: Speed = {chosen_speed} m/min, Agent ID = {chosen_agent.id}, "
            f"Static Trapped Reason = '{chosen_static_res.trapped_reason}', Trap Time = {chosen_static_res.trap_time_s:.1f} s"
        )
        hazard_obj = speed_hazards[chosen_speed]
        exits_list = speed_exits[chosen_speed]

        # Re-plan routes from origin for visualization
        static_plan = StaticRouter().plan(
            G,
            hazard_obj,
            chosen_agent,
            chosen_agent.depart_time_s,
            chosen_agent.origin,
            exits_list,
            routing_params,
        )
        td_plan = TimeDependentRouter().plan(
            G,
            hazard_obj,
            chosen_agent,
            chosen_agent.depart_time_s,
            chosen_agent.origin,
            exits_list,
            routing_params,
        )

        t_trap = (
            chosen_static_res.trap_time_s
            if chosen_static_res.trap_time_s is not None
            else chosen_agent.depart_time_s
        )

        plot_failure_example(
            G=G,
            hazard=hazard_obj,
            agent=chosen_agent,
            t_trap=t_trap,
            static_plan=static_plan,
            td_plan=td_plan,
            exits=exits_list,
            out_path=out_dir / "failure_example.png",
        )


if __name__ == "__main__":
    main()

