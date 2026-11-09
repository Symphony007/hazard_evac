#!/usr/bin/env python
"""CLI: build flood hazard, print summary, save visualisations.

Usage
-----
    python scripts/run_hazard_demo.py \
        --graph-config configs/study_area.yaml \
        --hazard-config configs/hazard.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HazardConfig, load_config  # noqa: E402
from src.graph.loader import get_or_build_graph  # noqa: E402
from src.hazard.flood import FloodHazard  # noqa: E402


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _edge_lines(G):
    """Return a list of coordinate sequences (one per edge) for plotting."""
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


def _edge_lengths_arr(G, edge_keys):
    """Return array of edge lengths matching *edge_keys* order."""
    return np.array([G.edges[u, v, k]["length"] for u, v, k in edge_keys])


def _graph_extent(G, margin_frac: float = 0.02):
    """Bounding box (x_min, x_max, y_min, y_max) with margin."""
    xs = [d["x"] for _, d in G.nodes(data=True)]
    ys = [d["y"] for _, d in G.nodes(data=True)]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    mx = (x_max - x_min) * margin_frac
    my = (y_max - y_min) * margin_frac
    return x_min - mx, x_max + mx, y_min - my, y_max + my


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def plot_snapshots(G, hazard: FloodHazard, cfg: HazardConfig, out_path: Path):
    """2×3 grid of road maps at t = 0, 10, 20, 30, 45, 60 min."""
    times_min = [0, 10, 20, 30, 45, 60]
    lines, edge_keys = _edge_lines(G)
    extent = _graph_extent(G)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="k")
    for ax, t_min in zip(axes.flat, times_min):
        t_sec = t_min * 60.0
        ax.set_facecolor("k")
        ax.set_title(f"t = {t_min} min", color="white", fontsize=13, pad=6)

        colors = []
        for u, v, k in edge_keys:
            sf = hazard.edge_speed_factor(u, v, k, t_sec)
            if sf <= 0.0:
                colors.append("#ff3333")   # closed
            elif sf < 1.0:
                colors.append("#ff9933")   # degraded
            else:
                colors.append("#777777")   # open

        lc = LineCollection(lines, colors=colors, linewidths=0.7)
        ax.add_collection(lc)

        # Source markers
        for s in hazard.sources:
            ax.plot(
                G.nodes[s]["x"], G.nodes[s]["y"],
                "*", color="cyan", markersize=10, zorder=5,
            )

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.tick_params(colors="white", labelsize=5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=150, facecolor="k", bbox_inches="tight")
    plt.close(fig)
    print(f"Snapshots saved to {out_path}")


def plot_hazard_curve(G, hazard: FloodHazard, cfg: HazardConfig, out_path: Path):
    """% of road length closed / degraded-or-closed vs time."""
    lines, edge_keys = _edge_lines(G)
    edge_lengths = _edge_lengths_arr(G, edge_keys)
    total_length = edge_lengths.sum()

    edge_arrivals = np.array(
        [hazard.edge_arrival_time(u, v, k) for u, v, k in edge_keys]
    )
    ramp_s = cfg.ramp_min * 60.0

    n_pts = 300
    times_min = np.linspace(0, cfg.horizon_min, n_pts)
    times_sec = times_min * 60.0

    pct_closed = np.zeros(n_pts)
    pct_affected = np.zeros(n_pts)
    for i, t in enumerate(times_sec):
        closed_mask = edge_arrivals <= t
        affected_mask = (edge_arrivals - ramp_s) <= t
        pct_closed[i] = 100.0 * edge_lengths[closed_mask].sum() / total_length
        pct_affected[i] = 100.0 * edge_lengths[affected_mask].sum() / total_length

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times_min, pct_affected, color="orange", linewidth=2,
            label="Degraded or closed")
    ax.plot(times_min, pct_closed, color="red", linewidth=2,
            label="Closed")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("% of total road length")
    ax.set_title("Flood spread over time")
    ax.legend()
    ax.set_xlim(0, cfg.horizon_min)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Hazard curve saved to {out_path}")


def plot_terrain(G, hazard: FloodHazard, out_path: Path):
    """Node elevation map."""
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="k")
    ax.set_facecolor("k")

    # Draw edges as thin dark lines for context
    lines, _ = _edge_lines(G)
    lc = LineCollection(lines, colors="#333333", linewidths=0.4)
    ax.add_collection(lc)

    # Scatter nodes coloured by elevation
    nodes = list(G.nodes)
    xs = [G.nodes[n]["x"] for n in nodes]
    ys = [G.nodes[n]["y"] for n in nodes]
    es = [hazard.elevation[n] for n in nodes]
    sc = ax.scatter(xs, ys, c=es, cmap="terrain", s=5, zorder=2)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Elevation (m)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Mark sources
    for s in hazard.sources:
        ax.plot(
            G.nodes[s]["x"], G.nodes[s]["y"],
            "*", color="cyan", markersize=12, zorder=5,
        )

    extent = _graph_extent(G)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_title("Synthetic Terrain (sources = cyan stars)", color="white", fontsize=13)
    ax.tick_params(colors="white", labelsize=6)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="k", bbox_inches="tight")
    plt.close(fig)
    print(f"Terrain plot saved to {out_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Flood hazard demo.")
    parser.add_argument(
        "--graph-config", type=str, default="configs/study_area.yaml",
        help="Path to the study-area YAML config.",
    )
    parser.add_argument(
        "--hazard-config", type=str, default="configs/hazard.yaml",
        help="Path to the hazard YAML config.",
    )
    args = parser.parse_args()

    # --- Load graph ---
    graph_cfg = load_config(args.graph_config)
    G = get_or_build_graph(graph_cfg)

    # --- Build hazard ---
    hazard_cfg = load_config(args.hazard_config, HazardConfig)
    print("Building flood hazard …")
    hazard = FloodHazard(G, hazard_cfg)

    # --- Print summary ---
    print("\n" + "=" * 55)
    print("FLOOD HAZARD SUMMARY")
    print("=" * 55)

    print(f"\nSource nodes ({len(hazard.sources)}):")
    for s in hazard.sources:
        lat, lon = G.nodes[s]["y"], G.nodes[s]["x"]
        elev = hazard.elevation[s]
        print(f"  Node {s}  lat={lat:.4f}  lon={lon:.4f}  elev={elev:.2f} m")

    # Edge arrival stats
    edge_keys = list(G.edges(keys=True))
    edge_arrivals = np.array(
        [hazard.edge_arrival_time(u, v, k) for u, v, k in edge_keys]
    )
    finite_arrivals = edge_arrivals[np.isfinite(edge_arrivals)]

    if len(finite_arrivals) > 0:
        first_closure_min = finite_arrivals.min() / 60.0
        print(f"\nFirst edge closure   : {first_closure_min:.2f} min")
    else:
        first_closure_min = float("inf")
        print("\nNo edges are closed within the simulation.")

    # % road length closed at milestones
    edge_lengths = np.array([G.edges[u, v, k]["length"] for u, v, k in edge_keys])
    total_length = edge_lengths.sum()

    print("\nRoad length closed:")
    for t_min in [10, 20, 30, 45, 60]:
        t_sec = t_min * 60.0
        closed_mask = edge_arrivals <= t_sec
        pct = 100.0 * edge_lengths[closed_mask].sum() / total_length
        print(f"  t = {t_min:2d} min : {pct:5.1f}%")

    print("=" * 55)

    # --- Plots ---
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    plot_snapshots(G, hazard, hazard_cfg, out_dir / "hazard_snapshots.png")
    plot_hazard_curve(G, hazard, hazard_cfg, out_dir / "hazard_curve.png")
    plot_terrain(G, hazard, out_dir / "terrain.png")


if __name__ == "__main__":
    main()

