#!/usr/bin/env python
"""CLI script: download, enrich, cache the study-area road graph.

Usage
-----
    python scripts/build_graph.py --config configs/study_area.yaml
    python scripts/build_graph.py --config configs/study_area.yaml --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path so `src` is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import osmnx as ox  # noqa: E402

from src.config import load_config  # noqa: E402
from src.graph.loader import build_graph, get_or_build_graph, save_graph  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the study-area road graph.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/study_area.yaml",
        help="Path to the study-area YAML config.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore cache and rebuild the graph from OSM.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Study area : {cfg.name}")
    print(f"Center     : ({cfg.center_lat}, {cfg.center_lon})")
    print(f"Radius     : {cfg.dist_m} m\n")

    if args.rebuild:
        print("[build] Rebuilding graph (ignoring cache) …")
        G = build_graph(cfg)
        saved = save_graph(G, cfg.cache_path)
        print(f"[build] Graph saved to {saved}")
    else:
        G = get_or_build_graph(cfg)

    # ---- Statistics ----
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    lengths = np.array([d["length"] for _, _, d in G.edges(data=True)])
    capacities = np.array([d["capacity"] for _, _, d in G.edges(data=True)])
    travel_times = np.array([d["travel_time"] for _, _, d in G.edges(data=True)])
    flow_caps = np.array([d["flow_capacity"] for _, _, d in G.edges(data=True)])

    total_road_km = lengths.sum() / 1000.0

    print("=" * 50)
    print("GRAPH STATISTICS")
    print("=" * 50)
    print(f"Nodes             : {n_nodes}")
    print(f"Edges             : {n_edges}")
    print(f"Total road length : {total_road_km:.2f} km")
    print(f"Capacity  mean    : {capacities.mean():.1f}")
    print(f"Capacity  median  : {np.median(capacities):.1f}")
    print(f"Travel-time min   : {travel_times.min():.2f} s")
    print(f"Travel-time max   : {travel_times.max():.2f} s")
    print(f"Flow-cap   mean   : {flow_caps.mean():.1f} veh/min")
    print(f"Flow-cap   median : {np.median(flow_caps):.1f} veh/min")
    print(f"Flow-cap   min    : {flow_caps.min():.1f} veh/min")
    print(f"Flow-cap   max    : {flow_caps.max():.1f} veh/min")
    print("=" * 50)

    # ---- Plot ----
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "graph_overview.png"

    fig, ax = ox.plot_graph(G, show=False, close=True, save=False, dpi=150)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
