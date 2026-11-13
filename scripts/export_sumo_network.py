#!/usr/bin/env python
"""Export road network graph to SUMO format and validate topology.

Usage:
    python scripts/export_sumo_network.py [--graph-config configs/study_area.yaml]
                                          [--scenario-config configs/scenario.yaml]
                                          [--sumo-config configs/sumo.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ScenarioConfig, StudyAreaConfig, SumoConfig, load_config
from src.graph.loader import get_or_build_graph
from src.sumo.network import export_network_to_sumo, validate_sumo_network


def main() -> None:
    parser = argparse.ArgumentParser(description="Export road network to SUMO format.")
    parser.add_argument("--graph-config", type=str, default="configs/study_area.yaml")
    parser.add_argument("--scenario-config", type=str, default="configs/scenario.yaml")
    parser.add_argument("--sumo-config", type=str, default="configs/sumo.yaml")
    args = parser.parse_args()

    graph_cfg = load_config(PROJECT_ROOT / args.graph_config, StudyAreaConfig)
    scenario_cfg = load_config(PROJECT_ROOT / args.scenario_config, ScenarioConfig)
    sumo_cfg = load_config(PROJECT_ROOT / args.sumo_config, SumoConfig)

    out_dir = PROJECT_ROOT / sumo_cfg.sumo_dir

    print("=" * 70)
    print("TASK 8a: SUMO NETWORK EXPORT & VALIDATION")
    print("=" * 70)
    print(f"Graph cache:       {graph_cfg.cache_path}")
    print(f"Output directory:  {out_dir}")
    print(f"Speed scale:       {scenario_cfg.vehicle_speed_scale} (urban crawl)")
    print("-" * 70)

    # 1. Load graph
    G = get_or_build_graph(graph_cfg)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 2. Export plain XML and compile via netconvert
    res = export_network_to_sumo(
        G=G,
        output_dir=out_dir,
        vehicle_speed_scale=scenario_cfg.vehicle_speed_scale,
    )
    print("Exported XML files and compiled SUMO network successfully:")
    print(f"  Nodes XML: {res['nod_path']}")
    print(f"  Edges XML: {res['edg_path']}")
    print(f"  Net XML:   {res['net_path']}")
    print(f"  Edge Map:  {res['map_path']}")
    print(f"  Log file:  {res['log_path']}")
    print("-" * 70)

    # 3. Validate network
    print("Validating SUMO network with sumolib...")
    val = validate_sumo_network(
        G=G,
        net_path=res["net_path"],
        edge_map=res["edge_map"],
        sample_routes_n=200,
        seed=42,
    )

    print("\n--- Network Validation Summary ---")
    print(f"Graph edge count:          {val['graph_edge_count']}")
    print(f"SUMO edge count:           {val['sumo_edge_count']}")
    if val["missing_edges"]:
        print(f"WARNING: Missing edges ({len(val['missing_edges'])}): {val['missing_edges']}")
    else:
        print("Missing edges:             0 (All graph edges present in SUMO)")

    if val["extra_edges"]:
        print(f"Extra edges in SUMO:       {len(val['extra_edges'])}")
    else:
        print("Extra edges:               0")

    print(f"Graph node count:          {val['graph_node_count']}")
    print(f"SUMO node count:           {val['sumo_node_count']}")
    print(f"Graph total length:        {val['graph_total_length_m']:.2f} m")
    print(f"SUMO total length:         {val['sumo_total_length_m']:.2f} m")
    print(f"Length difference:         {val['length_diff_pct']:.4f}% (Threshold: <= 0.5%)")

    print("\n--- Route Connectivity Check ---")
    print(f"Routes sampled (Dijkstra): {val['routes_sampled']}")
    print(f"Consecutive edge pairs:    {val['consecutive_pairs_checked']}")
    print(f"Pairs connected in SUMO:   {val['consecutive_pairs_connected']}")
    if val["disconnected_pairs"]:
        print(f"WARNING: Disconnected pairs ({len(val['disconnected_pairs'])}): {val['disconnected_pairs'][:5]}")
    else:
        print("Disconnected pairs:        0 (100% of consecutive route edges are connected)")

    print("=" * 70)

    # Acceptance assertions: non-loop graph edges must all be present, length within 0.5%, and routes 100% connected
    self_loop_missing = [eid for eid in val["missing_edges"] if eid.startswith("e") and eid[1:].split("_")[0] == eid[1:].split("_")[1]]
    non_loop_missing = [eid for eid in val["missing_edges"] if eid not in self_loop_missing]
    assert len(non_loop_missing) == 0, f"Unexpected non-loop edges missing in SUMO: {non_loop_missing}"
    assert val["length_diff_pct"] <= 0.5, f"Length difference {val['length_diff_pct']:.4f}% exceeds 0.5%"
    assert val["consecutive_pairs_connected"] == val["consecutive_pairs_checked"], "Not all consecutive route edges are connected"
    print("SUCCESS: Network export and validation passed all criteria.\n")


if __name__ == "__main__":
    main()
