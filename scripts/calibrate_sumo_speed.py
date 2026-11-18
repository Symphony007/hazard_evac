#!/usr/bin/env python
"""SUMO speed-matching calibration (Task 8b - Part B).

Goal: adjust a global edge-speed multiplier `m` (sumo.speed_multiplier) so that
SUMO free-flow travel times match point-queue travel times (ratio 1.0 ± 5%).

Algorithm
---------
For each iteration:
  1. Export SUMO network with speeds * vehicle_speed_scale * m.
  2. Re-calibrate tau at the new network median speed (bisection, reuses calibrate.py).
  3. Run point-queue simulator with time_dependent router on demand 100, seed 0.
  4. Replay the same agents in SUMO WITHOUT flood (all flood_times = inf).
  5. Compute ratio = SUMO_mean_evac / PQ_mean_evac on common-safe agents.
  6. Update m <- m * ratio.
  7. Stop if |ratio - 1| <= 0.05 or max 5 iterations.

Writes: sumo/net_unmatched.net.xml (copy of original before speed-matching),
        sumo/net.net.xml (speed-matched network),
        sumo/calibration.json (updated with final m, tau, ratio, speed sweep),
        configs/sumo.yaml (updated tau and speed_multiplier).

OPEN-LOOP NOTE: Routes are planned by TimeDependentRouter in the point-queue
simulator and replayed in SUMO without re-routing (open-loop).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import traci

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    HazardConfig,
    ScenarioConfig,
    StudyAreaConfig,
    SumoConfig,
    TrafficConfig,
    load_config,
)
from src.graph.loader import get_or_build_graph
from src.hazard.flood import FloodHazard
from src.routing.base import make_routing_params
from src.routing.planners import TimeDependentRouter
from src.scenario import build_exits, generate_agents
from src.sim.multi_agent import simulate_population
from src.sumo.calibrate import (
    calibrate_tau_bisection,
    evaluate_speed_sweep,
    write_calibration_artifacts,
)
from src.sumo.network import (
    edge_to_sumo_id,
    export_network_to_sumo,
    find_sumo_binary,
    project_graph_nodes,
)
from src.sumo.replay import replay_routes, write_replay_routes


def _median_speed_ms(G, vehicle_speed_scale: float, speed_multiplier: float) -> float:
    """Return median speed in m/s over all graph edges after applying both multipliers."""
    speeds = []
    for _, _, d in G.edges(data=True):
        speeds.append(float(d.get("speed_kph", 30.0)) * vehicle_speed_scale * speed_multiplier / 3.6)
    return float(np.median(speeds)) if speeds else 4.57


def _update_yaml(yaml_path: Path, tau: float, speed_multiplier: float) -> None:
    """Update tau and speed_multiplier in a YAML file in-place."""
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    tau_written = False
    sm_written = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("tau:"):
            new_lines.append(f"tau: {tau:.4f}")
            tau_written = True
        elif stripped.startswith("speed_multiplier:"):
            new_lines.append(f"speed_multiplier: {speed_multiplier:.6f}")
            sm_written = True
        else:
            new_lines.append(line)
    if not tau_written:
        new_lines.append(f"tau: {tau:.4f}")
    if not sm_written:
        new_lines.append(f"speed_multiplier: {speed_multiplier:.6f}")
    yaml_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def run_speed_matching(
    graph_cfg: StudyAreaConfig,
    hazard_cfg: HazardConfig,
    scenario_cfg: ScenarioConfig,
    traffic_cfg: TrafficConfig,
    sumo_cfg: SumoConfig,
    demand: int = 100,
    seed_index: int = 0,
    max_iter: int = 5,
    target_ratio: float = 1.0,
    tolerance: float = 0.05,
    out_dir: Path = PROJECT_ROOT / "sumo",
    sumo_yaml_path: Path = PROJECT_ROOT / "configs" / "sumo.yaml",
) -> dict:
    """Iterate speed multiplier until SUMO/PQ travel-time ratio is within tolerance.

    Parameters
    ----------
    demand : int
        Number of evacuees for the calibration run.
    seed_index : int
        Seed index (hazard seed = base+seed_index, scenario seed = base+seed_index).
    max_iter : int
        Maximum number of correction iterations (default 5).
    target_ratio : float
        Desired SUMO/PQ ratio (1.0).
    tolerance : float
        Acceptable deviation from target_ratio (0.05 = ±5%).

    Returns
    -------
    dict
        Summary of iterations and final parameters.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    G = get_or_build_graph(graph_cfg)
    params = make_routing_params(scenario_cfg)

    h_cfg = replace(
        hazard_cfg,
        spread_speed_m_per_min=60.0,
        seed=3000 + seed_index,
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

    rng = np.random.default_rng(4000 + seed_index)
    agents = generate_agents(
        G=G,
        hazard=hazard,
        exits=exits,
        scenario_cfg=replace(scenario_cfg, n_evacuees=demand),
        rng=rng,
    )

    # --- Save original (unmatched) net before any speed scaling ---
    unmatched_net = out_dir / "net_unmatched.net.xml"
    matched_net = out_dir / "net.net.xml"
    if matched_net.exists() and not unmatched_net.exists():
        shutil.copy2(str(matched_net), str(unmatched_net))
        print(f"[speed_match] Saved original network as {unmatched_net.name}")

    # Read edge_map (needed for replay)
    map_file = out_dir / "edge_map.json"
    with open(map_file, "r", encoding="utf-8") as f:
        edge_map_data = json.load(f)
    edge_map_g2s: dict[str, str] = edge_map_data["graph_to_sumo"]
    edge_map_s2g: dict[str, list] = edge_map_data["sumo_to_graph"]

    vtype_file = out_dir / "vtype.add.xml"

    # --- Run point-queue baseline (independent of SUMO, run once) ---
    router = TimeDependentRouter()
    pq_res = simulate_population(
        G=G,
        hazard=hazard,
        router=router,
        agents=agents,
        exits=exits,
        params=params,
        traffic_cfg=traffic_cfg,
    )
    pq_safe = {r.agent_id: r for r in pq_res.agent_results if r.safe}
    pq_mean_evac = float(np.mean([r.evac_time_s for r in pq_safe.values()])) if pq_safe else 0.0
    print(f"[speed_match] PQ baseline: {len(pq_safe)}/{len(agents)} safe, mean evac = {pq_mean_evac:.1f}s")

    if not pq_safe:
        print("[speed_match] WARNING: no safe agents in point-queue baseline; skipping speed matching.")
        return {"converged": False, "reason": "no_pq_safe_agents"}

    m = float(getattr(sumo_cfg, "speed_multiplier", 1.0))
    tau = float(sumo_cfg.tau)
    history: list[dict] = []

    print(f"\n{'='*70}")
    print("TASK 8b: SUMO SPEED MATCHING")
    print(f"{'='*70}")
    print(f"Demand: {demand} | Seed: {seed_index} | Target ratio: {target_ratio:.2f} ± {tolerance:.2f}")
    print(f"Initial speed_multiplier: {m:.4f} | Initial tau: {tau:.4f}s")
    print("-" * 70)

    final_ratio = None
    for iteration in range(1, max_iter + 1):
        print(f"\n[Iter {iteration}] speed_multiplier = {m:.6f}")

        # 1. Export network with new speed multiplier
        print(f"  Exporting network with vehicle_speed_scale={scenario_cfg.vehicle_speed_scale:.2f} * m={m:.4f} ...")
        export_result = export_network_to_sumo(
            G=G,
            output_dir=out_dir,
            vehicle_speed_scale=scenario_cfg.vehicle_speed_scale * m,
            run_netconvert=True,
        )
        # Update edge_map (might have changed if m changes edge ordering, but ids are stable)
        # edge_map is deterministic from graph structure, not speed.

        # 2. Re-calibrate tau at new median speed
        med_speed = _median_speed_ms(G, scenario_cfg.vehicle_speed_scale, m)
        print(f"  Network median speed: {med_speed:.3f} m/s ({med_speed*3.6:.1f} kph)")
        print(f"  Calibrating tau at median speed ...")
        tau, achieved_q, cal_history = calibrate_tau_bisection(
            speed_ms=med_speed,
            target_discharge=float(graph_cfg.flow_rate_veh_per_min_per_lane),
            tau_min=1.0,
            tau_max=6.0,
            tol=0.5,
            max_iter=8,
        )
        print(f"  Calibrated tau = {tau:.4f}s, discharge = {achieved_q:.2f} veh/min/lane")

        # Update vtype.add.xml with new tau
        _write_vtype_xml(vtype_file, tau, sumo_cfg)

        # 3. Build agents_with_routes for SUMO replay (no flood: all inf)
        flood_times_inf: dict[str, float] = {sid: float("inf") for sid in edge_map_s2g.keys()}

        # Gather routes from pq_res (safe agents)
        agents_with_routes = []
        for a in agents:
            if a.id in pq_safe:
                r = pq_safe[a.id]
                if r.route_edges:
                    agents_with_routes.append((a, r.route_edges))

        if not agents_with_routes:
            print("  WARNING: No agents have recorded route_edges; skipping SUMO replay.")
            break

        # 4. Replay in SUMO (no flood)
        print(f"  Replaying {len(agents_with_routes)} agent routes in SUMO (no flood)...")
        label = f"speedmatch_iter{iteration}"
        outcomes = replay_routes(
            net_path=matched_net,
            vtype_path=vtype_file,
            agents_with_routes=agents_with_routes,
            edge_map_g2s=edge_map_g2s,
            flood_times_by_sumo_edge=flood_times_inf,
            step_length_s=sumo_cfg.step_length_s,
            time_to_teleport=sumo_cfg.time_to_teleport,
            horizon_s=3600,
            sumo_label=label,
        )

        # 5. Compute SUMO/PQ ratio on common safe agents
        sumo_arrived = {aid: oc for aid, oc in outcomes.items() if oc.status == "ARRIVED"}
        common_ids = sorted(set(sumo_arrived.keys()) & set(pq_safe.keys()))

        if not common_ids:
            print(f"  WARNING: No common safe agents between SUMO ({len(sumo_arrived)}) and PQ ({len(pq_safe)})")
            break

        sumo_evac = []
        pq_evac = []
        for aid in common_ids:
            oc = sumo_arrived[aid]
            if oc.actual_arrival_s is not None and oc.planned_depart_s is not None:
                sumo_evac.append(oc.actual_arrival_s - oc.planned_depart_s)
            pq_evac.append(pq_safe[aid].evac_time_s)

        if not sumo_evac:
            print("  WARNING: Could not compute SUMO evac times; check tripinfo output.")
            break

        sumo_mean = float(np.mean(sumo_evac))
        pq_mean = float(np.mean(pq_evac))
        ratio = sumo_mean / pq_mean if pq_mean > 0 else 1.0
        final_ratio = ratio

        print(f"  Common safe agents: {len(common_ids)}")
        print(f"  PQ mean evac:   {pq_mean:.1f}s | SUMO mean evac: {sumo_mean:.1f}s | Ratio: {ratio:.4f}")

        history.append({
            "iteration": iteration,
            "speed_multiplier": m,
            "tau": tau,
            "median_speed_ms": med_speed,
            "sumo_mean_evac_s": sumo_mean,
            "pq_mean_evac_s": pq_mean,
            "ratio": ratio,
            "n_common_agents": len(common_ids),
        })

        if abs(ratio - target_ratio) <= tolerance:
            print(f"\n  CONVERGED: ratio {ratio:.4f} within ±{tolerance:.2f} of target {target_ratio:.2f}")
            break

        # 6. Update multiplier
        m_new = m * ratio
        print(f"  Updating speed_multiplier: {m:.6f} -> {m_new:.6f}")
        m = m_new

    # 7. Evaluate final speed sweep and write artifacts
    print(f"\n[speed_match] Running final speed sweep (tau={tau:.4f}s)...")
    sweep = evaluate_speed_sweep(tau=tau, warmup_s=30.0, window_s=120.0)

    write_calibration_artifacts(
        output_dir=out_dir,
        calibrated_tau=tau,
        target_discharge=float(graph_cfg.flow_rate_veh_per_min_per_lane),
        median_speed=_median_speed_ms(G, scenario_cfg.vehicle_speed_scale, m),
        median_discharge=achieved_q,  # type: ignore[possibly-undefined]
        sweep_results=sweep,
        history=cal_history,  # type: ignore[possibly-undefined]
        vtype_params={"speed_multiplier": m},
    )

    # Also add speed-matching history and final m to calibration.json
    cal_json_path = out_dir / "calibration.json"
    with open(cal_json_path, "r", encoding="utf-8") as f:
        cal_data = json.load(f)
    cal_data["speed_matching"] = {
        "final_speed_multiplier": m,
        "final_tau": tau,
        "final_ratio": final_ratio,
        "iterations": history,
    }
    with open(cal_json_path, "w", encoding="utf-8") as f:
        json.dump(cal_data, f, indent=2)

    # Update configs/sumo.yaml
    _update_yaml(sumo_yaml_path, tau, m)
    print(f"[speed_match] Updated {sumo_yaml_path.name}: tau={tau:.4f}s, speed_multiplier={m:.6f}")

    converged = final_ratio is not None and abs(final_ratio - target_ratio) <= tolerance
    print(f"\n{'='*70}")
    print(f"RESULT: speed_multiplier = {m:.6f} | tau = {tau:.4f}s | ratio = {final_ratio:.4f}")
    print(f"STATUS: {'CONVERGED' if converged else 'NOT CONVERGED (ran max iterations)'}")
    print(f"{'='*70}")

    return {
        "converged": converged,
        "speed_multiplier": m,
        "tau": tau,
        "final_ratio": final_ratio,
        "iterations": history,
        "speed_sweep": sweep,
    }


def _write_vtype_xml(vtype_path: Path, tau: float, cfg: SumoConfig) -> None:
    """Write updated vtype.add.xml with new tau value."""
    with open(vtype_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("<additional>\n")
        f.write(
            f'    <vType id="calib_vtype" '
            f'length="{cfg.vehicle_length:.2f}" '
            f'minGap="{cfg.min_gap:.2f}" '
            f'accel="{cfg.accel:.2f}" '
            f'decel="{cfg.decel:.2f}" '
            f'sigma="{cfg.sigma:.2f}" '
            f'tau="{tau:.4f}"/>\n'
        )
        f.write("</additional>\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="SUMO speed-matching calibration (Task 8b).")
    parser.add_argument("--graph-config", default="configs/study_area.yaml")
    parser.add_argument("--hazard-config", default="configs/hazard.yaml")
    parser.add_argument("--scenario-config", default="configs/scenario.yaml")
    parser.add_argument("--traffic-config", default="configs/traffic.yaml")
    parser.add_argument("--sumo-config", default="configs/sumo.yaml")
    parser.add_argument("--demand", type=int, default=100)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=5)
    args = parser.parse_args()

    graph_cfg = load_config(PROJECT_ROOT / args.graph_config, StudyAreaConfig)
    hazard_cfg = load_config(PROJECT_ROOT / args.hazard_config, HazardConfig)
    scenario_cfg = load_config(PROJECT_ROOT / args.scenario_config, ScenarioConfig)
    traffic_cfg = load_config(PROJECT_ROOT / args.traffic_config, TrafficConfig)
    sumo_cfg = load_config(PROJECT_ROOT / args.sumo_config, SumoConfig)
    out_dir = PROJECT_ROOT / sumo_cfg.sumo_dir
    sumo_yaml = PROJECT_ROOT / args.sumo_config

    run_speed_matching(
        graph_cfg=graph_cfg,
        hazard_cfg=hazard_cfg,
        scenario_cfg=scenario_cfg,
        traffic_cfg=traffic_cfg,
        sumo_cfg=sumo_cfg,
        demand=args.demand,
        seed_index=args.seed_index,
        max_iter=args.max_iter,
        out_dir=out_dir,
        sumo_yaml_path=sumo_yaml,
    )


if __name__ == "__main__":
    main()

