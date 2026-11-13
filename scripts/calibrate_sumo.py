#!/usr/bin/env python
"""Calibrate SUMO car-following headway (tau) to match point-queue capacity.

Usage:
    python scripts/calibrate_sumo.py [--net-file sumo/net.net.xml]
                                     [--sumo-config configs/sumo.yaml]
                                     [--study-config configs/study_area.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sumolib
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import StudyAreaConfig, SumoConfig, load_config
from src.sumo.calibrate import (
    calibrate_tau_bisection,
    evaluate_speed_sweep,
    write_calibration_artifacts,
)


def compute_median_network_speed(net_file: Path) -> float:
    """Compute median speed of exported SUMO network edges."""
    if not net_file.exists():
        raise FileNotFoundError(f"SUMO network file {net_file} not found. Run export first.")
    net = sumolib.net.readNet(str(net_file))
    speeds = [e.getSpeed() for e in net.getEdges(withInternal=False)]
    return float(np.median(speeds))


def update_sumo_yaml_tau(yaml_path: Path, tau_val: float) -> None:
    """Update calibrated tau field in configs/sumo.yaml."""
    if not yaml_path.exists():
        return
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["tau"] = round(float(tau_val), 4)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate SUMO car-following model.")
    parser.add_argument("--sumo-config", type=str, default="configs/sumo.yaml")
    parser.add_argument("--study-config", type=str, default="configs/study_area.yaml")
    args = parser.parse_args()

    sumo_cfg = load_config(PROJECT_ROOT / args.sumo_config, SumoConfig)
    study_cfg = load_config(PROJECT_ROOT / args.study_config, StudyAreaConfig)

    out_dir = PROJECT_ROOT / sumo_cfg.sumo_dir
    net_file = out_dir / "net.net.xml"

    print("=" * 70)
    print("TASK 8a: CAR-FOLLOWING MODEL CALIBRATION (SUMO TraCI)")
    print("=" * 70)

    # 1. Target capacity & density
    target_discharge = float(study_cfg.flow_rate_veh_per_min_per_lane)  # 12 veh/min/lane
    jam_density = float(study_cfg.capacity_per_lane_veh_per_km)         # 120 veh/km/lane
    headway_target_s = 60.0 / target_discharge                         # 5.0 s
    spacing_target_m = 1000.0 / jam_density                            # 8.33 m

    print(f"Target capacity:        {target_discharge:.1f} veh/min/lane (headway {headway_target_s:.1f}s)")
    print(f"Target jam density:     {jam_density:.1f} veh/km/lane (spacing {spacing_target_m:.2f}m)")
    print(f"Vehicle length:         {sumo_cfg.vehicle_length:.1f} m")
    print(f"Minimum gap (minGap):   {sumo_cfg.min_gap:.1f} m  (sum = {sumo_cfg.vehicle_length + sumo_cfg.min_gap:.1f}m)")
    print(f"Acceleration:           {sumo_cfg.accel:.1f} m/s^2")
    print(f"Deceleration:           {sumo_cfg.decel:.1f} m/s^2")
    print(f"Sigma (imperfection):   {sumo_cfg.sigma:.1f}")

    # 2. Compute median edge speed from exported network
    median_speed = compute_median_network_speed(net_file)
    print(f"Exported median speed:  {median_speed:.2f} m/s ({median_speed * 3.6:.1f} kph)")
    print("-" * 70)

    # 3. Bisection search for tau on [1.0, 6.0] s
    print(f"Bisection search for tau in [1.0, 6.0]s at speed {median_speed:.2f} m/s...")
    calibrated_tau, median_discharge, history = calibrate_tau_bisection(
        speed_ms=median_speed,
        target_discharge=target_discharge,
        tau_min=1.0,
        tau_max=6.0,
        tol=0.15,
        max_iter=10,
        warmup_s=60.0,
        window_s=300.0,
    )

    print("\n--- Bisection Iteration History ---")
    for row in history:
        print(
            f"  Iter {row['iteration']:2d}: tau = {row['tau_mid']:.4f} s | "
            f"Discharge = {row['discharge_veh_per_min']:5.2f} veh/min/lane | "
            f"Error = {row['diff']:+5.2f}"
        )

    print("-" * 70)
    print(f"Calibrated tau:          {calibrated_tau:.4f} s")
    print(f"Achieved discharge:      {median_discharge:.2f} veh/min/lane (Target: {target_discharge:.1f})")
    err_pct = abs(median_discharge - target_discharge) / target_discharge * 100.0
    print(f"Discharge error:         {err_pct:.2f}% (Acceptance: <= 15%)")

    # 4. Evaluate speed sweep at 2.0, 3.5, 5.0, 7.0 m/s and 2-lane edge
    print("\nEvaluating speed sweep and multi-lane scaling...")
    speeds = [2.0, 3.5, 5.0, 7.0]
    sweep_results = evaluate_speed_sweep(
        tau=calibrated_tau,
        speeds=speeds,
        warmup_s=60.0,
        window_s=300.0,
    )

    print("\n--- Speed Sweep & Multi-Lane Results ---")
    deviating_speeds: list[str] = []
    for s in speeds:
        key = f"{s:.1f}_ms"
        val = sweep_results[key]
        diff_pct = (val - target_discharge) / target_discharge * 100.0
        status = "OK" if abs(diff_pct) <= 15.0 else "DEVIATES (>15%)"
        if status != "OK":
            deviating_speeds.append(f"{s:.1f} m/s ({diff_pct:+.1f}%)")
        print(f"  Speed {s:4.1f} m/s ({s*3.6:4.1f} kph): {val:5.2f} veh/min/lane ({diff_pct:+5.1f}%) [{status}]")

    q_2lane_total = sweep_results["2_lane_total"]
    q_2lane_per_lane = sweep_results["2_lane_per_lane"]
    print(f"  2-lane edge total:        {q_2lane_total:5.2f} veh/min ({q_2lane_per_lane:5.2f} veh/min/lane, ratio vs 1-lane: {q_2lane_total/median_discharge:.2f}x)")

    if deviating_speeds:
        print(f"\nNotice on speed sensitivity: At lower speeds, kinematic acceleration / spacing causes minor deviations: {', '.join(deviating_speeds)}")

    # 5. Write artifacts
    vtype_xml, calib_json = write_calibration_artifacts(
        output_dir=out_dir,
        calibrated_tau=calibrated_tau,
        target_discharge=target_discharge,
        median_speed=median_speed,
        median_discharge=median_discharge,
        sweep_results=sweep_results,
        history=history,
        vtype_params={
            "length": sumo_cfg.vehicle_length,
            "minGap": sumo_cfg.min_gap,
            "accel": sumo_cfg.accel,
            "decel": sumo_cfg.decel,
            "sigma": sumo_cfg.sigma,
        },
    )
    print(f"\nWrote calibrated vehicle type XML: {vtype_xml}")
    print(f"Wrote calibration data JSON:      {calib_json}")

    # Update configs/sumo.yaml
    update_sumo_yaml_tau(PROJECT_ROOT / args.sumo_config, calibrated_tau)
    print(f"Updated tau in {args.sumo_config} -> {calibrated_tau:.4f} s")

    print("=" * 70)
    assert err_pct <= 15.0, f"Discharge error {err_pct:.2f}% exceeds 15% tolerance"
    print("SUCCESS: Calibration passed acceptance threshold (within +-15% of 12 veh/min/lane).\n")


if __name__ == "__main__":
    main()

