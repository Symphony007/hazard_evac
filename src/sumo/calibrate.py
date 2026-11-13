"""SUMO car-following model calibration using TraCI queue discharge.

Calibrates vehicle type headway (tau) such that saturation queue discharge
matches the point-queue model capacity:
- Discharge target: flow_rate_veh_per_min_per_lane (12 veh/min/lane)
- Jam density: ~120 veh/km/lane (~8.3m spacing = length 5.0m + minGap 3.3m)
- Car-following parameters: accel 2.6 m/s^2, decel 4.5 m/s^2, sigma 0.0 (deterministic)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import traci

from src.sumo.network import find_sumo_binary


def build_calibration_network(
    work_dir: str | Path,
    speed_ms: float,
    n_lanes: int = 1,
    length_e0: float = 1500.0,
    length_e1: float = 2000.0,
) -> Path:
    """Generate a straight 2-edge calibration network via netconvert."""
    w_dir = Path(work_dir)
    nod_file = w_dir / "nodes.nod.xml"
    edg_file = w_dir / "edges.edg.xml"
    net_file = w_dir / "net.net.xml"

    # Node 0 at 0, Node 1 at length_e0, Node 2 at length_e0 + length_e1
    x1 = float(length_e0)
    x2 = float(length_e0 + length_e1)

    with open(nod_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">\n')
        f.write('    <node id="N0" x="0.0" y="0.0"/>\n')
        f.write(f'    <node id="N1" x="{x1:.1f}" y="0.0"/>\n')
        f.write(f'    <node id="N2" x="{x2:.1f}" y="0.0"/>\n')
        f.write("</nodes>\n")

    with open(edg_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">\n')
        f.write(
            f'    <edge id="E0" from="N0" to="N1" numLanes="{n_lanes}" '
            f'speed="{speed_ms:.3f}" length="{length_e0:.1f}" priority="3"/>\n'
        )
        f.write(
            f'    <edge id="E1" from="N1" to="N2" numLanes="{n_lanes}" '
            f'speed="{speed_ms:.3f}" length="{length_e1:.1f}" priority="3"/>\n'
        )
        f.write("</edges>\n")

    netconvert_bin = find_sumo_binary("netconvert")
    cmd = [
        netconvert_bin,
        "--node-files", str(nod_file.resolve()),
        "--edge-files", str(edg_file.resolve()),
        "-o", str(net_file.resolve()),
        "--no-turnarounds", "false",
        "--tls.discard-loaded", "true",
        "--tls.guess", "false",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"netconvert failed: {proc.stderr}")

    return net_file


def write_calibration_routes(
    rou_file: str | Path,
    tau: float,
    n_vehicles: int = 100,
    n_lanes: int = 1,
    length: float = 5.0,
    min_gap: float = 3.3,
    accel: float = 2.6,
    decel: float = 4.5,
    sigma: float = 0.0,
    e0_length: float = 1500.0,
) -> None:
    """Write route file with a standing queue of vehicles placed at bumper-to-bumper spacing."""
    spacing = length + min_gap + 0.05  # ~8.35 m
    with open(rou_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("<routes>\n")
        f.write(
            f'    <vType id="calib_vtype" length="{length:.2f}" minGap="{min_gap:.2f}" '
            f'accel="{accel:.2f}" decel="{decel:.2f}" sigma="{sigma:.2f}" tau="{tau:.4f}"/>\n'
        )
        f.write('    <route id="r0" edges="E0 E1"/>\n')

        head_pos = e0_length - 50.0  # Head of queue 50m before junction N1
        for lane_idx in range(n_lanes):
            for i in range(n_vehicles):
                pos = head_pos - i * spacing
                if pos < 10.0:
                    break
                vid = f"veh_{lane_idx}_{i}"
                f.write(
                    f'    <vehicle id="{vid}" type="calib_vtype" route="r0" '
                    f'depart="0.0" departLane="{lane_idx}" departPos="{pos:.2f}" departSpeed="0.0"/>\n'
                )
        f.write("</routes>\n")


def measure_discharge(
    speed_ms: float,
    tau: float,
    n_lanes: int = 1,
    n_vehicles: int = 100,
    warmup_s: float = 60.0,
    window_s: float = 300.0,
    length: float = 5.0,
    min_gap: float = 3.3,
    accel: float = 2.6,
    decel: float = 4.5,
    sigma: float = 0.0,
) -> float:
    """Measure single-lane saturation discharge rate (veh/min/lane) over a 5-minute window."""
    temp_dir = tempfile.mkdtemp(prefix="sumo_calib_")
    try:
        net_file = build_calibration_network(
            work_dir=temp_dir,
            speed_ms=speed_ms,
            n_lanes=n_lanes,
            length_e0=1500.0,
            length_e1=2000.0,
        )
        rou_file = Path(temp_dir) / "routes.rou.xml"
        write_calibration_routes(
            rou_file=rou_file,
            tau=tau,
            n_vehicles=n_vehicles,
            n_lanes=n_lanes,
            length=length,
            min_gap=min_gap,
            accel=accel,
            decel=decel,
            sigma=sigma,
            e0_length=1500.0,
        )

        sumo_bin = find_sumo_binary("sumo")
        cmd = [
            sumo_bin,
            "-n", str(net_file.resolve()),
            "-r", str(rou_file.resolve()),
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
            "--step-length", "1.0",
        ]

        traci.start(cmd)
        entered_e1: dict[str, float] = {}

        total_steps = int(warmup_s + window_s)
        for _ in range(total_steps):
            traci.simulationStep()
            t_now = traci.simulation.getTime()
            for vid in traci.edge.getLastStepVehicleIDs("E1"):
                if vid not in entered_e1:
                    entered_e1[vid] = t_now

        traci.close()

        # Count vehicles entering E1 inside the measurement window [warmup_s, warmup_s + window_s)
        window_start = warmup_s
        window_end = warmup_s + window_s
        count_in_window = sum(
            1 for vid, t in entered_e1.items()
            if window_start <= t < window_end
        )

        discharge_veh_per_min = count_in_window / (window_s / 60.0) / float(n_lanes)
        return float(discharge_veh_per_min)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def calibrate_tau_bisection(
    speed_ms: float,
    target_discharge: float = 12.0,
    tau_min: float = 1.0,
    tau_max: float = 6.0,
    tol: float = 0.15,
    max_iter: int = 12,
    warmup_s: float = 60.0,
    window_s: float = 300.0,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Find tau on [tau_min, tau_max] such that saturation discharge equals target_discharge.

    Discharge rate q is monotonically decreasing with tau (larger headway -> lower flow).
    Bisection rule:
    - If q > target -> tau is too small -> search higher [tau_mid, tau_high]
    - If q < target -> tau is too large -> search lower [tau_low, tau_mid]
    """
    history: list[dict[str, Any]] = []
    low = tau_min
    high = tau_max

    best_tau = (low + high) / 2.0
    best_discharge = 0.0

    for iteration in range(1, max_iter + 1):
        mid = (low + high) / 2.0
        q = measure_discharge(
            speed_ms=speed_ms,
            tau=mid,
            n_lanes=1,
            warmup_s=warmup_s,
            window_s=window_s,
        )
        diff = q - target_discharge
        history.append({
            "iteration": iteration,
            "tau_low": low,
            "tau_high": high,
            "tau_mid": mid,
            "discharge_veh_per_min": q,
            "diff": diff,
        })

        best_tau = mid
        best_discharge = q

        if abs(diff) <= tol:
            break

        if q > target_discharge:
            # Discharge too high -> need larger headway -> increase low bound
            low = mid
        else:
            # Discharge too low -> need smaller headway -> decrease high bound
            high = mid

    return best_tau, best_discharge, history


def evaluate_speed_sweep(
    tau: float,
    speeds: list[float] | None = None,
    warmup_s: float = 60.0,
    window_s: float = 300.0,
) -> dict[str, Any]:
    """Evaluate saturation discharge rate across speeds and 2-lane edge."""
    if speeds is None:
        speeds = [2.0, 3.5, 5.0, 7.0]

    sweep_results: dict[str, float] = {}
    for s in speeds:
        q = measure_discharge(
            speed_ms=s,
            tau=tau,
            n_lanes=1,
            warmup_s=warmup_s,
            window_s=window_s,
        )
        sweep_results[f"{s:.1f}_ms"] = q

    # Test 2-lane edge at median speed (e.g. 5.0 or 4.57 m/s)
    q_2lane = measure_discharge(
        speed_ms=speeds[2] if len(speeds) > 2 else 5.0,
        tau=tau,
        n_lanes=2,
        warmup_s=warmup_s,
        window_s=window_s,
    )
    # q_2lane is returned per lane by measure_discharge (total / 2)
    sweep_results["2_lane_per_lane"] = q_2lane
    sweep_results["2_lane_total"] = q_2lane * 2.0

    return sweep_results


def write_calibration_artifacts(
    output_dir: str | Path,
    calibrated_tau: float,
    target_discharge: float,
    median_speed: float,
    median_discharge: float,
    sweep_results: dict[str, Any],
    history: list[dict[str, Any]],
    vtype_params: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write calibrated vehicle type to vtype.add.xml and numbers to calibration.json."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vtype_xml_path = out_dir / "vtype.add.xml"
    calib_json_path = out_dir / "calibration.json"

    params = {
        "length": 5.0,
        "minGap": 3.3,
        "accel": 2.6,
        "decel": 4.5,
        "sigma": 0.0,
        "tau": round(calibrated_tau, 4),
    }
    if vtype_params:
        params.update(vtype_params)

    # 1. Write vtype.add.xml
    with open(vtype_xml_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("<additional>\n")
        f.write(
            f'    <vType id="calib_vtype" '
            f'length="{params["length"]:.2f}" '
            f'minGap="{params["minGap"]:.2f}" '
            f'accel="{params["accel"]:.2f}" '
            f'decel="{params["decel"]:.2f}" '
            f'sigma="{params["sigma"]:.2f}" '
            f'tau="{params["tau"]:.4f}"/>\n'
        )
        f.write("</additional>\n")

    # 2. Write calibration.json
    payload = {
        "target_discharge_veh_per_min": target_discharge,
        "median_speed_ms": median_speed,
        "calibrated_tau_s": calibrated_tau,
        "calibrated_discharge_veh_per_min": median_discharge,
        "acceptance_ratio": median_discharge / target_discharge,
        "within_15_pct": abs(median_discharge - target_discharge) / target_discharge <= 0.15,
        "vehicle_type": params,
        "speed_sweep": sweep_results,
        "bisection_history": history,
    }
    with open(calib_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return vtype_xml_path, calib_json_path

