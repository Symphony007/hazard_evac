"""SUMO open-loop replay engine for evacuation routes (Task 8b).

Open-Loop Limitation
--------------------
Routes are fixed (computed by the point-queue router) and replayed in SUMO
without any re-routing.  SUMO provides microscopic car-following physics,
junction conflicts, and vehicle-level flood closure, but routers do NOT react
to SUMO's real-time traffic state.  This is intentional: the goal is to
validate whether the router's chosen paths survive microscopic traffic, not to
re-optimise them.

Flood Closure Rule
------------------
At each simulation step t, every vehicle whose current road edge e has
flood_time[e] <= t is immediately removed (trapped).  This matches the
point-queue rule: being on an edge when it closes => trapped.
We do NOT use setDisallowed because that only blocks new entries; existing
vehicles need to be detected and removed.  Subscriptions (DOMAIN_VEHICLE,
VAR_ROAD_ID) give us per-vehicle road id without per-vehicle API calls inside
the step loop.

Efficiency
----------
- One subscription per vehicle (set at insertion), read in bulk via
  getAllSubscriptionResults() each step.
- Tripinfo XML output captures duration/waitingTime/timeLoss/departDelay for
  all arrivals without per-step queries.
- No other per-vehicle TraCI calls are made inside the step loop.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import traci
import traci.constants as tc

from src.sumo.network import edge_to_sumo_id, find_sumo_binary


# ---------------------------------------------------------------------------
# Data classes for per-vehicle outcomes
# ---------------------------------------------------------------------------

@dataclass
class VehicleOutcome:
    """Outcome for one vehicle in a SUMO replay run.

    Status values
    -------------
    ARRIVED       - Vehicle reached an exit (tripinfo recorded).
    TRAPPED       - Removed because its current edge flooded at time `trap_time`.
    UNFINISHED    - Still in network at horizon (gridlock or too slow).
    NEVER_INSERTED - SUMO never inserted the vehicle (departure capacity issue).
    """

    agent_id: int
    status: str            # "ARRIVED" | "TRAPPED" | "UNFINISHED" | "NEVER_INSERTED"

    # --- ARRIVED fields (from tripinfo) ---
    duration_s: float | None = None        # total trip duration from actual departure
    waiting_time_s: float | None = None    # cumulative time stopped (speed < threshold)
    time_loss_s: float | None = None       # time lost vs free-flow travel
    depart_delay_s: float | None = None    # delay between intended and actual departure

    # --- TRAPPED fields ---
    trap_time_s: float | None = None
    trap_edge_sumo_id: str | None = None   # SUMO edge id where vehicle was when trapped

    # --- Shared timing (for ARRIVED and common-safe metric) ---
    planned_depart_s: float | None = None  # agent.depart_time_s
    actual_depart_s: float | None = None   # actual insertion time in SUMO
    actual_arrival_s: float | None = None  # time vehicle reached exit (ARRIVED only)


# ---------------------------------------------------------------------------
# Route file writer (open-loop, one vehicle per agent with route)
# ---------------------------------------------------------------------------

def write_replay_routes(
    rou_path: str | Path,
    agents_with_routes: list[tuple[Any, list[tuple]]],  # (Evacuee, route_edges)
    edge_map_g2s: dict[str, str],
    vtype_id: str = "calib_vtype",
) -> list[int]:
    """Write a SUMO routes XML file for open-loop replay.

    Parameters
    ----------
    rou_path : str or Path
        Output route file path.
    agents_with_routes : list of (Evacuee, route_edges)
        Only agents with non-empty route_edges should be passed here.
        Agents already at an exit (no route) or with no route are excluded.
    edge_map_g2s : dict
        Mapping from "u,v,k" graph edge key to SUMO edge id.
    vtype_id : str
        Vehicle type id from vtype.add.xml.

    Returns
    -------
    list[int]
        Agent IDs successfully written to the route file, sorted by departure time.
    """
    # Sort by departure time, break ties by agent id (SUMO requires chronological order)
    sorted_pairs = sorted(agents_with_routes, key=lambda x: (float(x[0].depart_time_s), int(x[0].id)))

    written_ids: list[int] = []

    with open(rou_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("<routes>\n")

        for agent, route_edges in sorted_pairs:
            if not route_edges:
                continue

            sumo_edge_ids: list[str] = []
            for u, v, k in route_edges:
                seid = edge_map_g2s.get(f"{u},{v},{k}")
                if seid is not None:
                    sumo_edge_ids.append(seid)

            # Deduplicate consecutive repeated edges (can happen at replan boundaries)
            deduped: list[str] = []
            for eid in sumo_edge_ids:
                if not deduped or deduped[-1] != eid:
                    deduped.append(eid)

            if not deduped:
                continue

            edges_str = " ".join(deduped)
            aid = int(agent.id)
            # departLane="best": SUMO picks the best lane for the route
            # departPos="free": place at the first free position
            # departSpeed="max": enter at free-flow speed
            # arrivalPos="max": exit at the far end of the last edge
            f.write(f'    <route id="r_{aid}" edges="{edges_str}"/>\n')
            f.write(
                f'    <vehicle id="veh_{aid}" type="{vtype_id}" route="r_{aid}" '
                f'depart="{agent.depart_time_s:.2f}" '
                f'departLane="best" departPos="free" departSpeed="max" arrivalPos="max"/>\n'
            )
            written_ids.append(aid)

        f.write("</routes>\n")

    return written_ids


# ---------------------------------------------------------------------------
# Tripinfo XML parser
# ---------------------------------------------------------------------------

def _parse_tripinfo(tripinfo_path: str | Path) -> dict[int, dict[str, float]]:
    """Parse SUMO tripinfo XML and return per-agent metrics.

    Returns
    -------
    dict[int, dict]
        agent_id -> {duration, waitingTime, timeLoss, departDelay, depart, arrival}
    """
    result: dict[int, dict[str, float]] = {}
    if not Path(tripinfo_path).exists():
        return result
    try:
        tree = ET.parse(str(tripinfo_path))
    except ET.ParseError:
        return result

    for elem in tree.getroot().findall("tripinfo"):
        vid = elem.get("id", "")
        if not vid.startswith("veh_"):
            continue
        try:
            aid = int(vid.replace("veh_", ""))
        except ValueError:
            continue
        result[aid] = {
            "duration": float(elem.get("duration", "nan")),
            "waitingTime": float(elem.get("waitingTime", "nan")),
            "timeLoss": float(elem.get("timeLoss", "nan")),
            "departDelay": float(elem.get("departDelay", "nan")),
            "depart": float(elem.get("depart", "nan")),
            "arrival": float(elem.get("arrival", "nan")),
        }
    return result


# ---------------------------------------------------------------------------
# Main replay function
# ---------------------------------------------------------------------------

def replay_routes(
    net_path: str | Path,
    vtype_path: str | Path,
    agents_with_routes: list[tuple[Any, list[tuple]]],  # (Evacuee, route_edges)
    edge_map_g2s: dict[str, str],
    flood_times_by_sumo_edge: dict[str, float],  # sumo_edge_id -> flood arrival time (s); inf if never
    step_length_s: float = 1.0,
    time_to_teleport: int = -1,
    horizon_s: int = 3600,
    work_dir: str | Path | None = None,
    sumo_label: str = "replay",
    vtype_id: str = "calib_vtype",
) -> dict[int, VehicleOutcome]:
    """Replay fixed router routes in SUMO with flood road closures.

    Open-Loop: routes are fixed at input; SUMO provides microscopic physics.
    Flood closure: vehicles on an edge whose flood_time <= t are removed at step t.
    Uses vehicle subscriptions + bulk getAllSubscriptionResults for efficiency.

    Parameters
    ----------
    net_path : str or Path
        Path to compiled SUMO net.net.xml.
    vtype_path : str or Path
        Path to vehicle type additional file (vtype.add.xml).
    agents_with_routes : list of (Evacuee, route_edges)
        Each entry is (agent, list_of_(u,v,k)_edges). Agents with empty
        route_edges are skipped (they are pre-counted as unsafe / already-at-exit).
    edge_map_g2s : dict
        Graph edge "u,v,k" -> SUMO edge id mapping.
    flood_times_by_sumo_edge : dict
        SUMO edge id -> flood arrival time in seconds (math.inf if the edge never floods).
    step_length_s : float
        SUMO simulation step length in seconds.
    time_to_teleport : int
        SUMO --time-to-teleport value (-1 = no teleport).
    horizon_s : int
        Maximum simulation time in seconds.
    work_dir : str or Path or None
        Directory to write temp files. Uses a system temp dir if None.
    sumo_label : str
        TraCI connection label (must be unique per parallel run).
    vtype_id : str
        Vehicle type id declared in vtype_path.

    Returns
    -------
    dict[int, VehicleOutcome]
        Outcome per agent_id for every agent in agents_with_routes.
    """
    sumo_bin = find_sumo_binary("sumo")

    # Choose working directory
    _own_tmpdir = None
    if work_dir is None:
        _own_tmpdir = tempfile.mkdtemp(prefix="sumo_replay_")
        work_dir_path = Path(_own_tmpdir)
    else:
        work_dir_path = Path(work_dir)
        work_dir_path.mkdir(parents=True, exist_ok=True)

    try:
        rou_path = work_dir_path / f"{sumo_label}_routes.rou.xml"
        tripinfo_path = work_dir_path / f"{sumo_label}_tripinfo.xml"

        # Write route file
        written_ids = write_replay_routes(
            rou_path=rou_path,
            agents_with_routes=agents_with_routes,
            edge_map_g2s=edge_map_g2s,
            vtype_id=vtype_id,
        )
        written_set: set[int] = set(written_ids)

        # Build planned_depart lookup
        planned_depart: dict[int, float] = {
            int(a.id): float(a.depart_time_s)
            for a, _ in agents_with_routes
        }

        # Mark agents with no SUMO route as NEVER_INSERTED upfront
        outcomes: dict[int, VehicleOutcome] = {}
        for agent, _ in agents_with_routes:
            aid = int(agent.id)
            if aid not in written_set:
                outcomes[aid] = VehicleOutcome(
                    agent_id=aid,
                    status="NEVER_INSERTED",
                    planned_depart_s=float(agent.depart_time_s),
                )

        # Launch SUMO via TraCI
        sumo_cmd = [
            sumo_bin,
            "-n", str(Path(net_path).resolve()),
            "-r", str(rou_path.resolve()),
            "-a", str(Path(vtype_path).resolve()),
            "--no-step-log", "true",
            "--time-to-teleport", str(time_to_teleport),
            "--step-length", str(step_length_s),
            "--tripinfo-output", str(tripinfo_path.resolve()),
        ]

        traci.start(sumo_cmd, label=sumo_label)
        conn = traci.getConnection(sumo_label)

        # Tracking state
        inserted_ids: set[str] = set()
        arrived_ids: set[str] = set()
        trapped_ids: set[str] = set()
        actual_depart_times: dict[str, float] = {}
        actual_arrival_times: dict[str, float] = {}

        # Subscription constant: we need VAR_ROAD_ID per vehicle
        # We subscribe at insertion and read bulk results each step.

        max_steps = int(horizon_s / step_length_s) + 1

        for _step in range(max_steps):
            conn.simulationStep()
            t_now = conn.simulation.getTime()

            # Register newly departed vehicles
            for vid in conn.simulation.getDepartedIDList():
                inserted_ids.add(vid)
                actual_depart_times[vid] = t_now
                # Subscribe to road id for this vehicle
                conn.vehicle.subscribe(vid, [tc.VAR_ROAD_ID])

            # Register newly arrived vehicles
            for vid in conn.simulation.getArrivedIDList():
                arrived_ids.add(vid)
                actual_arrival_times[vid] = t_now

            # Flood closure: check all subscribed (active) vehicles in bulk
            sub_results = conn.vehicle.getAllSubscriptionResults()
            to_remove: list[str] = []
            for vid, vals in sub_results.items():
                if vid in trapped_ids or vid in arrived_ids:
                    continue
                road_id: str = vals.get(tc.VAR_ROAD_ID, "")
                # Skip internal junction edges (start with ":")
                if road_id.startswith(":") or not road_id:
                    continue
                flood_t = flood_times_by_sumo_edge.get(road_id, float("inf"))
                if flood_t <= t_now:
                    to_remove.append(vid)

            for vid in to_remove:
                # Read road id before removal (already in sub_results)
                road_id = sub_results[vid].get(tc.VAR_ROAD_ID, "")
                trapped_ids.add(vid)
                conn.vehicle.remove(vid)
                aid = int(vid.replace("veh_", ""))
                outcomes[aid] = VehicleOutcome(
                    agent_id=aid,
                    status="TRAPPED",
                    trap_time_s=t_now,
                    trap_edge_sumo_id=road_id,
                    planned_depart_s=planned_depart.get(aid),
                    actual_depart_s=actual_depart_times.get(vid),
                )

            # Early termination: all inserted vehicles resolved
            active_in_network = conn.simulation.getMinExpectedNumber()
            if len(inserted_ids) >= len(written_ids) and active_in_network == 0:
                break

        conn.close()

        # Parse tripinfo for arrived vehicles
        tripinfo = _parse_tripinfo(tripinfo_path)

        # Build outcomes for ARRIVED vehicles
        for vid in arrived_ids:
            aid = int(vid.replace("veh_", ""))
            if aid in outcomes:
                continue  # already handled (should not happen)
            info = tripinfo.get(aid, {})
            outcomes[aid] = VehicleOutcome(
                agent_id=aid,
                status="ARRIVED",
                duration_s=info.get("duration"),
                waiting_time_s=info.get("waitingTime"),
                time_loss_s=info.get("timeLoss"),
                depart_delay_s=info.get("departDelay"),
                planned_depart_s=planned_depart.get(aid),
                actual_depart_s=actual_depart_times.get(vid),
                actual_arrival_s=actual_arrival_times.get(vid),
            )

        # Mark UNFINISHED vehicles (inserted but not arrived or trapped)
        for vid in inserted_ids:
            aid = int(vid.replace("veh_", ""))
            if aid not in outcomes:
                outcomes[aid] = VehicleOutcome(
                    agent_id=aid,
                    status="UNFINISHED",
                    planned_depart_s=planned_depart.get(aid),
                    actual_depart_s=actual_depart_times.get(vid),
                )

        # Mark any written_ids that were never inserted
        for aid in written_ids:
            if aid not in outcomes:
                outcomes[aid] = VehicleOutcome(
                    agent_id=aid,
                    status="NEVER_INSERTED",
                    planned_depart_s=planned_depart.get(aid),
                )

        return outcomes

    finally:
        # Close TraCI connection if still open
        try:
            traci.getConnection(sumo_label).close()
        except Exception:
            pass
        # Clean up owned temp directory
        if _own_tmpdir is not None:
            import shutil
            shutil.rmtree(_own_tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Convenience: build flood_times_by_sumo_edge from hazard + edge_map
# ---------------------------------------------------------------------------

def build_flood_time_map(
    hazard: Any,
    edge_map_s2g: dict[str, list[int]],
) -> dict[str, float]:
    """Build a dict mapping SUMO edge id -> flood arrival time (seconds).

    Parameters
    ----------
    hazard : HazardModel
        Hazard providing edge_arrival_time(u, v, k).
    edge_map_s2g : dict
        SUMO edge id -> [u, v, k] graph edge key (from edge_map.json).

    Returns
    -------
    dict[str, float]
        sumo_edge_id -> flood arrival time in seconds (math.inf if never floods).
    """
    import math
    flood_map: dict[str, float] = {}
    for sumo_id, (u, v, k) in edge_map_s2g.items():
        try:
            t_flood = hazard.edge_arrival_time(int(u), int(v), int(k))
        except Exception:
            t_flood = math.inf
        flood_map[sumo_id] = float(t_flood) if t_flood is not None else math.inf
    return flood_map

