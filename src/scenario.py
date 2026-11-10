"""Scenario generation: boundary exit selection and evacuee agent sampling.

Exits model agents leaving the study area. In a bounded bounding-box graph,
boundary roads are truncated; placing exits along a peripheral ring ensures
agents travel outward toward the network perimeter.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Sequence

import networkx as nx
import numpy as np
import osmnx as ox

from src.routing.base import Evacuee

if TYPE_CHECKING:
    from src.config import ScenarioConfig, StudyAreaConfig
    from src.hazard.base import HazardModel


class ExitList(list):
    """List of exit nodes carrying sector and fallback diagnostics.

    Attributes
    ----------
    used_fallback_exits : bool
        True if any exit was selected from flooded ring nodes as a fallback.
    n_fallback_exits : int
        Number of fallback exits selected.
    n_exits_effective : int
        Number of effective exits returned.
    sectors_with_exit : list[int]
        List of sector indices that have an exit.
    sectors_dropped : list[int]
        List of sector indices dropped due to early flooding.
    safe_time_s : float
        Threshold time (s) required for an exit to be considered safe.
    info : dict[str, Any]
        Dictionary summary of exit diagnostics.
    """

    used_fallback_exits: bool = False
    n_fallback_exits: int = 0
    n_exits_effective: int = 0
    sectors_with_exit: list[int]
    sectors_dropped: list[int]
    safe_time_s: float = 0.0
    info: dict[str, Any]

    def __init__(
        self,
        items: Sequence[int],
        used_fallback: bool = False,
        n_fallback: int = 0,
        info: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(items)
        self.used_fallback_exits = bool(used_fallback)
        self.n_fallback_exits = int(n_fallback)
        self.n_exits_effective = len(items)
        if info is not None:
            self.info = dict(info)
            self.sectors_with_exit = list(info.get("sectors_with_exit", []))
            self.sectors_dropped = list(info.get("sectors_dropped", []))
            self.safe_time_s = float(info.get("safe_time_s", 0.0))
        else:
            self.info = {
                "n_exits_effective": len(items),
                "sectors_with_exit": [],
                "sectors_dropped": [],
                "used_fallback_exits": self.used_fallback_exits,
                "n_fallback_exits": self.n_fallback_exits,
                "safe_time_s": 0.0,
            }
            self.sectors_with_exit = []
            self.sectors_dropped = []
            self.safe_time_s = 0.0


class AgentList(list):
    """List of generated evacuee agents carrying scenario diagnostics.

    Attributes
    ----------
    departure_window_capped : bool
        True if the departure window was capped because all origins flood early.
    effective_window_s : float
        Actual maximum departure time sampled.
    """

    departure_window_capped: bool = False
    effective_window_s: float = 0.0

    def __init__(
        self,
        items: Sequence[Evacuee],
        capped: bool = False,
        window_s: float = 0.0,
    ) -> None:
        super().__init__(items)
        self.departure_window_capped = bool(capped)
        self.effective_window_s = float(window_s)


def exit_spread_report(
    G: "nx.MultiDiGraph",
    exits: Sequence[int],
    cfg: Any = None,
    center_lat: float = 22.4985,
    center_lon: float = 88.3450,
) -> dict[str, float | int]:
    """Compute geometric and angular spread diagnostics for boundary exits.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network graph containing node coordinates 'y' (lat) and 'x' (lon).
    exits : Sequence[int]
        List of selected exit node IDs.
    cfg : Any, optional
        Configuration object (e.g. StudyAreaConfig) containing center_lat and center_lon.
    center_lat : float, optional
        Fallback latitude of study-area center.
    center_lon : float, optional
        Fallback longitude of study-area center.

    Returns
    -------
    dict
        Diagnostics containing:
        - pairwise_mean_dist_m: Mean great-circle distance between exit pairs in metres.
        - pairwise_min_dist_m: Minimum great-circle distance between any exit pair in metres.
        - angular_coverage_deg: Angular coverage (360 - largest empty gap) in degrees.
        - largest_gap_deg: Largest empty angular gap between consecutive exits in degrees.
        - n_distinct_sectors: Number of distinct 45-degree sectors (out of 8) containing an exit.
    """
    if cfg is not None:
        center_lat = getattr(cfg, "center_lat", center_lat)
        center_lon = getattr(cfg, "center_lon", center_lon)

    if len(exits) < 2:
        return {
            "pairwise_mean_dist_m": 0.0,
            "pairwise_min_dist_m": 0.0,
            "angular_coverage_deg": 0.0,
            "largest_gap_deg": 360.0,
            "n_distinct_sectors": len(exits),
        }

    # 1. Pairwise distances
    coords = [(G.nodes[e]["y"], G.nodes[e]["x"]) for e in exits]
    distances = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = ox.distance.great_circle(
                coords[i][0], coords[i][1], coords[j][0], coords[j][1]
            )
            distances.append(d)

    mean_dist = float(np.mean(distances)) if distances else 0.0
    min_dist = float(np.min(distances)) if distances else 0.0

    # 2. Angles relative to center
    angles = []
    for lat, lon in coords:
        dy = lat - center_lat
        dx = lon - center_lon
        deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        angles.append(deg)

    angles.sort()
    gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
    gaps.append((angles[0] + 360.0) - angles[-1])
    largest_gap = float(max(gaps))
    coverage = float(360.0 - largest_gap)

    # 3. 8 equal angular sectors (45 degrees each: 0..7)
    sectors = {int(a // 45) % 8 for a in angles}
    n_sectors = len(sectors)

    return {
        "pairwise_mean_dist_m": mean_dist,
        "pairwise_min_dist_m": min_dist,
        "angular_coverage_deg": coverage,
        "largest_gap_deg": largest_gap,
        "n_distinct_sectors": n_sectors,
    }


def build_exits(
    G: "nx.MultiDiGraph",
    hazard: "HazardModel",
    scenario_cfg: "ScenarioConfig",
    center_lat: float = 22.4985,
    center_lon: float = 88.3450,
    horizon_min: float = 60.0,
    return_details: bool = False,
) -> tuple[ExitList, dict[str, Any]]:
    """Select perimeter ring exits using sector-based placement or legacy latest-arrival.

    Notes
    -----
    Exits model agents leaving the study area across the network perimeter.
    Because exit safety depends on the hazard spread across the road network,
    exits are selected per (flood speed, seed).

    Exit Modes
    ----------
    - 'sector': Divides the boundary perimeter into n_exits equal angular sectors.
      In each sector, selects the node with the latest arrival time among those
      safe past safe_time_s = (depart_window_min + exit_safe_margin_min) * 60.
      Sectors with no qualifying safe nodes are dropped. If fewer than
      min_exits_required sectors qualify, falls back to latest arrival node per sector.
    - 'legacy_latest': Prioritizes boundary nodes never flooded within horizon_min,
      falling back to the latest-flooding ring nodes via farthest-point sampling.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network graph with node lat/lon attributes 'y' and 'x'.
    hazard : HazardModel
        Hazard simulator providing node arrival times and sources.
    scenario_cfg : ScenarioConfig
        Scenario configuration containing n_exits, exit_mode, exit_min_dist_from_center_m, etc.
    center_lat : float
        Latitude of the study area center.
    center_lon : float
        Longitude of the study area center.
    horizon_min : float
        Simulation time horizon in minutes.
    return_details : bool
        Maintained for backwards-compatibility; function returns tuple of (exits, exit_info).

    Returns
    -------
    tuple[ExitList, dict[str, Any]]
        Selected exit node IDs (ExitList instance) and info dict containing:
        - n_exits_effective: int
        - sectors_with_exit: list[int]
        - sectors_dropped: list[int]
        - used_fallback_exits: bool
        - n_fallback_exits: int
        - safe_time_s: float
    """
    n_exits = scenario_cfg.n_exits
    min_dist_m = scenario_cfg.exit_min_dist_from_center_m
    exit_mode = getattr(scenario_cfg, "exit_mode", "sector")

    # 1. Candidate ring nodes >= exit_min_dist_from_center_m from center
    ring_nodes: list[int] = []
    for n, data in G.nodes(data=True):
        lat = data["y"]
        lon = data["x"]
        d = ox.distance.great_circle(lat, lon, center_lat, center_lon)
        if d >= min_dist_m:
            ring_nodes.append(n)

    if not ring_nodes:
        raise RuntimeError(
            f"No boundary nodes found >= {min_dist_m}m from center ({center_lat}, {center_lon})"
        )

    # Pre-extract sources for tie-breaking
    sources = getattr(hazard, "sources", [])

    def _dist_to_source(node: int) -> float:
        if not sources:
            return 0.0
        lat = G.nodes[node]["y"]
        lon = G.nodes[node]["x"]
        return min(
            ox.distance.great_circle(lat, lon, G.nodes[s]["y"], G.nodes[s]["x"])
            for s in sources
        )

    # -------------------------------------------------------------------------
    # Mode A: Legacy Latest-Arrival Farthest-Point Sampling
    # -------------------------------------------------------------------------
    if exit_mode == "legacy_latest":
        horizon_s = horizon_min * 60.0
        preferred = [n for n in ring_nodes if hazard.node_arrival_time(n) > horizon_s]
        used_fallback = False
        n_fallback = 0

        if len(preferred) >= n_exits:
            candidate_pool = preferred
        else:
            n_needed = n_exits - len(preferred)
            remaining = [n for n in ring_nodes if hazard.node_arrival_time(n) <= horizon_s]
            remaining.sort(key=lambda n: hazard.node_arrival_time(n), reverse=True)
            fallback = remaining[:n_needed]
            used_fallback = True
            n_fallback = len(fallback)
            print(
                f"[scenario] WARNING: Only {len(preferred)}/{n_exits} preferred unflooded exits "
                f"found on boundary ring; selected {len(fallback)} fallback exit(s) with latest arrival times."
            )
            candidate_pool = preferred + fallback

        if len(candidate_pool) <= n_exits:
            selected = list(candidate_pool)
        else:
            if sources:
                candidate_pool.sort(key=lambda n: (_dist_to_source(n), n), reverse=True)
                first_exit = candidate_pool[0]
            else:
                candidate_pool.sort()
                first_exit = candidate_pool[0]

            selected = [first_exit]
            remaining_set = set(candidate_pool)
            remaining_set.remove(first_exit)
            coords = {n: (G.nodes[n]["y"], G.nodes[n]["x"]) for n in candidate_pool}

            while len(selected) < n_exits and remaining_set:
                best_candidate: int | None = None
                best_min_dist = -1.0
                for cand in sorted(remaining_set):
                    lat_c, lon_c = coords[cand]
                    min_dist_to_selected = min(
                        ox.distance.great_circle(lat_c, lon_c, coords[sel][0], coords[sel][1])
                        for sel in selected
                    )
                    if min_dist_to_selected > best_min_dist:
                        best_min_dist = min_dist_to_selected
                        best_candidate = cand
                if best_candidate is None:
                    break
                selected.append(best_candidate)
                remaining_set.remove(best_candidate)

        info = {
            "n_exits_effective": len(selected),
            "sectors_with_exit": [],
            "sectors_dropped": [],
            "used_fallback_exits": bool(used_fallback),
            "n_fallback_exits": int(n_fallback),
            "safe_time_s": float(horizon_s),
        }
        res = ExitList(selected, used_fallback=used_fallback, n_fallback=n_fallback, info=info)
        return res, info

    # -------------------------------------------------------------------------
    # Mode B: Sector-Based Exits
    # -------------------------------------------------------------------------
    # NOTE: Exits are chosen per (flood speed, seed) because they depend on the hazard.
    lat0_rad = math.radians(center_lat)
    cos_lat0 = math.cos(lat0_rad)

    # Group ring nodes into n_exits equal angular sectors
    sector_width = 360.0 / n_exits
    sector_nodes: dict[int, list[int]] = {s: [] for s in range(n_exits)}

    for n in ring_nodes:
        lat = G.nodes[n]["y"]
        lon = G.nodes[n]["x"]
        dx = (lon - center_lon) * cos_lat0
        dy = lat - center_lat
        angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        sec_idx = int(angle // sector_width)
        if sec_idx >= n_exits:
            sec_idx = n_exits - 1
        sector_nodes[sec_idx].append(n)

    safe_margin_min = getattr(scenario_cfg, "exit_safe_margin_min", 15.0)
    safe_time_s = (scenario_cfg.depart_window_min + safe_margin_min) * 60.0
    min_exits_req = getattr(scenario_cfg, "min_exits_required", 3)

    # Tie-break key: latest arrival time, farthest from nearest source, lowest node id
    def _node_priority_key(n: int) -> tuple[float, float, int]:
        return (hazard.node_arrival_time(n), _dist_to_source(n), -int(n))

    selected_exits: list[int] = []
    sectors_with_exit: list[int] = []
    sectors_dropped: list[int] = []

    for s in range(n_exits):
        qualifying = [n for n in sector_nodes[s] if hazard.node_arrival_time(n) >= safe_time_s]
        if qualifying:
            best_node = max(qualifying, key=_node_priority_key)
            selected_exits.append(best_node)
            sectors_with_exit.append(s)
        else:
            sectors_dropped.append(s)

    used_fallback = False
    n_fallback = 0

    # Fallback if fewer than min_exits_required sectors qualify
    if len(sectors_with_exit) < min_exits_req:
        used_fallback = True
        print(
            f"[scenario] WARNING: Only {len(sectors_with_exit)}/{min_exits_req} sectors had safe exits "
            f"(threshold {safe_time_s/60:.1f} min); triggered fallback across all sectors."
        )
        selected_exits = []
        sectors_with_exit = []
        sectors_dropped = []
        for s in range(n_exits):
            if sector_nodes[s]:
                best_node = max(sector_nodes[s], key=_node_priority_key)
                selected_exits.append(best_node)
                sectors_with_exit.append(s)
            else:
                sectors_dropped.append(s)
        n_fallback = sum(1 for n in selected_exits if hazard.node_arrival_time(n) < safe_time_s)

    info = {
        "n_exits_effective": len(selected_exits),
        "sectors_with_exit": sorted(sectors_with_exit),
        "sectors_dropped": sorted(sectors_dropped),
        "used_fallback_exits": bool(used_fallback),
        "n_fallback_exits": int(n_fallback),
        "safe_time_s": float(safe_time_s),
    }
    res = ExitList(selected_exits, used_fallback=used_fallback, n_fallback=n_fallback, info=info)
    return res, info


def generate_agents(
    G: "nx.MultiDiGraph",
    hazard: "HazardModel",
    exits: set[int] | Sequence[int],
    scenario_cfg: "ScenarioConfig",
    rng: np.random.Generator,
) -> list[Evacuee]:
    """Generate evacuee agents with uniform departure delays and origins.

    Origins are sampled uniformly from nodes that are not exits and satisfy
    hazard.node_arrival_time(n) > depart_time_s + 30 (not already flooded at
    departure).

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network graph.
    hazard : HazardModel
        Hazard simulator providing node arrival times.
    exits : set[int] or Sequence[int]
        Set of safe exit nodes.
    scenario_cfg : ScenarioConfig
        Scenario configuration (n_evacuees, depart_window_min).
    rng : np.random.Generator
        Seeded NumPy random number generator for reproducible sampling.

    Returns
    -------
    list[Evacuee]
        List of Evacuee instances with deterministic IDs, origins, and departure times.
    """
    exits_set = set(exits)
    all_candidate_nodes = sorted([n for n in G.nodes if n not in exits_set])
    depart_window_s = scenario_cfg.depart_window_min * 60.0

    # Ensure departure window does not exceed the time when all origins are flooded
    finite_arrivals = [
        hazard.node_arrival_time(n) for n in all_candidate_nodes
        if np.isfinite(hazard.node_arrival_time(n))
    ]
    max_arrival = max(finite_arrivals) if finite_arrivals else float("inf")
    capped = False
    if max_arrival <= (depart_window_s + 30.0):
        effective_window_s = max(0.0, max_arrival - 30.1)
        capped = True
        print(
            f"[scenario] WARNING: All origins flooded by t={max_arrival:.1f}s; "
            f"capping departure window to {effective_window_s:.1f}s (was {depart_window_s:.1f}s)."
        )
    else:
        effective_window_s = depart_window_s

    agents: list[Evacuee] = []
    for agent_id in range(scenario_cfg.n_evacuees):
        depart_time_s = float(rng.uniform(0.0, effective_window_s))
        # Valid origins must not be flooded within 30s of departure
        eligible = [
            n for n in all_candidate_nodes
            if hazard.node_arrival_time(n) > (depart_time_s + 30.0)
        ]
        if not eligible:
            raise RuntimeError(
                f"No eligible unflooded origins found for agent {agent_id} at t={depart_time_s:.1f}s"
            )
        idx = int(rng.integers(0, len(eligible)))
        origin = eligible[idx]
        agents.append(
            Evacuee(
                id=agent_id,
                origin=origin,
                depart_time_s=depart_time_s,
            )
        )

    return AgentList(agents, capped=capped, window_s=effective_window_s)


def shortest_baseline_length(
    G: "nx.MultiDiGraph",
    exits: set[int] | Sequence[int],
) -> dict[int, float]:
    """Compute the no-hazard shortest road length from every node to its nearest exit.

    Uses Dijkstra's algorithm on G.reverse(copy=False) with source nodes set to
    exits and weight='length'.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network graph.
    exits : set[int] or Sequence[int]
        Target exit node IDs.

    Returns
    -------
    dict[int, float]
        Mapping from node ID to minimum distance in metres to the nearest exit.
    """
    exits_set = set(exits)
    # Reverse graph view (O(1) memory, no copying)
    G_rev = G.reverse(copy=False)
    lengths = dict(
        nx.multi_source_dijkstra_path_length(
            G_rev,
            sources=exits_set,
            weight="length",
        )
    )
    return lengths
