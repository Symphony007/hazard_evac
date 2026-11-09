"""Scenario generation: boundary exit selection and evacuee agent sampling.

Exits model agents leaving the study area. In a bounded bounding-box graph,
boundary roads are truncated; placing exits along a peripheral ring ensures
agents travel outward toward the network perimeter.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Sequence

import networkx as nx
import numpy as np
import osmnx as ox

from src.routing.base import Evacuee

if TYPE_CHECKING:
    from src.config import ScenarioConfig, StudyAreaConfig
    from src.hazard.base import HazardModel


def build_exits(
    G: "nx.MultiDiGraph",
    hazard: "HazardModel",
    scenario_cfg: "ScenarioConfig",
    center_lat: float = 22.4985,
    center_lon: float = 88.3450,
    horizon_min: float = 60.0,
) -> list[int]:
    """Select n_exits on the perimeter ring using farthest-point sampling.

    Notes
    -----
    Exits model 'leaving the study area'. The square download box creates
    artificial boundary stubs at the study area perimeter; exits are
    deliberately placed on the outer boundary ring to represent evacuation
    routes leading outside the hazard impact zone.

    Procedure
    ---------
    1. Identify candidate boundary ring nodes with distance from study area center
       >= scenario_cfg.exit_min_dist_from_center_m.
    2. Preferred candidates: ring nodes with arrival_time > horizon_min * 60 (never
       flooded within the simulation horizon).
    3. Fallback: if fewer than n_exits preferred nodes exist (e.g. at extreme flood
       speeds), supplement with ring nodes having the latest arrival times, and
       emit a warning.
    4. Apply farthest-point sampling to select n_exits well-separated boundary nodes,
       seeded by the candidate farthest from the nearest flood source.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network graph with node lat/lon attributes 'y' and 'x'.
    hazard : HazardModel
        Hazard simulator providing node arrival times and sources.
    scenario_cfg : ScenarioConfig
        Scenario configuration containing n_exits and exit_min_dist_from_center_m.
    center_lat : float
        Latitude of the study area center.
    center_lon : float
        Longitude of the study area center.
    horizon_min : float
        Simulation time horizon in minutes.

    Returns
    -------
    list[int]
        List of selected exit node IDs (length equal to min(n_exits, available)).
    """
    n_exits = scenario_cfg.n_exits
    min_dist_m = scenario_cfg.exit_min_dist_from_center_m
    horizon_s = horizon_min * 60.0

    # 1. Candidate ring nodes
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

    # 2. Preferred: never flooded within horizon
    preferred = [n for n in ring_nodes if hazard.node_arrival_time(n) > horizon_s]

    if len(preferred) >= n_exits:
        candidate_pool = preferred
    else:
        n_needed = n_exits - len(preferred)
        # Sort remaining ring nodes by arrival time descending (latest flood first)
        remaining = [n for n in ring_nodes if hazard.node_arrival_time(n) <= horizon_s]
        remaining.sort(key=lambda n: hazard.node_arrival_time(n), reverse=True)
        fallback = remaining[:n_needed]
        print(
            f"[scenario] WARNING: Only {len(preferred)}/{n_exits} preferred unflooded exits "
            f"found on boundary ring; selected {len(fallback)} fallback exit(s) with latest arrival times."
        )
        candidate_pool = preferred + fallback

    if len(candidate_pool) <= n_exits:
        return candidate_pool

    # 4. Farthest-point sampling on candidate_pool
    # Deterministic seed: candidate with maximum distance to nearest flood source
    sources = getattr(hazard, "sources", [])
    if sources:
        def dist_to_nearest_source(node: int) -> float:
            lat = G.nodes[node]["y"]
            lon = G.nodes[node]["x"]
            return min(
                ox.distance.great_circle(lat, lon, G.nodes[s]["y"], G.nodes[s]["x"])
                for s in sources
            )
        candidate_pool.sort(key=lambda n: (dist_to_nearest_source(n), n), reverse=True)
        first_exit = candidate_pool[0]
    else:
        candidate_pool.sort()
        first_exit = candidate_pool[0]

    selected: list[int] = [first_exit]
    remaining_set = set(candidate_pool)
    remaining_set.remove(first_exit)

    # Pre-extract coordinates for fast distance calculation
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

    return selected


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
    if max_arrival <= (depart_window_s + 30.0):
        effective_window_s = max(0.0, max_arrival - 30.1)
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

    return agents


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
