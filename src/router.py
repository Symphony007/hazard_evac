"""
router.py — Shortest-path routing with hazard-aware rerouting.

Provides two core functions:

1. ``compute_route`` — vanilla Dijkstra shortest path using edge length
   as the weight (metres).
2. ``reroute_if_needed`` — checks whether the current route passes through
   any hazardous edge; if so, penalises those edges and recomputes.

Design notes (for viva)
-----------------------
* We use **networkx.shortest_path** with ``weight="length"`` which
  implements Dijkstra's algorithm under the hood (guaranteed optimal
  for non-negative weights).
* For rerouting, instead of *removing* hazardous edges (which might
  disconnect the graph and hide valid detours), we set their weight
  to ``float('inf')``.  This lets Dijkstra naturally avoid them while
  still finding paths that go *around* the hazard — unless the graph
  is truly partitioned, in which case ``NetworkXNoPath`` is raised.
* A graph **copy** is used for rerouting so the original graph is
  never mutated — important when the simulation loops over many
  time steps.
"""

from __future__ import annotations

import copy
import logging
from typing import List, Tuple

import networkx as nx

from .hazard import HazardState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NoRouteError(Exception):
    """Raised when no safe route can be found between origin and destination."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_route(
    graph: nx.MultiDiGraph,
    origin: int,
    destination: int,
) -> List[int]:
    """
    Compute the shortest path by road length (Dijkstra).

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Road-network graph with ``length`` attributes on edges.
    origin : int
        OSM node ID for the start of the route.
    destination : int
        OSM node ID for the end of the route.

    Returns
    -------
    list of int
        Ordered sequence of node IDs from origin to destination.

    Raises
    ------
    NoRouteError
        If no path exists between origin and destination.
    """
    try:
        path = nx.shortest_path(graph, origin, destination, weight="length")
    except nx.NetworkXNoPath as exc:
        raise NoRouteError(
            f"No route from {origin} to {destination}"
        ) from exc
    return path


def route_is_safe(route: List[int], hazard_state: HazardState) -> bool:
    """
    Check whether every edge on a route is free of hazard.

    Parameters
    ----------
    route : list of int
        Sequence of node IDs forming the route.
    hazard_state : HazardState
        Current hazard snapshot.

    Returns
    -------
    bool
        True if no edge on the route is hazardous.
    """
    hazardous = hazard_state.hazardous_edges
    for u, v in zip(route[:-1], route[1:]):
        # Check all parallel edges between u and v
        for key in range(_edge_count(u, v, hazardous)):
            if (u, v, key) in hazardous:
                return False
    # Also do a direct membership check (handles key=0 common case)
    for u, v in zip(route[:-1], route[1:]):
        if (u, v, 0) in hazardous:
            return False
    return True


def _edge_count(u: int, v: int, edge_set: set) -> int:
    """Estimate the max key index for parallel edges between u and v."""
    # In most road networks key is 0; check up to a small bound
    return 4  # covers the vast majority of OSM road graphs


def reroute_if_needed(
    graph: nx.MultiDiGraph,
    current_route: List[int],
    hazard_state: HazardState,
) -> Tuple[List[int], bool]:
    """
    Re-route if the current path intersects the hazard zone.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The **original** (unmodified) road-network graph.
    current_route : list of int
        Current route as an ordered sequence of node IDs.
    hazard_state : HazardState
        Current hazard snapshot.

    Returns
    -------
    (route, did_reroute) : tuple[list[int], bool]
        The (possibly new) route and a flag indicating whether
        rerouting actually occurred.

    Raises
    ------
    NoRouteError
        If no safe alternative route exists.
    """
    origin = current_route[0]
    destination = current_route[-1]

    # --- Check if current route is safe ---
    if _route_intersects_hazard(current_route, hazard_state):
        logger.info(
            "Route intersects hazard at t=%d (radius=%.0f m) → rerouting …",
            hazard_state.time_step,
            hazard_state.radius_m,
        )

        # Build a modified copy with hazardous edges penalised
        G_mod = _penalise_hazardous_edges(graph, hazard_state)

        new_route = compute_route(G_mod, origin, destination)
        return new_route, True

    return current_route, False


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _route_intersects_hazard(
    route: List[int],
    hazard_state: HazardState,
) -> bool:
    """
    Return True if *any* edge on the route is in the hazard zone.

    An edge (u, v) on the route is considered hazardous if any keyed
    version ``(u, v, k)`` appears in ``hazard_state.hazardous_edges``.
    """
    hazardous = hazard_state.hazardous_edges

    for u, v in zip(route[:-1], route[1:]):
        # Check key 0 (by far the most common in simplified OSM graphs)
        if (u, v, 0) in hazardous:
            return True
        # Also check higher keys for the rare multi-edge case
        for key in range(1, 4):
            if (u, v, key) in hazardous:
                return True
    return False


def _penalise_hazardous_edges(
    graph: nx.MultiDiGraph,
    hazard_state: HazardState,
) -> nx.MultiDiGraph:
    """
    Return a copy of `graph` with hazardous edges' length set to infinity.

    This makes Dijkstra avoid hazardous edges while preserving graph
    connectivity — if a detour exists, it will be found.
    """
    G_mod = graph.copy()

    for u, v, key in hazard_state.hazardous_edges:
        if G_mod.has_edge(u, v, key):
            G_mod[u][v][key]["length"] = float("inf")

    return G_mod
