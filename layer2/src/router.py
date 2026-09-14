"""
router.py -- Shortest-path routing with hazard-aware rerouting (Layer 2 copy).

Copied from Layer 1 and kept as-is.  The multi-agent routing logic that
builds on top of this lives in ``multi_agent_router.py``.

Provides:
  - ``compute_route``       -- vanilla Dijkstra shortest path (weight=length)
  - ``reroute_if_needed``   -- check + reroute around hazardous edges

Design notes (for viva)
-----------------------
* Dijkstra via ``networkx.shortest_path`` with ``weight="length"``.
* Hazardous edges are penalised (length -> inf), not removed, so the
  graph stays connected and Dijkstra can still find detours.
* A graph **copy** is used so the original is never mutated.
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
    weight: str = "length",
) -> List[int]:
    """
    Compute the shortest path using Dijkstra.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Road-network graph with the specified weight attribute on edges.
    origin : int
        OSM node ID for the start of the route.
    destination : int
        OSM node ID for the end of the route.
    weight : str, default "length"
        Edge attribute to use as the routing cost.  Layer 2 passes
        "congestion_cost" here when congestion is active.

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
        path = nx.shortest_path(graph, origin, destination, weight=weight)
    except nx.NetworkXNoPath as exc:
        raise NoRouteError(
            f"No route from {origin} to {destination}"
        ) from exc
    return path


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

    if _route_intersects_hazard(current_route, hazard_state):
        logger.info(
            "Route intersects hazard at t=%d (radius=%.0f m) -> rerouting ...",
            hazard_state.time_step,
            hazard_state.radius_m,
        )

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
    """Return True if *any* edge on the route is in the hazard zone."""
    hazardous = hazard_state.hazardous_edges
    for u, v in zip(route[:-1], route[1:]):
        if (u, v, 0) in hazardous:
            return True
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
    """
    G_mod = graph.copy()
    for u, v, key in hazard_state.hazardous_edges:
        if G_mod.has_edge(u, v, key):
            G_mod[u][v][key]["length"] = float("inf")
    return G_mod
