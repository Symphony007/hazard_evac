"""
congestion.py -- BPR volume-delay congestion model for road networks.

Implements a traffic-engineering-standard Bureau of Public Roads (BPR)
cost function that makes road segments progressively more expensive as
more evacuees are routed onto them.  This is the key mechanism that
causes the multi-agent router to *diversify* routes rather than
funneling all evacuees onto the same shortest path.

BPR formula
-----------
    adjusted_cost = base_length * (1 + alpha * (volume / capacity) ^ beta)

Default parameters (US Bureau of Public Roads, 1964):
    alpha = 0.15
    beta  = 4.0

These are the standard defaults used in transportation engineering
worldwide and are well-accepted in the academic literature.

Edge capacity
-------------
Capacity is derived from the OSM ``lanes`` attribute when available.
Each lane is assigned a base capacity of 1800 vehicles/hour (the
Highway Capacity Manual default for a single lane of urban arterial).
When lane data is missing, a default capacity of 1 lane (1800 veh/hr)
is assumed.  For this simulation, capacity is treated as a relative
measure: what matters is the *ratio* volume/capacity, not the absolute
number.  We use a per-edge capacity as a simple integer representing
the number of simultaneous agents the edge can comfortably handle
before congestion effects kick in.

Design notes (for viva)
-----------------------
* The BPR function is convex and monotonically increasing, which
  guarantees that Dijkstra still finds the global optimum at each
  agent's routing step (no negative edges).
* Volume is tracked per directed edge ``(u, v, key)``.
* ``apply_to_graph`` writes a ``congestion_cost`` attribute onto every
  edge, so Dijkstra can use ``weight="congestion_cost"`` directly.

References
----------
- Bureau of Public Roads (1964). Traffic Assignment Manual.
- Highway Capacity Manual (HCM), Transportation Research Board.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default BPR parameters
# ---------------------------------------------------------------------------

DEFAULT_ALPHA = 0.15     # BPR shape parameter (congestion sensitivity)
DEFAULT_BETA = 4.0       # BPR exponent (non-linearity of delay growth)
DEFAULT_CAPACITY = 4     # agents per edge before congestion sets in
CAPACITY_PER_LANE = 4    # agents per lane (scaled for simulation size)


# ---------------------------------------------------------------------------
# Statistics container
# ---------------------------------------------------------------------------

@dataclass
class CongestionStats:
    """Summary statistics for the current congestion state."""
    total_edges: int
    loaded_edges: int         # edges with load > 0
    max_load: int             # highest load on any single edge
    max_load_edge: Tuple[int, int, int]  # (u, v, key) of the most loaded edge
    avg_load_on_used: float   # average load across edges that have load > 0
    total_volume: int         # sum of all edge loads


# ---------------------------------------------------------------------------
# Congestion model
# ---------------------------------------------------------------------------

class CongestionModel:
    """
    Track per-edge traffic load and compute BPR-adjusted travel costs.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The road-network graph.  Edge ``lanes`` attributes are used to
        derive per-edge capacity; if missing, ``DEFAULT_CAPACITY`` is used.
    alpha : float, default 0.15
        BPR alpha parameter (congestion sensitivity).
    beta : float, default 4.0
        BPR beta parameter (exponent controlling non-linearity).
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
    ) -> None:
        self.graph = graph
        self.alpha = alpha
        self.beta = beta

        # Per-edge load: (u, v, key) -> number of agents currently using it
        self._load: Dict[Tuple[int, int, int], int] = defaultdict(int)

        # Pre-compute per-edge capacity from OSM lane data
        self._capacity: Dict[Tuple[int, int, int], int] = {}
        for u, v, key, data in graph.edges(keys=True, data=True):
            lanes = self._parse_lanes(data.get("lanes", None))
            self._capacity[(u, v, key)] = lanes * CAPACITY_PER_LANE

    # -------------------------------------------------------------------
    # Lane parsing
    # -------------------------------------------------------------------

    @staticmethod
    def _parse_lanes(lanes_attr) -> int:
        """
        Parse the OSM ``lanes`` attribute into an integer.

        OSM stores lanes as strings, sometimes with semicolons for
        segments with varying lane counts (e.g. "2;3").  We take the
        maximum in that case.  Falls back to 1 if unparseable or missing.
        """
        if lanes_attr is None:
            return 1
        try:
            # Handle "2;3" -> max(2, 3) = 3
            if isinstance(lanes_attr, str) and ";" in lanes_attr:
                return max(int(x) for x in lanes_attr.split(";"))
            return max(1, int(float(lanes_attr)))
        except (ValueError, TypeError):
            return 1

    # -------------------------------------------------------------------
    # Load tracking
    # -------------------------------------------------------------------

    def add_route(self, route: List[int]) -> None:
        """
        Register a route: increment load on every edge along the path.

        Parameters
        ----------
        route : list of int
            Ordered sequence of node IDs forming the route.
        """
        for u, v in zip(route[:-1], route[1:]):
            # Use key=0 (the dominant case in simplified OSM graphs)
            self._load[(u, v, 0)] += 1

    def remove_route(self, route: List[int]) -> None:
        """
        Unregister a route: decrement load on every edge along the path.

        Parameters
        ----------
        route : list of int
            Ordered sequence of node IDs forming the route.
        """
        for u, v in zip(route[:-1], route[1:]):
            edge_key = (u, v, 0)
            if self._load[edge_key] > 0:
                self._load[edge_key] -= 1

    def get_load(self, u: int, v: int, key: int = 0) -> int:
        """Return the current load (number of agents) on edge (u, v, key)."""
        return self._load[(u, v, key)]

    @property
    def max_load(self) -> int:
        """Return the maximum load on any single edge."""
        return max(self._load.values()) if self._load else 0

    # -------------------------------------------------------------------
    # BPR cost computation
    # -------------------------------------------------------------------

    def get_adjusted_cost(
        self, u: int, v: int, key: int = 0, base_length: float | None = None
    ) -> float:
        """
        Compute the BPR-adjusted traversal cost for a single edge.

        Parameters
        ----------
        u, v, key : int
            Edge identifier.
        base_length : float or None
            If None, read from the graph's ``length`` attribute.

        Returns
        -------
        float
            The congestion-adjusted cost:
            ``base_length * (1 + alpha * (load/capacity)^beta)``
        """
        if base_length is None:
            try:
                base_length = float(self.graph[u][v][key]["length"])
            except (KeyError, ValueError):
                base_length = 1.0  # fallback

        load = self._load[(u, v, key)]
        capacity = self._capacity.get((u, v, key), DEFAULT_CAPACITY)

        # BPR formula: cost = base * (1 + alpha * (vol/cap)^beta)
        ratio = load / capacity
        cost = base_length * (1.0 + self.alpha * (ratio ** self.beta))
        return cost

    # -------------------------------------------------------------------
    # Bulk graph update
    # -------------------------------------------------------------------

    def apply_to_graph(self, graph: nx.MultiDiGraph | None = None) -> nx.MultiDiGraph:
        """
        Write ``congestion_cost`` attributes onto every edge of a graph copy.

        The router can then use ``weight="congestion_cost"`` to route
        with congestion awareness.

        Parameters
        ----------
        graph : networkx.MultiDiGraph or None
            If None, use ``self.graph``.  A **copy** is always returned.

        Returns
        -------
        networkx.MultiDiGraph
            Graph copy with ``congestion_cost`` edge attributes set.
        """
        G = (graph or self.graph).copy()

        for u, v, key, data in G.edges(keys=True, data=True):
            base_length = float(data.get("length", 1.0))
            G[u][v][key]["congestion_cost"] = self.get_adjusted_cost(
                u, v, key, base_length=base_length
            )

        return G

    # -------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------

    def get_stats(self) -> CongestionStats:
        """
        Compute summary statistics for the current congestion state.

        Returns
        -------
        CongestionStats
            Dataclass with total_edges, loaded_edges, max_load, etc.
        """
        total_edges = self.graph.number_of_edges()
        loaded = {k: v for k, v in self._load.items() if v > 0}
        loaded_edges = len(loaded)

        if loaded:
            max_edge = max(loaded, key=loaded.get)
            max_load_val = loaded[max_edge]
            avg_load = sum(loaded.values()) / loaded_edges
            total_vol = sum(loaded.values())
        else:
            max_edge = (0, 0, 0)
            max_load_val = 0
            avg_load = 0.0
            total_vol = 0

        return CongestionStats(
            total_edges=total_edges,
            loaded_edges=loaded_edges,
            max_load=max_load_val,
            max_load_edge=max_edge,
            avg_load_on_used=avg_load,
            total_volume=total_vol,
        )
