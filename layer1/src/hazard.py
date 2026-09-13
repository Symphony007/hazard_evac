"""
hazard.py — Radius-based hazard spread simulation.

Models a hazard (e.g. flood, chemical spill, earthquake shockwave) that
originates at a single node and spreads outward at a constant rate over
discrete time steps.

At each time step *t*, the hazard occupies a circle of radius
``spread_rate_m × t`` metres centred on the origin node.  Any graph node
whose great-circle distance to the origin falls within that radius is
marked hazardous, along with every edge that touches a hazardous node.

Design notes (for viva)
-----------------------
* We use the **haversine formula** (great-circle distance on a sphere)
  because OSMnx graphs store coordinates in unprojected WGS-84 (EPSG:4326).
  This avoids the need for an expensive graph re-projection step.
* The simulation is *deterministic and memoryless* — the hazard state at
  time *t* depends only on *t*, not on any prior state.  This makes it
  trivial to query any arbitrary time step without running a loop.
* A more realistic model (anisotropic spread, wind direction, terrain
  resistance) is planned for Layer 2.

References
----------
- Haversine formula: https://en.wikipedia.org/wiki/Haversine_formula
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Set, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Data container for a single snapshot of the hazard
# ---------------------------------------------------------------------------

@dataclass
class HazardState:
    """
    Immutable snapshot of the hazard at a specific time step.

    Attributes
    ----------
    time_step : int
        The discrete time step this state corresponds to.
    radius_m : float
        Current hazard radius in metres.
    origin_node : int
        OSM node ID where the hazard originated.
    hazardous_nodes : set of int
        Node IDs that fall within the hazard radius.
    hazardous_edges : set of (int, int, int)
        Edge keys ``(u, v, key)`` where at least one endpoint is hazardous.
    """
    time_step: int
    radius_m: float
    origin_node: int
    hazardous_nodes: Set[int] = field(default_factory=set)
    hazardous_edges: Set[Tuple[int, int, int]] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Haversine distance (metres)
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the great-circle distance between two points on Earth.

    Parameters
    ----------
    lat1, lon1 : float
        Coordinates of point 1 in decimal degrees.
    lat2, lon2 : float
        Coordinates of point 2 in decimal degrees.

    Returns
    -------
    float
        Distance in metres.
    """
    R = 6_371_000  # Earth's mean radius in metres

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (math.sin(d_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# ---------------------------------------------------------------------------
# Simulation class
# ---------------------------------------------------------------------------

class HazardSimulation:
    """
    Simulate a radially-expanding hazard on a road-network graph.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The road-network graph (unprojected, WGS-84).
    origin_node : int
        OSM node ID where the hazard starts.
    spread_rate_m : float, default 200
        How many metres the hazard radius grows per time step.

    Example
    -------
    >>> sim = HazardSimulation(G, origin_node=123456, spread_rate_m=300)
    >>> state = sim.get_hazard_state(time_step=3)
    >>> print(state.radius_m)      # 900.0
    >>> print(len(state.hazardous_nodes))
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        origin_node: int,
        spread_rate_m: float = 200,
    ) -> None:
        if origin_node not in graph:
            raise ValueError(f"origin_node {origin_node} not found in graph")

        self.graph = graph
        self.origin_node = origin_node
        self.spread_rate_m = spread_rate_m

        # Cache the origin's coordinates (looked up once)
        node_data = graph.nodes[origin_node]
        self.origin_lat: float = node_data["y"]
        self.origin_lon: float = node_data["x"]

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_hazard_state(self, time_step: int) -> HazardState:
        """
        Compute the hazard state at a given discrete time step.

        The hazard radius is ``spread_rate_m × time_step``.  All nodes
        within that radius (great-circle) are marked hazardous, and every
        edge that has at least one hazardous endpoint is also flagged.

        Parameters
        ----------
        time_step : int
            Non-negative integer representing the current time step.
            At time_step=0 only the origin node itself is hazardous.

        Returns
        -------
        HazardState
            Snapshot of the hazard at this time step.
        """
        if time_step < 0:
            raise ValueError("time_step must be non-negative")

        radius = self.spread_rate_m * time_step
        hazardous_nodes: set[int] = set()

        # --- Mark nodes within radius ---
        for node, data in self.graph.nodes(data=True):
            dist = _haversine_m(
                self.origin_lat, self.origin_lon,
                data["y"], data["x"],
            )
            if dist <= radius:
                hazardous_nodes.add(node)

        # At t=0 the origin itself is always hazardous
        hazardous_nodes.add(self.origin_node)

        # --- Mark edges that touch a hazardous node ---
        hazardous_edges: set[tuple[int, int, int]] = set()
        for u, v, key in self.graph.edges(keys=True):
            if u in hazardous_nodes or v in hazardous_nodes:
                hazardous_edges.add((u, v, key))

        return HazardState(
            time_step=time_step,
            radius_m=radius,
            origin_node=self.origin_node,
            hazardous_nodes=hazardous_nodes,
            hazardous_edges=hazardous_edges,
        )
