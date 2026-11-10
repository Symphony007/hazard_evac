"""Point-queue (Vickrey) traffic state model and per-edge congestion tracking.

Modelling Assumptions (Point-Queue Model):
------------------------------------------
1. Edge Traversal and Bottleneck Discharge:
   Every edge e has free transit time:
       tau(e, t) = free_flow_time / vehicle_speed_scale / max(speed_factor(t), sf_floor)
   evaluated at entry time t. Vehicles discharge at the downstream end with
   minimum headway h(e) = 60.0 / flow_capacity(e) seconds (FIFO order).
2. Unlimited Storage (No Spatial Spillback):
   Vehicles stack vertically at the link exit point. Queue storage is unlimited;
   queues do not spill back onto upstream links or block upstream junctions.
   Edges where peak_occupancy exceeds physical storage capacity are flagged via
   diagnostics (n_edges_over_storage, max_congestion).
3. Junction and Exit Capacity:
   Intersections/nodes have no internal capacity or signal delays. Boundary exits
   have unlimited absorption capacity (immediate departure upon link exit).
4. Independent Links:
   Parallel edges between the same node pair (u, v, k) maintain distinct,
   independent queues.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx


class TrafficState:
    """Tracks edge exit schedules, queue delays, and vehicle occupancy.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network graph with 'flow_capacity' and 'capacity' edge attributes.
    """

    def __init__(self, G: nx.MultiDiGraph) -> None:
        self.G = G
        self.last_exit: dict[tuple[int, int, int], float] = {}
        self.headway: dict[tuple[int, int, int], float] = {}
        self.occupancy_queues: dict[tuple[int, int, int], deque[float]] = {}
        self.peak_occupancy: dict[tuple[int, int, int], int] = {}
        self.total_queue_delay: dict[tuple[int, int, int], float] = {}
        self.max_queue_wait: dict[tuple[int, int, int], float] = {}
        self.n_vehicles: dict[tuple[int, int, int], int] = {}

        for u, v, k, data in G.edges(keys=True, data=True):
            e = (u, v, k)
            self.last_exit[e] = float("-inf")
            fc = float(data.get("flow_capacity", 12.0))
            self.headway[e] = 60.0 / max(fc, 0.001)
            self.occupancy_queues[e] = deque()
            self.peak_occupancy[e] = 0
            self.total_queue_delay[e] = 0.0
            self.max_queue_wait[e] = 0.0
            self.n_vehicles[e] = 0

    def enter(self, edge: tuple[int, int, int], t: float, tau: float) -> float:
        """Commit vehicle entry onto *edge* at time *t* with transit duration *tau*.

        Parameters
        ----------
        edge : tuple[int, int, int]
            MultiDiGraph edge key (u, v, k).
        t : float
            Edge entry time in seconds.
        tau : float
            Free-flow traversal time (including hazard degradation) in seconds.

        Returns
        -------
        float
            Scheduled link discharge time t_exit in seconds.
        """
        h = self.headway[edge]
        prev_exit = self.last_exit[edge]
        t_exit = max(t + tau, prev_exit + h)
        queue_wait = max(0.0, t_exit - (t + tau))

        self.last_exit[edge] = t_exit
        self.total_queue_delay[edge] += queue_wait
        if queue_wait > self.max_queue_wait[edge]:
            self.max_queue_wait[edge] = queue_wait
        self.n_vehicles[edge] += 1

        # Drop vehicles discharged prior to current entry time t
        q = self.occupancy_queues[edge]
        while q and q[0] <= t:
            q.popleft()
        q.append(t_exit)

        occ = len(q)
        if occ > self.peak_occupancy[edge]:
            self.peak_occupancy[edge] = occ

        return t_exit

    def predicted_extra_delay(
        self,
        edge: tuple[int, int, int],
        t: float,
        tau: float,
    ) -> float:
        """Calculate queuing delay beyond free transit for an entry at *t* with transit *tau*.

        Parameters
        ----------
        edge : tuple[int, int, int]
            Edge key (u, v, k).
        t : float
            Anticipated entry time in seconds.
        tau : float
            Free-flow / degraded transit time in seconds.

        Returns
        -------
        float
            Anticipated queue delay in seconds (>= 0.0).
        """
        if edge not in self.headway:
            return 0.0
        h = self.headway[edge]
        prev_exit = self.last_exit[edge]
        return max(0.0, prev_exit + h - (t + tau))

    def extra_delay_s(
        self,
        u: int,
        v: int,
        k: int,
        t: float,
        tau: float | None = None,
    ) -> float:
        """TrafficView protocol method providing live queue delay estimates.

        Parameters
        ----------
        u, v, k : int
            Edge endpoints and key.
        t : float
            Current planning / entry time.
        tau : float, optional
            Estimated transit duration. If omitted, uses free_flow_time from graph.

        Returns
        -------
        float
            Estimated queuing delay in seconds.
        """
        edge = (u, v, k)
        if edge not in self.headway:
            return 0.0
        if tau is None:
            data = self.G.edges[u, v, k]
            tau = float(data.get("free_flow_time", data.get("travel_time", 1.0)))
        return self.predicted_extra_delay(edge, t, tau)

