"""Discharge capacity reservation table for coordinated multi-agent routing.

Models road edge flow capacity via discrete discharge slots of width
h(e) = 60 / flow_capacity(e) seconds, corresponding to the point-queue
discharge headway.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx


class ReservationTable:
    """Per-edge discharge time reservation table.

    Each road edge maintains a sorted sequence of reserved discharge times.
    To satisfy capacity constraints, consecutive discharges on edge e must be
    separated by at least h(e) = 60 / flow_capacity(e) seconds.

    FIFO / Monotonicity Property:
    -----------------------------
    The mapping y -> earliest_exit(e, y) is monotonically non-decreasing in y.
    If vehicle A arrives at the discharge point of edge e earlier than vehicle B
    (y_A <= y_B), earliest_exit(e, y_A) <= earliest_exit(e, y_B) because conflicting
    reservations only advance the departure time to the end of the conflicting slot,
    preserving non-decreasing order. This ensures the FIFO property holds, making
    standard label-setting Dijkstra's algorithm exact and optimal.
    """

    def __init__(
        self,
        G: "nx.MultiDiGraph | None" = None,
        default_headway_s: float = 5.0,
        eps: float = 1e-9,
    ) -> None:
        """Initialize the reservation table.

        Parameters
        ----------
        G : nx.MultiDiGraph, optional
            Road graph from which edge flow capacities and headways are extracted.
        default_headway_s : float
            Fallback headway in seconds when edge is not in graph or headway is unknown.
        eps : float
            Numerical tolerance for slot boundary comparisons (default 1e-9).
        """
        self.G = G
        self.default_headway_s = float(default_headway_s)
        self.eps = float(eps)
        self._reservations: dict[Any, list[float]] = defaultdict(list)
        self._headways: dict[Any, float] = {}

        if G is not None:
            for u, v, k, data in G.edges(keys=True, data=True):
                fc = float(data.get("flow_capacity", 12.0))
                h = 60.0 / max(fc, 1e-6)
                self._headways[(u, v, k)] = h

    def set_headway(self, edge: Any, headway_s: float) -> None:
        """Explicitly set the discharge headway for an edge."""
        self._headways[edge] = float(headway_s)

    def get_headway(self, edge: Any, headway_s: float | None = None) -> float:
        """Return the discharge headway for an edge."""
        if headway_s is not None:
            return float(headway_s)
        if edge in self._headways:
            return self._headways[edge]
        if isinstance(edge, tuple) and len(edge) == 2 and self.G is not None:
            k0 = (edge[0], edge[1], 0)
            if k0 in self._headways:
                return self._headways[k0]
        return self.default_headway_s

    def earliest_exit(
        self,
        edge: Any,
        y: float,
        headway_s: float | None = None,
    ) -> float:
        """Find the smallest exit time x >= y such that |x - r| >= h for all reserved r.

        Uses bisection to locate the nearest reservations and walks forward
        resolving any contiguous conflicting reservation windows. Gaps between
        existing reservations are utilized if wide enough (>= 2h).

        Parameters
        ----------
        edge : Any
            Edge key, typically (u, v, k).
        y : float
            Earliest possible exit time (arrival at tail + free transit traversal).
        headway_s : float, optional
            Override headway in seconds.

        Returns
        -------
        float
            Smallest feasible conflict-free exit time x >= y.
        """
        h = self.get_headway(edge, headway_s)
        res = self._reservations.get(edge)
        if not res:
            return float(y)

        x = float(y)
        idx = bisect.bisect_right(res, x)

        # Check if x conflicts with the preceding reservation
        if idx > 0:
            prev_r = res[idx - 1]
            if x < prev_r + h - self.eps:
                x = prev_r + h

        # Walk forward while x conflicts with subsequent reservations
        while idx < len(res) and x > res[idx] - h + self.eps:
            if x < res[idx] + h - self.eps:
                x = res[idx] + h
            idx += 1

        return float(x)

    def reserve(self, edge: Any, x: float) -> None:
        """Insert a reserved discharge time x into the edge's reservation schedule."""
        bisect.insort(self._reservations[edge], float(x))

    def get_reservations(self, edge: Any) -> list[float]:
        """Return a copy of the reserved discharge times on an edge."""
        return list(self._reservations.get(edge, []))

    def clear(self) -> None:
        """Clear all active reservations."""
        self._reservations.clear()

    def copy(self) -> ReservationTable:
        """Return an independent deep copy of the reservation table."""
        tbl = ReservationTable(
            G=self.G,
            default_headway_s=self.default_headway_s,
            eps=self.eps,
        )
        tbl._headways = dict(self._headways)
        tbl._reservations = defaultdict(list, {k: list(v) for k, v in self._reservations.items()})
        return tbl

