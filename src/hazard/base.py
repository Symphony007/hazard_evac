"""Abstract hazard model interface.

Every hazard implementation (flood, fire, …) must implement this
interface so that the routing layer can query edge status at any time
without knowing hazard internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class HazardModel(ABC):
    """Stable interface between the hazard engine and the routing core."""

    @abstractmethod
    def node_arrival_time(self, n: int) -> float:
        """Seconds until hazard reaches node *n*; ``np.inf`` if never."""

    @abstractmethod
    def edge_arrival_time(self, u: int, v: int, k: int) -> float:
        """Seconds until hazard reaches edge *(u, v, k)*; = min(T(u), T(v))."""

    @abstractmethod
    def is_edge_closed(self, u: int, v: int, k: int, t: float) -> bool:
        """True if edge *(u, v, k)* is impassable at time *t* (seconds)."""

    @abstractmethod
    def edge_speed_factor(self, u: int, v: int, k: int, t: float) -> float:
        """Speed multiplier in [0, 1] for edge *(u, v, k)* at time *t*.

        - 1.0 if *t* ≤ arrival − ramp (no degradation yet)
        - Linearly decreases to *min_speed_factor* at *t* = arrival
        - 0.0 if *t* ≥ arrival (edge closed)
        """

    @abstractmethod
    def is_node_flooded(self, n: int, t: float) -> bool:
        """True if node *n* is flooded at time *t* (seconds)."""

    @abstractmethod
    def closed_edges(self, t: float) -> set[tuple[int, int, int]]:
        """Set of *(u, v, k)* tuples that are closed at time *t*."""

    @abstractmethod
    def closure_schedule(self) -> list[tuple[float, tuple[int, int, int]]]:
        """All edge closures sorted by time.

        Returns a list of ``(time_s, (u, v, k))`` for every edge with a
        finite arrival time, sorted ascending by *time_s*.
        """

