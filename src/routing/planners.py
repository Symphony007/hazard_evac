"""Single-agent evacuation planners implementing the Router interface.

Implements three planning paradigms:
1. StaticRouter: Computes path once at departure using a frozen hazard snapshot.
2. ReactiveRouter: Computes initial path frozen at departure; replans with fresh
   frozen snapshots whenever an upcoming planned edge is found closed en-route.
3. TimeDependentRouter: Computes time-dependent earliest-arrival path using full
   hazard forecast (assumes perfect hazard knowledge).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from src.routing.base import Evacuee, Route, Router, RoutingParams, TrafficView
from src.routing.search import earliest_arrival_search

if TYPE_CHECKING:
    import networkx as nx
    from src.hazard.base import HazardModel


class StaticRouter(Router):
    """Static routing: plans once at departure using a frozen hazard snapshot.

    The route is computed with edge costs and closures evaluated at departure
    time t0. The agent follows this route blindly without replanning, regardless
    of subsequent hazard spread or closures.
    """

    name: str = "static"
    replans: bool = False

    def plan(
        self,
        G: "nx.MultiDiGraph",
        hazard: "HazardModel",
        evacuee: Evacuee,
        t_now_s: float,
        current_node: int,
        exits: set[int] | Sequence[int],
        params: RoutingParams,
        traffic: TrafficView | None = None,
    ) -> Route | None:
        """Compute fastest route on the frozen snapshot at *t_now_s*."""
        return earliest_arrival_search(
            G=G,
            hazard=hazard,
            source=current_node,
            t_start=t_now_s,
            exits=exits,
            params=params,
            mode="frozen",
        )


class ReactiveRouter(Router):
    """Reactive routing: re-routes on observed road closure.

    Computes an initial route at departure using a frozen hazard snapshot.
    During execution, the simulator inspects whether any upcoming edge along the
    planned path is closed at the current time. If so, ReactiveRouter replans
    from the agent's current node with a fresh frozen snapshot at the current
    time. It has no lookahead/forecast capability.
    """

    name: str = "reactive"
    replans: bool = True

    def plan(
        self,
        G: "nx.MultiDiGraph",
        hazard: "HazardModel",
        evacuee: Evacuee,
        t_now_s: float,
        current_node: int,
        exits: set[int] | Sequence[int],
        params: RoutingParams,
        traffic: TrafficView | None = None,
    ) -> Route | None:
        """Compute fastest route on the frozen snapshot at *t_now_s*."""
        return earliest_arrival_search(
            G=G,
            hazard=hazard,
            source=current_node,
            t_start=t_now_s,
            exits=exits,
            params=params,
            mode="frozen",
        )


class TimeDependentRouter(Router):
    """Time-dependent routing: optimal path using hazard forecast.

    Computes the earliest-arrival route at departure time using true time-dependent
    edge travel times and the edge usability rule (t_entry + tt < edge_arrival_time).

    Assumptions
    -----------
    Assumes perfect forecast of hazard arrival times and speed factors.
    In the 8th semester phase, this assumption will be relaxed to an
    uncertainty-aware probabilistic hazard belief.
    """

    name: str = "time_dependent"
    replans: bool = False

    def plan(
        self,
        G: "nx.MultiDiGraph",
        hazard: "HazardModel",
        evacuee: Evacuee,
        t_now_s: float,
        current_node: int,
        exits: set[int] | Sequence[int],
        params: RoutingParams,
        traffic: TrafficView | None = None,
    ) -> Route | None:
        """Compute earliest-arrival route using time-dependent hazard forecast."""
        return earliest_arrival_search(
            G=G,
            hazard=hazard,
            source=current_node,
            t_start=t_now_s,
            exits=exits,
            params=params,
            mode="time_dependent",
        )


class SelfishRouter(Router):
    """Selfish routing: 'navigation app' baseline with live congestion delays.

    Semantics
    ---------
    Computes earliest-arrival route to any exit in FROZEN mode using the current
    snapshot at planning time:
    - Closed edges at t_now_s are excluded.
    - Hazard speed factors are frozen at t_now_s.
    - Live congestion delays from traffic.extra_delay_s are added to edge costs.
    - Has NO flood forecast and no coordination.
    - Replans before entering an edge if an upcoming planned edge is closed now,
      and periodically whenever (t - last_plan_time) >= replan_interval_s.
    """

    name: str = "selfish"
    replans: bool = True
    replan_interval_s: float | None = 60.0

    def __init__(self, replan_interval_s: float | None = 60.0) -> None:
        self.replan_interval_s = replan_interval_s

    def plan(
        self,
        G: "nx.MultiDiGraph",
        hazard: "HazardModel",
        evacuee: Evacuee,
        t_now_s: float,
        current_node: int,
        exits: set[int] | Sequence[int],
        params: RoutingParams,
        traffic: TrafficView | None = None,
    ) -> Route | None:
        """Compute fastest route using frozen hazard snapshot plus live traffic delay."""
        extra_delay_fn = (
            (lambda u, v, k, t: traffic.extra_delay_s(u, v, k, t))
            if traffic is not None
            else None
        )
        return earliest_arrival_search(
            G=G,
            hazard=hazard,
            source=current_node,
            t_start=t_now_s,
            exits=exits,
            params=params,
            mode="frozen",
            extra_delay=extra_delay_fn,
        )

