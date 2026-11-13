"""Coordinated (CCRP-style) evacuation router using capacity reservation tables.

Implements capacity-aware evacuation planning with both online (departure-time)
and offline (all agents planned upfront with perfect demand knowledge) modes.

Key Modeling Assumptions:
-------------------------
1. Centralized planner: Coordinates agent routes using a shared reservation schedule.
2. Perfect flood forecast: The planner queries the ground-truth flood arrival times
   a(e) and speed degradation curves.
3. Exit-time reservation: Capacity slots reserve only the EXIT time of each edge,
   matching the point-queue bottleneck discharge model.
4. No displacement tracking: A committed reservation is assumed fixed and not
   displaced by later planners.
5. Online mode: Agents are planned as they depart; no advance knowledge of future
   agent departures.
6. Offline mode: Perfect advance knowledge of all agent origins and departure times,
   prioritizing evacuation routes for agents at nodes with earliest flood arrival.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence

from src.routing.base import Evacuee, Route, Router, RoutingParams
from src.routing.search import earliest_arrival_search
from src.traffic.reservation import ReservationTable

if TYPE_CHECKING:
    import networkx as nx
    from src.hazard.base import HazardModel
    from src.routing.base import TrafficView


class CoordinatedRouter(Router):
    """Coordinated capacity-aware evacuation router.

    In "online" mode, plans routes dynamically at agent departure time and
    reserves edge discharge slots in a shared ReservationTable.

    In "offline" mode, pre-plans all routes in prepare() ordered by ascending
    flood arrival time at origin (most endangered first), providing an upper-bound
    benchmark.
    """

    def __init__(
        self,
        planning_mode: Literal["online", "offline"] = "online",
        name: str | None = None,
        flood_safety_margin_s: float = 0.0,
        replans: bool = False,
        replan_interval_s: float | None = None,
        fallback_on_infeasible: bool = True,
    ) -> None:
        """Initialize the coordinated router.

        Parameters
        ----------
        planning_mode : {"online", "offline"}
            Whether to plan at departure time or pre-plan all agents in prepare().
        name : str, optional
            Router name (defaults to "coordinated_{planning_mode}").
        flood_safety_margin_s : float
            Buffer subtracted from edge flood arrival time (default: 0.0).
        replans : bool
            Whether periodic or dynamic replanning is enabled (default: False).
        replan_interval_s : float or None
            Periodic replan interval in seconds (default: None).
        fallback_on_infeasible : bool
            Whether to fall back to forecast-only search if reservation search fails (default: True).
        """
        self.planning_mode: Literal["online", "offline"] = planning_mode
        self.name = name if name is not None else f"coordinated_{planning_mode}"
        self.flood_safety_margin_s = float(flood_safety_margin_s)
        self.replans = bool(replans)
        self.replan_interval_s = float(replan_interval_s) if replan_interval_s is not None else None
        self.fallback_on_infeasible = bool(fallback_on_infeasible)

        self.table: ReservationTable | None = None
        self._routes: dict[int, Route | None] = {}

    def prepare(
        self,
        G: "nx.MultiDiGraph",
        hazard: "HazardModel",
        agents: Sequence[Evacuee],
        exits: set[int] | Sequence[int],
        params: RoutingParams,
        traffic_cfg: Any | None = None,
    ) -> None:
        """Initialize a fresh reservation table and pre-plan in offline mode."""
        # Always create a fresh table per run; never leak state across runs
        self.table = ReservationTable(G)
        self._routes.clear()

        if self.planning_mode == "offline":
            exits_set = set(exits)
            # Sort agents: ascending hazard arrival time at origin (most endangered first),
            # breaking ties by departure time then agent ID
            def sort_key(a: Evacuee) -> tuple[float, float, int]:
                t_arr = float(hazard.node_arrival_time(a.origin))
                return (t_arr, float(a.depart_time_s), int(a.id))

            sorted_agents = sorted(agents, key=sort_key)
            for a in sorted_agents:
                route = self._plan_and_commit(
                    G=G,
                    hazard=hazard,
                    evacuee=a,
                    t_now_s=a.depart_time_s,
                    current_node=int(a.origin),
                    exits=exits_set,
                    params=params,
                )
                self._routes[a.id] = route

    def _plan_and_commit(
        self,
        G: "nx.MultiDiGraph",
        hazard: "HazardModel",
        evacuee: Evacuee,
        t_now_s: float,
        current_node: int,
        exits: set[int] | Sequence[int],
        params: RoutingParams,
    ) -> Route | None:
        """Find earliest-arrival route in reservation mode and commit slots.

        Implements two-stage planning:
        Stage 1: reservation-aware earliest-arrival search.
        Stage 2 (fallback): if stage 1 returns None and fallback_on_infeasible is True,
        plan with forecast-only time-dependent search, commit chosen slots, and set
        route.fallback_used = True.
        """
        if self.table is None:
            self.table = ReservationTable(G)

        # Stage 1: reservation-aware earliest-arrival search
        route = earliest_arrival_search(
            G=G,
            hazard=hazard,
            source=current_node,
            t_start=t_now_s,
            exits=exits,
            params=params,
            mode="reservation",
            table=self.table,
            flood_safety_margin_s=self.flood_safety_margin_s,
        )

        if route is not None:
            route.fallback_used = False
        elif self.fallback_on_infeasible:
            # Stage 2: fallback to forecast-only time-dependent search (no reservations)
            fallback_route = earliest_arrival_search(
                G=G,
                hazard=hazard,
                source=current_node,
                t_start=t_now_s,
                exits=exits,
                params=params,
                mode="time_dependent",
                extra_delay=None,
                flood_safety_margin_s=self.flood_safety_margin_s,
            )
            if fallback_route is not None:
                fallback_route.fallback_used = True
                route = fallback_route

        if route is None:
            return None

        if not route.edges:
            route.planned_arrival_s = t_now_s
            return route

        # Commit reservations along the chosen route
        t = t_now_s
        speed_scale = params.vehicle_speed_scale
        sf_floor = params.sf_floor

        for u, v, k in route.edges:
            data = G[u][v][k]
            fft = float(data.get("free_flow_time", data.get("travel_time", 0.0)))
            sf = max(hazard.edge_speed_factor(u, v, k, t), sf_floor)
            tau = fft / speed_scale / sf
            y = t + tau
            x = self.table.earliest_exit((u, v, k), y)
            self.table.reserve((u, v, k), x)
            t = x

        route.planned_arrival_s = t
        return route

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
        """Compute an exit route.

        In online mode, plans at t_now_s and commits to table.
        In offline mode, returns the pre-planned route.
        """
        if self.planning_mode == "offline":
            return self._routes.get(evacuee.id, None)

        return self._plan_and_commit(
            G=G,
            hazard=hazard,
            evacuee=evacuee,
            t_now_s=t_now_s,
            current_node=current_node,
            exits=exits,
            params=params,
        )

