"""Core shortest-path search algorithms for evacuation routing.

Supports multi-target Dijkstra stopping at the earliest-reached exit, with
either a frozen hazard snapshot ("frozen") or dynamic hazard forecasting
("time_dependent").

FIFO Property
-------------
Traversing an edge e entered at time t takes:
    tt(e, t) = free_flow_time / vehicle_speed_scale / max(speed_factor(t), sf_floor)
Because flood degradation only reduces speeds over time (speed_factor(t) is
non-increasing in t), traversal time tt(e, t) is non-decreasing in entry time t.
Consequently, arrival time t + tt(e, t) is non-decreasing in departure time t:
an agent departing earlier always arrives at the end of edge e no later than an
agent departing later. Therefore, First-In-First-Out (FIFO) holds, guaranteeing
that Dijkstra's algorithm with time-dependent edge costs computes optimal
earliest-arrival paths without requiring label correction or waiting.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Literal, Sequence

from src.routing.base import Route, RoutingParams

if TYPE_CHECKING:
    import networkx as nx
    from src.hazard.base import HazardModel


def earliest_arrival_search(
    G: "nx.MultiDiGraph",
    hazard: "HazardModel",
    source: int,
    t_start: float,
    exits: set[int] | Sequence[int],
    params: RoutingParams,
    mode: Literal["frozen", "time_dependent"],
) -> Route | None:
    """Find the earliest-arrival route from *source* to any exit.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road graph with edge attributes 'length' and 'free_flow_time' (or 'travel_time').
    hazard : HazardModel
        Hazard simulator providing closure and speed factor queries.
    source : int
        Starting node ID.
    t_start : float
        Departure time in seconds from simulation start.
    exits : set[int] or Sequence[int]
        Target exit node IDs.
    params : RoutingParams
        Speed scale and speed-factor floor parameters.
    mode : {"frozen", "time_dependent"}
        - "frozen": Edge costs and closures evaluated statically at *t_start*.
          Edges closed at *t_start* are removed. Edge cost = tt(e, t_start).
        - "time_dependent": Edge cost tt(e, t_u) evaluated at arrival time t_u at
          tail u. Edge is unusable if t_u + tt >= hazard.edge_arrival_time(u, v, k).

    Returns
    -------
    Route or None
        Optimal route to the earliest reached exit, or None if no exit is reachable.
    """
    exits_set = set(exits)
    if not exits_set:
        return None

    # Trivial case: source is already an exit
    if source in exits_set:
        return Route(
            edges=[],
            nodes=[source],
            length_m=0.0,
            planned_arrival_s=t_start,
        )

    # In time_dependent mode, if source itself is flooded at departure, cannot leave
    if mode == "time_dependent":
        if hazard.is_node_flooded(source, t_start):
            return None

    speed_scale = params.vehicle_speed_scale
    sf_floor = params.sf_floor
    is_frozen = (mode == "frozen")

    # Priority queue: (arrival_time, node)
    pq: list[tuple[float, int]] = [(t_start, source)]
    # Best known arrival time at each node
    dist: dict[int, float] = {source: t_start}
    # Predecessor map: v -> (u, edge_key, edge_length)
    preds: dict[int, tuple[int, int, float]] = {}

    target_exit: int | None = None

    while pq:
        t_u, u = heapq.heappop(pq)

        # Skip stale heap entries
        if t_u > dist[u]:
            continue

        # Multi-target search stops when the first exit is settled
        if u in exits_set:
            target_exit = u
            break

        # Explore outgoing edges
        out_edges = G.out_edges(u, keys=True, data=True)
        for _, v, k, data in out_edges:
            length = float(data.get("length", 0.0))
            fft = float(data.get("free_flow_time", data.get("travel_time", 0.0)))

            if is_frozen:
                # Frozen snapshot evaluated at t_start
                if hazard.is_edge_closed(u, v, k, t_start):
                    continue
                sf = max(hazard.edge_speed_factor(u, v, k, t_start), sf_floor)
                tt = fft / speed_scale / sf
                t_arr = t_u + tt
            else:
                # Time-dependent: evaluated at arrival time at u (t_u)
                if hazard.is_edge_closed(u, v, k, t_u):
                    continue
                sf = max(hazard.edge_speed_factor(u, v, k, t_u), sf_floor)
                tt = fft / speed_scale / sf
                t_arr = t_u + tt
                # Usability rule: agent must clear the edge before flood arrival
                a_e = hazard.edge_arrival_time(u, v, k)
                if t_arr >= a_e:
                    continue

            if t_arr < dist.get(v, float("inf")):
                dist[v] = t_arr
                preds[v] = (u, k, length)
                heapq.heappush(pq, (t_arr, v))

    if target_exit is None:
        return None

    # Reconstruct path from target_exit back to source
    path_nodes: list[int] = []
    path_edges: list[tuple[int, int, int]] = []
    total_length = 0.0

    curr = target_exit
    while curr != source:
        path_nodes.append(curr)
        u, k, length = preds[curr]
        path_edges.append((u, curr, k))
        total_length += length
        curr = u
    path_nodes.append(source)

    path_nodes.reverse()
    path_edges.reverse()

    return Route(
        edges=path_edges,
        nodes=path_nodes,
        length_m=total_length,
        planned_arrival_s=dist[target_exit],
    )

