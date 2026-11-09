"""Single-agent evacuation simulator.

Executes a single evacuee's journey through the road network under dynamic
hazard conditions, tracking route progress, replanning events, and safety
outcomes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

from src.routing.base import Evacuee, Router, RoutingParams

if TYPE_CHECKING:
    import networkx as nx
    from src.hazard.base import HazardModel


@dataclass
class AgentResult:
    """Execution outcome for a single evacuating agent.

    Parameters
    ----------
    agent_id : int
        Agent identifier.
    router_name : str
        Name of the routing algorithm used.
    safe : bool
        True if the agent successfully reached an exit node.
    trapped_reason : {None, "no_route", "edge_closed_en_route"}
        Reason for failure if safe is False.
    depart_time_s : float
        Departure time from origin in seconds.
    finish_time_s : float | None
        Simulation time at which exit was reached (None if trapped).
    evac_time_s : float | None
        Total evacuation duration finish_time - depart_time (None if trapped).
    path_nodes : list[int]
        Sequence of nodes traversed during movement.
    path_length_m : float
        Physical road distance traversed in metres.
    n_replans : int
        Number of replanning events triggered en-route.
    compute_ms : float
        Total CPU time spent in router.plan() for this agent, in milliseconds.
    """

    agent_id: int
    router_name: str
    safe: bool
    trapped_reason: Literal["no_route", "edge_closed_en_route"] | None
    depart_time_s: float
    finish_time_s: float | None
    evac_time_s: float | None
    path_nodes: list[int | str]
    path_length_m: float
    n_replans: int
    compute_ms: float
    trap_time_s: float | None = None


def simulate_agent(
    G: "nx.MultiDiGraph",
    hazard: "HazardModel",
    router: Router,
    evacuee: Evacuee,
    exits: set[int] | Sequence[int],
    params: RoutingParams,
    max_steps: int = 10000,
) -> AgentResult:
    """Simulate an individual evacuee moving through the network.

    Procedure
    ---------
    1. Initialize t = evacuee.depart_time_s, current node = origin.
    2. Request initial plan from router. If None -> trapped with "no_route".
    3. Traverse planned edges sequentially:
       a. If router.replans and any remaining planned edge is closed at current
          time t, replan from current node. If replan returns None -> trapped
          with "no_route".
       b. Compute true traversal time tt(e, t) using actual hazard state.
       c. Check usability rule: if t + tt >= edge_arrival_time -> trapped with
          "edge_closed_en_route".
       d. Otherwise advance t += tt, update current node, record distance.
    4. Upon reaching any exit node -> mark SAFE, record finish_time_s.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network graph.
    hazard : HazardModel
        Active hazard simulator.
    router : Router
        Routing algorithm instance (Static, Reactive, or TimeDependent).
    evacuee : Evacuee
        Agent initial state (origin and departure time).
    exits : set[int] or Sequence[int]
        Target exit nodes.
    params : RoutingParams
        Speed scaling and floor parameters.
    max_steps : int
        Safety bound on edge traversal iterations to prevent infinite loops.

    Returns
    -------
    AgentResult
        Detailed outcome and metrics for this agent.
    """
    exits_set = set(exits)
    t = evacuee.depart_time_s
    node = evacuee.origin
    path_nodes = [node]
    path_length_m = 0.0
    n_replans = 0
    total_compute_s = 0.0

    # Origin is already an exit
    if node in exits_set:
        return AgentResult(
            agent_id=evacuee.id,
            router_name=router.name,
            safe=True,
            trapped_reason=None,
            depart_time_s=t,
            finish_time_s=t,
            evac_time_s=0.0,
            path_nodes=path_nodes,
            path_length_m=0.0,
            n_replans=0,
            compute_ms=0.0,
        )

    # Initial plan
    t_start_plan = time.perf_counter()
    initial_route = router.plan(G, hazard, evacuee, t, node, exits_set, params)
    total_compute_s += time.perf_counter() - t_start_plan

    if initial_route is None or len(initial_route.edges) == 0:
        return AgentResult(
            agent_id=evacuee.id,
            router_name=router.name,
            safe=False,
            trapped_reason="no_route",
            depart_time_s=t,
            finish_time_s=None,
            evac_time_s=None,
            path_nodes=path_nodes,
            path_length_m=0.0,
            n_replans=0,
            compute_ms=total_compute_s * 1000.0,
            trap_time_s=t,
        )

    remaining_edges = list(initial_route.edges)
    steps = 0

    while node not in exits_set and steps < max_steps:
        steps += 1

        # Reactive replanning trigger: inspect downstream path for closed edges
        if router.replans:
            if any(hazard.is_edge_closed(u, v, k, t) for u, v, k in remaining_edges):
                t_replan = time.perf_counter()
                new_route = router.plan(G, hazard, evacuee, t, node, exits_set, params)
                total_compute_s += time.perf_counter() - t_replan
                n_replans += 1

                if new_route is None or len(new_route.edges) == 0:
                    return AgentResult(
                        agent_id=evacuee.id,
                        router_name=router.name,
                        safe=False,
                        trapped_reason="no_route",
                        depart_time_s=evacuee.depart_time_s,
                        finish_time_s=None,
                        evac_time_s=None,
                        path_nodes=path_nodes,
                        path_length_m=path_length_m,
                        n_replans=n_replans,
                        compute_ms=total_compute_s * 1000.0,
                        trap_time_s=t,
                    )
                remaining_edges = list(new_route.edges)

        if not remaining_edges:
            if node in exits_set:
                break
            return AgentResult(
                agent_id=evacuee.id,
                router_name=router.name,
                safe=False,
                trapped_reason="no_route",
                depart_time_s=evacuee.depart_time_s,
                finish_time_s=None,
                evac_time_s=None,
                path_nodes=path_nodes,
                path_length_m=path_length_m,
                n_replans=n_replans,
                compute_ms=total_compute_s * 1000.0,
                trap_time_s=t,
            )

        u, v, k = remaining_edges.pop(0)

        # Retrieve edge attributes
        edge_data = G.edges[u, v, k]
        length_m = float(edge_data.get("length", 0.0))
        fft = float(edge_data.get("free_flow_time", edge_data.get("travel_time", 0.0)))

        # True traversal time on edge entered at time t
        sf = max(hazard.edge_speed_factor(u, v, k, t), params.sf_floor)
        tt = fft / params.vehicle_speed_scale / sf
        a_e = hazard.edge_arrival_time(u, v, k)

        # Usability check: must clear the edge before closure
        if (t + tt >= a_e) or hazard.is_edge_closed(u, v, k, t):
            # Trapped either before entry or en-route when edge closes at a_e
            trap_instant = min(t + tt, a_e) if np.isfinite(a_e) else t
            return AgentResult(
                agent_id=evacuee.id,
                router_name=router.name,
                safe=False,
                trapped_reason="edge_closed_en_route",
                depart_time_s=evacuee.depart_time_s,
                finish_time_s=None,
                evac_time_s=None,
                path_nodes=path_nodes,
                path_length_m=path_length_m,
                n_replans=n_replans,
                compute_ms=total_compute_s * 1000.0,
                trap_time_s=trap_instant,
            )

        # Advance along edge
        t += tt
        node = v
        path_nodes.append(node)
        path_length_m += length_m

    if node in exits_set:
        return AgentResult(
            agent_id=evacuee.id,
            router_name=router.name,
            safe=True,
            trapped_reason=None,
            depart_time_s=evacuee.depart_time_s,
            finish_time_s=t,
            evac_time_s=t - evacuee.depart_time_s,
            path_nodes=path_nodes,
            path_length_m=path_length_m,
            n_replans=n_replans,
            compute_ms=total_compute_s * 1000.0,
        )

    # Reached step limit without reaching an exit
    return AgentResult(
        agent_id=evacuee.id,
        router_name=router.name,
        safe=False,
        trapped_reason="no_route",
        depart_time_s=evacuee.depart_time_s,
        finish_time_s=None,
        evac_time_s=None,
        path_nodes=path_nodes,
        path_length_m=path_length_m,
        n_replans=n_replans,
        compute_ms=total_compute_s * 1000.0,
    )
