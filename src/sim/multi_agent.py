"""Multi-agent evacuation simulator with point-queue road capacities.

Modelling Assumptions
---------------------
1. Point-Queue Traffic Dynamics:
   Links have infinite spatial storage (no spatial spillback). Vehicles traverse
   an edge at free-flow/degraded speed tau(e, t), then queue vertically at the
   downstream end and discharge at flow_capacity(e) (one vehicle per headway h, FIFO).
2. En-Route Hazard Trapping:
   If an agent's scheduled link exit time exceeds the hazard arrival time on that edge
   (t_exit >= hazard.edge_arrival_time(e)), the agent is trapped with reason
   'edge_closed_en_route'. The traffic state is NOT committed for trapped vehicles.
3. Node and Exit Capacity:
   Junctions/nodes have unlimited capacity (no conflict delays). Boundary exits
   have unlimited absorption capacity.
4. Independent Links:
   Parallel edges maintain independent queues.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from src.routing.base import Evacuee, Router, RoutingParams
from src.sim.single_agent import AgentResult
from src.traffic.state import TrafficState

if TYPE_CHECKING:
    from src.hazard.base import HazardModel


@dataclass
class PopulationResult:
    """Outcome of a multi-agent population evacuation simulation.

    Parameters
    ----------
    agent_results : list[AgentResult]
        Outcomes for all individual simulated evacuees.
    edge_stats : pd.DataFrame
        Per-edge statistics (peak_occupancy, total_queue_delay_s, n_vehicles, storage_capacity).
    """

    agent_results: list[AgentResult]
    edge_stats: pd.DataFrame


def simulate_population(
    G: nx.MultiDiGraph,
    hazard: "HazardModel",
    router: Router,
    agents: Sequence[Evacuee],
    exits: set[int] | Sequence[int],
    params: RoutingParams,
    traffic_cfg: Any | None = None,
) -> PopulationResult:
    """Simulate a population of interacting evacuee agents under point-queue congestion.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network graph.
    hazard : HazardModel
        Hazard simulator providing closures and edge arrival times.
    router : Router
        Routing algorithm used by all agents.
    agents : Sequence[Evacuee]
        List of evacuee agents with origins and departure times.
    exits : set[int] or Sequence[int]
        Collection of safe exit node IDs.
    params : RoutingParams
        Speed scale and speed-factor floor parameters.
    traffic_cfg : Any, optional
        Traffic configuration parameters.

    Returns
    -------
    PopulationResult
        Aggregated outcomes per agent and per edge.
    """
    traffic_state = TrafficState(G)
    exits_set = set(exits)

    # Optional pre-run router initialization / offline pre-planning
    router.prepare(G, hazard, agents, exits_set, params, traffic_cfg)

    # Priority queue: (time, agent_id, node)
    # Tie-breaking by agent_id ensures strict determinism.
    pq: list[tuple[float, int, int]] = []
    for a in sorted(agents, key=lambda x: (x.depart_time_s, x.id)):
        heapq.heappush(pq, (float(a.depart_time_s), int(a.id), int(a.origin)))

    agent_map = {a.id: a for a in agents}
    agent_plan: dict[int, list[tuple[int, int, int]]] = {}
    agent_path_nodes: dict[int, list[int]] = {a.id: [int(a.origin)] for a in agents}
    agent_path_length: dict[int, float] = {a.id: 0.0 for a in agents}
    agent_queue_delay: dict[int, float] = {a.id: 0.0 for a in agents}
    agent_replans: dict[int, int] = {a.id: 0 for a in agents}
    agent_compute_s: dict[int, float] = {a.id: 0.0 for a in agents}
    last_plan_time: dict[int, float] = {a.id: float("-inf") for a in agents}

    agent_results: dict[int, AgentResult] = {}
    active_agents: set[int] = {a.id for a in agents}

    while pq and active_agents:
        t, agent_id, curr_node = heapq.heappop(pq)
        if agent_id not in active_agents:
            continue
        agent = agent_map[agent_id]

        # 1. Check if agent reached an exit
        if curr_node in exits_set:
            active_agents.remove(agent_id)
            finish_time = t
            evac_time = max(0.0, finish_time - agent.depart_time_s)
            agent_results[agent_id] = AgentResult(
                agent_id=agent_id,
                router_name=router.name,
                safe=True,
                trapped_reason=None,
                depart_time_s=agent.depart_time_s,
                finish_time_s=finish_time,
                evac_time_s=evac_time,
                path_nodes=agent_path_nodes[agent_id],
                path_length_m=agent_path_length[agent_id],
                n_replans=agent_replans[agent_id],
                compute_ms=agent_compute_s[agent_id] * 1000.0,
                trap_time_s=None,
                queue_delay_s=agent_queue_delay[agent_id],
            )
            continue

        # 2. Decide whether to plan or replan
        first_plan = (last_plan_time[agent_id] == float("-inf"))
        needs_replan = first_plan

        if not needs_replan:
            remaining = agent_plan.get(agent_id, [])
            if router.replans and any(hazard.is_edge_closed(u, v, k, t) for u, v, k in remaining):
                needs_replan = True
            replan_int = getattr(router, "replan_interval_s", None)
            if replan_int is not None and (t - last_plan_time[agent_id] >= replan_int - 1e-9):
                needs_replan = True

        if needs_replan:
            if not first_plan:
                agent_replans[agent_id] += 1
            t_start = time.perf_counter()
            route = router.plan(
                G=G,
                hazard=hazard,
                evacuee=agent,
                t_now_s=t,
                current_node=curr_node,
                exits=exits_set,
                params=params,
                traffic=traffic_state,
            )
            agent_compute_s[agent_id] += (time.perf_counter() - t_start)
            last_plan_time[agent_id] = t

            if route is None or not route.edges:
                active_agents.remove(agent_id)
                agent_results[agent_id] = AgentResult(
                    agent_id=agent_id,
                    router_name=router.name,
                    safe=False,
                    trapped_reason="no_route",
                    depart_time_s=agent.depart_time_s,
                    finish_time_s=None,
                    evac_time_s=None,
                    path_nodes=agent_path_nodes[agent_id],
                    path_length_m=agent_path_length[agent_id],
                    n_replans=agent_replans[agent_id],
                    compute_ms=agent_compute_s[agent_id] * 1000.0,
                    trap_time_s=t,
                    queue_delay_s=agent_queue_delay[agent_id],
                )
                continue

            agent_plan[agent_id] = list(route.edges)

        # 3. Step forward onto next planned edge
        if not agent_plan.get(agent_id):
            active_agents.remove(agent_id)
            agent_results[agent_id] = AgentResult(
                agent_id=agent_id,
                router_name=router.name,
                safe=False,
                trapped_reason="no_route",
                depart_time_s=agent.depart_time_s,
                finish_time_s=None,
                evac_time_s=None,
                path_nodes=agent_path_nodes[agent_id],
                path_length_m=agent_path_length[agent_id],
                n_replans=agent_replans[agent_id],
                compute_ms=agent_compute_s[agent_id] * 1000.0,
                trap_time_s=t,
                queue_delay_s=agent_queue_delay[agent_id],
            )
            continue

        u_next, v_next, k_next = agent_plan[agent_id].pop(0)
        edge = (u_next, v_next, k_next)
        edge_data = G.edges[u_next, v_next, k_next]
        fft = float(edge_data.get("free_flow_time", edge_data.get("travel_time", 1.0)))
        sf = max(hazard.edge_speed_factor(u_next, v_next, k_next, t), params.sf_floor)
        tau = fft / params.vehicle_speed_scale / sf

        h = traffic_state.headway[edge]
        prev_exit = traffic_state.last_exit[edge]
        candidate_exit = max(t + tau, prev_exit + h)

        # Usability check: must clear edge prior to flood arrival
        a_e = hazard.edge_arrival_time(u_next, v_next, k_next)
        if candidate_exit >= a_e:
            active_agents.remove(agent_id)
            agent_path_nodes[agent_id].append(v_next)
            agent_path_length[agent_id] += float(edge_data.get("length", 0.0))
            agent_results[agent_id] = AgentResult(
                agent_id=agent_id,
                router_name=router.name,
                safe=False,
                trapped_reason="edge_closed_en_route",
                depart_time_s=agent.depart_time_s,
                finish_time_s=None,
                evac_time_s=None,
                path_nodes=agent_path_nodes[agent_id],
                path_length_m=agent_path_length[agent_id],
                n_replans=agent_replans[agent_id],
                compute_ms=agent_compute_s[agent_id] * 1000.0,
                trap_time_s=a_e,
                queue_delay_s=agent_queue_delay[agent_id],
            )
            continue

        # Commit entry to traffic state
        actual_exit = traffic_state.enter(edge, t, tau)
        queue_wait = max(0.0, actual_exit - (t + tau))
        agent_queue_delay[agent_id] += queue_wait
        agent_path_nodes[agent_id].append(v_next)
        agent_path_length[agent_id] += float(edge_data.get("length", 0.0))

        heapq.heappush(pq, (actual_exit, agent_id, v_next))

    # Any remaining un-finished agents (e.g. queue drained)
    for aid in list(active_agents):
        agent = agent_map[aid]
        agent_results[aid] = AgentResult(
            agent_id=aid,
            router_name=router.name,
            safe=False,
            trapped_reason="no_route",
            depart_time_s=agent.depart_time_s,
            finish_time_s=None,
            evac_time_s=None,
            path_nodes=agent_path_nodes[aid],
            path_length_m=agent_path_length[aid],
            n_replans=agent_replans[aid],
            compute_ms=agent_compute_s[aid] * 1000.0,
            trap_time_s=float(np.nan),
            queue_delay_s=agent_queue_delay[aid],
        )

    ordered_results = [agent_results[a.id] for a in agents]

    # Build edge statistics DataFrame
    edge_rows = []
    for u, v, k, data in G.edges(keys=True, data=True):
        e = (u, v, k)
        cap = int(data.get("capacity", 2))
        peak = int(traffic_state.peak_occupancy[e])
        tot_delay = float(traffic_state.total_queue_delay[e])
        n_veh = int(traffic_state.n_vehicles[e])
        max_wait = float(traffic_state.max_queue_wait[e])
        edge_rows.append(
            {
                "u": u,
                "v": v,
                "k": k,
                "peak_occupancy": peak,
                "storage_capacity": cap,
                "congestion_ratio": float(peak / cap) if cap > 0 else 0.0,
                "total_queue_delay_s": tot_delay,
                "max_queue_wait_s": max_wait,
                "n_vehicles": n_veh,
                "flow_capacity": float(data.get("flow_capacity", 12.0)),
                "highway": str(data.get("highway", "unknown")),
                "lanes_n": int(data.get("lanes_n", 1)),
                "length": float(data.get("length", 0.0)),
            }
        )
    edge_stats_df = pd.DataFrame(edge_rows)

    return PopulationResult(agent_results=ordered_results, edge_stats=edge_stats_df)


def simulate_population_all(
    routers: Sequence[Router],
    G: nx.MultiDiGraph,
    hazard: "HazardModel",
    agents: Sequence[Evacuee],
    exits: set[int] | Sequence[int],
    params: RoutingParams,
    traffic_cfg: Any | None = None,
) -> dict[str, PopulationResult]:
    """Execute multi-agent simulation across multiple routers on the identical population.

    Each router operates on a fresh TrafficState instance.
    """
    results: dict[str, PopulationResult] = {}
    for r in routers:
        results[r.name] = simulate_population(
            G=G,
            hazard=hazard,
            router=r,
            agents=agents,
            exits=exits,
            params=params,
            traffic_cfg=traffic_cfg,
        )
    return results

