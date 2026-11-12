"""Exact time-expanded min-cost-flow benchmark for dynamic evacuation routing.

Model Formulation
-----------------
Time is discretized in uniform steps of length dt seconds (default 5s).
The continuous horizon horizon_s is partitioned into steps k = 0, 1, ..., K
where K = horizon_s / dt.

Rounding Modes (Dual Bounds):
-----------------------------
1. "ceil" (default, conservative lower bound):
   - Every time is rounded against the agents (conservative):
     * Departure step: k_dep = ceil(depart_time / dt).
     * Travel steps: m = max(1, ceil(tau(e, k*dt) / dt)).
     * Usability: (k + m) * dt < a(e).
   - This provides a conservative lower bound on the true continuous-time optimum
     (up to entry-capacity discretization).
2. "floor" (optimistic upper bound):
   - Every time is rounded in favor of the agents (optimistic):
     * Departure step: k_dep = floor(depart_time / dt).
     * Travel steps: m = floor(tau(e, k*dt) / dt) (may be 0 for short links inside one time layer).
     * Usability: k*dt + tau(e, k*dt) < a(e) (exact arrival time entered at start of step).
   - This provides an approximate upper bound on the continuous-time optimum.
The true optimum of the continuous-time model should lie between [ceil, floor].

Time-expanded network:
1. Space-time nodes (v, k):
   Represents physical node v at time instant t = k * dt, for k = 0..K.
   A node (v, k) is pruned if k < E(v) (earliest arrival time from any supply node)
   or if k * dt >= hazard.node_arrival_time(v) (node is flooded), or if no exit can
   be reached from (v, k) prior to horizon or flood closures.

2. Holdover (waiting) arcs:
   (v, k) -> (v, k + 1)
   Capacity: unlimited (modeled as INT_MAX).
   Cost: 1 step.
   Allowed only while (k + 1) * dt < hazard.node_arrival_time(v).
   Models queueing / holding at a node with unlimited storage, matching the
   simulator's point-queue model.

3. Edge traversal arcs:
   For every physical directed road edge e = (u, v, key) and step k:
   Let t = k * dt.
   Traversal duration:
     tau(e, t) = free_flow_time / vehicle_speed_scale / max(speed_factor(e, t), sf_floor)
     m = max(1, ceil(tau(e, t) / dt)) if rounding == "ceil" else floor(tau(e, t) / dt)
   Target node is (v, k + m).
   Arc capacity:
     c = max(1, floor(flow_capacity * dt / 60))
     (Exact for graph with dt = 5: flow_capacity = 12 veh/min -> c = 1 veh/step).
   Cost: m steps.
   Usability rule:
     If rounding == "ceil": (k + m) * dt < a(e)
     If rounding == "floor": k * dt + tau(e, t) < a(e)

4. Supply & Demand:
   Every agent contributes 1 unit of supply at (origin, k_dep), where
   k_dep = ceil(depart_time / dt) in ceil mode or floor(depart_time / dt) in floor mode.
   A super-sink node T has total demand = number of evacuees.

5. Exit arcs:
   For every exit x and every valid time step k:
   (x, k) -> T with unlimited capacity and cost 0.
   Agents reaching an exit can immediately sink at zero additional cost.
   If an agent already originates at an exit at departure time, it can sink directly.

6. Unsaved penalty arcs:
   From each supply node (origin, k_dep) directly to T:
   Capacity: unlimited (or supply count), cost: M.
   Big-M is set to big_m_factor * (K + 1) * n_agents.
   Because any safe path costs at most (K + 1) steps, the min-cost objective
   strictly prioritizes maximizing the number of safe agents first (avoiding
   penalty M), and then minimizing total travel duration among safe agents.

Modelling Assumptions & Discretization Details:
- Single commodity: evacuees are interchangeable units of flow.
- Holdover arcs model queueing at road junctions with unlimited storage capacity.
- dt discretization rounds edge traversal times up or down depending on mode.
- Capacities are integer per step c = floor(flow_capacity * dt / 60).
- Perfect hazard forecast is assumed across the full evacuation horizon.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import networkx as nx
import numpy as np

from src.config import ExactConfig
from src.routing.base import Evacuee, RoutingParams

if TYPE_CHECKING:
    from src.hazard.base import HazardModel


INT_CAPACITY_INF = 100_000_000  # Large integer capacity representing unlimited flow


@dataclass
class ExactModel:
    """Built time-expanded network ready for min-cost-flow solving."""

    # Node mapping
    node_to_id: dict[Any, int]
    id_to_node: list[Any]
    sink_id: int
    supplies: dict[int, int]  # node_id -> supply (positive) / demand (negative)

    # Arc attributes
    start_nodes: list[int]
    end_nodes: list[int]
    capacities: list[int]
    unit_costs: list[int]
    arc_metadata: list[dict[str, Any]]

    # Statistics
    n_nodes_raw: int
    n_arcs_raw: int
    n_nodes: int
    n_arcs: int
    build_time_s: float
    big_m: int
    dt_s: int
    horizon_s: int
    K: int
    n_agents: int
    rounding: str = "ceil"


@dataclass
class ExactResult:
    """Solution outcomes from exact time-expanded min-cost flow."""

    n_agents: int
    n_safe: int
    pct_safe_model: float
    total_travel_steps: int
    solve_time_s: float
    build_time_s: float
    n_nodes: int
    n_arcs: int
    status: str
    optimal_cost: int
    arc_flows: list[int]
    rounding: str = "ceil"


def _forward_earliest_arrival(
    G: nx.MultiDiGraph,
    hazard: "HazardModel",
    supply_nodes: dict[int | str, int],  # origin -> min k_dep
    params: RoutingParams,
    dt_s: int,
    K: int,
    rounding: str = "ceil",
) -> dict[int | str, int]:
    """Compute earliest step E(v) that any agent can reach node v (ignoring capacity).

    Uses a forward time-dependent Dijkstra starting from all (origin, min_k_dep).
    """
    earliest_step: dict[int | str, int] = {}
    pq: list[tuple[int, int | str]] = []

    for orig, k_dep in supply_nodes.items():
        if orig not in earliest_step or k_dep < earliest_step[orig]:
            earliest_step[orig] = k_dep
            heapq.heappush(pq, (k_dep, orig))

    while pq:
        curr_k, u = heapq.heappop(pq)
        if curr_k > earliest_step[u] or curr_k > K:
            continue

        t_now = curr_k * dt_s
        if t_now >= hazard.node_arrival_time(u):
            continue

        for _, v, k_edge, data in G.out_edges(u, keys=True, data=True):
            fft = float(data.get("free_flow_time", data.get("travel_time", 1.0)))
            sf = max(hazard.edge_speed_factor(u, v, k_edge, t_now), params.sf_floor)
            tau = fft / params.vehicle_speed_scale / sf
            m = max(1, math.ceil(tau / dt_s)) if rounding == "ceil" else math.floor(tau / dt_s)
            arr_step = curr_k + m

            if arr_step > K:
                continue

            # Usability check: must clear edge prior to flood arrival
            a_e = hazard.edge_arrival_time(u, v, k_edge)
            if rounding == "ceil":
                if (arr_step * dt_s) >= a_e:
                    continue
            else:
                if (curr_k * dt_s + tau) >= a_e:
                    continue

            # Target node flood check
            if (arr_step * dt_s) >= hazard.node_arrival_time(v):
                continue

            if v not in earliest_step or arr_step < earliest_step[v]:
                earliest_step[v] = arr_step
                heapq.heappush(pq, (arr_step, v))

    return earliest_step


def build_network(
    G: nx.MultiDiGraph,
    hazard: "HazardModel",
    agents: Sequence[Evacuee],
    exits: set[int] | Sequence[int],
    params: RoutingParams,
    exact_cfg: ExactConfig,
    prune: bool = True,
) -> ExactModel:
    """Build the time-expanded network for exact evacuation optimization.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network graph with length, free_flow_time, and flow_capacity.
    hazard : HazardModel
        Flood hazard simulator.
    agents : Sequence[Evacuee]
        Evacuating population.
    exits : set[int] | Sequence[int]
        Target safe exit nodes.
    params : RoutingParams
        Vehicle speed scale and hazard speed floor.
    exact_cfg : ExactConfig
        Discretization parameters and size constraints.
    prune : bool
        Whether to perform forward earliest arrival and reverse exit reachability pruning.

    Returns
    -------
    ExactModel
        Network structure prepared for SimpleMinCostFlow solver.
    """
    t_start = time.perf_counter()
    dt = exact_cfg.dt_s
    horizon = exact_cfg.horizon_s
    K = horizon // dt
    exits_set = set(exits)
    n_agents = len(agents)
    rounding = getattr(exact_cfg, "rounding", "ceil")

    # 1. Map supply requirements: (origin, k_dep) -> count
    supplies_raw: dict[tuple[int | str, int], int] = {}
    origin_min_k: dict[int | str, int] = {}
    for a in agents:
        if rounding == "ceil":
            k_dep = min(K, math.ceil(a.depart_time_s / dt))
        else:
            k_dep = min(K, math.floor(a.depart_time_s / dt))
        supplies_raw[(a.origin, k_dep)] = supplies_raw.get((a.origin, k_dep), 0) + 1
        if a.origin not in origin_min_k or k_dep < origin_min_k[a.origin]:
            origin_min_k[a.origin] = k_dep

    # 2. Forward Reachability / Earliest Arrival Pruning
    if prune:
        E_v = _forward_earliest_arrival(G, hazard, origin_min_k, params, dt, K, rounding=rounding)
    else:
        E_v = {n: 0 for n in G.nodes}

    # Pre-calculate node flood step limits
    node_flood_step: dict[int | str, int] = {}
    for n in G.nodes:
        t_flood = hazard.node_arrival_time(n)
        node_flood_step[n] = min(K + 1, math.ceil(t_flood / dt)) if math.isfinite(t_flood) else K + 1

    # Raw size estimation before pruning
    n_nodes_raw = len(G.nodes) * (K + 1) + 1
    # Holdover arcs: ~ |V| * K; edge arcs: ~ |E| * (K+1)
    n_arcs_raw = len(G.nodes) * K + len(G.edges) * (K + 1) + len(supplies_raw) + len(exits_set) * (K + 1)

    # 3. Candidate nodes and forward candidate arcs
    candidate_nodes: set[tuple[int | str, int]] = set()
    for (orig, k_dep) in supplies_raw:
        if k_dep <= K and k_dep < node_flood_step.get(orig, 0):
            candidate_nodes.add((orig, k_dep))

    candidate_arcs: list[tuple[tuple[int | str, int], tuple[int | str, int], int, int, dict[str, Any]]] = []
    # Items: (u_node, v_node, capacity, unit_cost, metadata)

    for u in G.nodes:
        e_u = E_v.get(u, K + 1) if prune else 0
        max_k_u = node_flood_step.get(u, K + 1)

        for k in range(e_u, min(K + 1, max_k_u)):
            curr_node = (u, k)
            candidate_nodes.add(curr_node)
            t_now = k * dt

            # Holdover arc: (u, k) -> (u, k + 1)
            if k + 1 <= K and (k + 1) < max_k_u:
                candidate_arcs.append((
                    curr_node,
                    (u, k + 1),
                    INT_CAPACITY_INF,
                    1,
                    {"type": "holdover", "u": u, "k": k},
                ))

            # Edge arcs: (u, k) -> (v, k + m)
            for _, v, k_edge, data in G.out_edges(u, keys=True, data=True):
                fft = float(data.get("free_flow_time", data.get("travel_time", 1.0)))
                sf = max(hazard.edge_speed_factor(u, v, k_edge, t_now), params.sf_floor)
                tau = fft / params.vehicle_speed_scale / sf
                m = max(1, math.ceil(tau / dt)) if rounding == "ceil" else math.floor(tau / dt)
                arr_step = k + m

                if arr_step > K:
                    continue

                # Usability rule: vehicle must clear edge prior to flood arrival
                a_e = hazard.edge_arrival_time(u, v, k_edge)
                if rounding == "ceil":
                    if (arr_step * dt) >= a_e:
                        continue
                else:
                    if (k * dt + tau) >= a_e:
                        continue

                # Target node flood check
                if arr_step >= node_flood_step.get(v, K + 1):
                    continue

                flow_cap = float(data.get("flow_capacity", 12.0))
                cap = max(1, math.floor(flow_cap * dt / 60.0))

                candidate_arcs.append((
                    curr_node,
                    (v, arr_step),
                    int(cap),
                    int(m),
                    {"type": "road", "edge": (u, v, k_edge), "k": k, "m": m},
                ))

    # 4. Reverse Reachability Pruning (from exits backwards)
    if prune:
        # Build reverse adjacency for reachability check
        rev_adj: dict[tuple[int | str, int], list[tuple[int | str, int]]] = {}
        for u_st, v_st, _, _, _ in candidate_arcs:
            if v_st not in rev_adj:
                rev_adj[v_st] = []
            rev_adj[v_st].append(u_st)

        # Seeds: all (exit, k) in candidate_nodes
        can_reach_exit: set[tuple[int | str, int]] = set()
        queue: list[tuple[int | str, int]] = []

        for x in exits_set:
            e_x = E_v.get(x, K + 1)
            max_k_x = node_flood_step.get(x, K + 1)
            for k in range(e_x, min(K + 1, max_k_x)):
                st = (x, k)
                if st in candidate_nodes:
                    can_reach_exit.add(st)
                    queue.append(st)

        while queue:
            curr = queue.pop(0)
            for pred in rev_adj.get(curr, []):
                if pred not in can_reach_exit:
                    can_reach_exit.add(pred)
                    queue.append(pred)

        # Retain only nodes that can reach an exit, PLUS supply nodes that cannot reach an exit
        # (so their unsaved arcs to sink can still be placed)
        final_nodes: set[tuple[int | str, int]] = set()
        for st in candidate_nodes:
            if st in can_reach_exit or st in supplies_raw:
                final_nodes.add(st)

        final_arcs = [
            arc for arc in candidate_arcs
            if arc[0] in can_reach_exit and arc[1] in can_reach_exit
        ]
    else:
        final_nodes = candidate_nodes
        final_arcs = candidate_arcs

    # 5. Build indexed model arrays for OR-Tools SimpleMinCostFlow
    sink_node = ("SINK", -1)
    all_nodes_list = sorted(list(final_nodes)) + [sink_node]
    node_to_id = {node: i for i, node in enumerate(all_nodes_list)}
    sink_id = node_to_id[sink_node]

    start_nodes: list[int] = []
    end_nodes: list[int] = []
    capacities: list[int] = []
    unit_costs: list[int] = []
    arc_metadata: list[dict[str, Any]] = []

    # Road and holdover arcs
    for u_st, v_st, cap, cost, meta in final_arcs:
        start_nodes.append(node_to_id[u_st])
        end_nodes.append(node_to_id[v_st])
        capacities.append(cap)
        unit_costs.append(cost)
        arc_metadata.append(meta)

    # Exit arcs: (x, k) -> SINK (cost = 0, cap = INT_CAPACITY_INF)
    for u_st in final_nodes:
        x, k = u_st
        if x in exits_set:
            start_nodes.append(node_to_id[u_st])
            end_nodes.append(sink_id)
            capacities.append(INT_CAPACITY_INF)
            unit_costs.append(0)
            arc_metadata.append({"type": "exit", "exit": x, "k": k})

    # Big-M for unsaved penalty
    # Any safe trip costs <= K steps.
    # M strictly dominates any possible travel time savings.
    big_m = int(exact_cfg.big_m_factor * (K + 1) * n_agents)

    # Unsaved arcs: (origin, k_dep) -> SINK with cost M
    for (orig, k_dep), supply_cnt in supplies_raw.items():
        st = (orig, k_dep)
        if st in node_to_id:
            u_id = node_to_id[st]
            start_nodes.append(u_id)
            end_nodes.append(sink_id)
            capacities.append(supply_cnt)
            unit_costs.append(big_m)
            arc_metadata.append({"type": "unsaved", "origin": orig, "k_dep": k_dep})

    # Node supplies: +supply at source nodes, -n_agents at sink
    supplies: dict[int, int] = {}
    total_supply = 0
    for (orig, k_dep), cnt in supplies_raw.items():
        if (orig, k_dep) in node_to_id:
            nid = node_to_id[(orig, k_dep)]
            supplies[nid] = supplies.get(nid, 0) + cnt
            total_supply += cnt

    supplies[sink_id] = -total_supply

    t_build = time.perf_counter() - t_start
    n_nodes = len(all_nodes_list)
    n_arcs = len(start_nodes)

    return ExactModel(
        node_to_id=node_to_id,
        id_to_node=all_nodes_list,
        sink_id=sink_id,
        supplies=supplies,
        start_nodes=start_nodes,
        end_nodes=end_nodes,
        capacities=capacities,
        unit_costs=unit_costs,
        arc_metadata=arc_metadata,
        n_nodes_raw=n_nodes_raw,
        n_arcs_raw=n_arcs_raw,
        n_nodes=n_nodes,
        n_arcs=n_arcs,
        build_time_s=t_build,
        big_m=big_m,
        dt_s=dt,
        horizon_s=horizon,
        K=K,
        n_agents=n_agents,
        rounding=rounding,
    )


def solve(model: ExactModel, exact_cfg: ExactConfig | None = None) -> ExactResult:
    """Solve the time-expanded min-cost flow problem using OR-Tools SimpleMinCostFlow.

    Parameters
    ----------
    model : ExactModel
        Built time-expanded model.
    exact_cfg : ExactConfig, optional
        Configuration containing max_arcs threshold.

    Returns
    -------
    ExactResult
        Solved flow statistics, safe agent count, travel time, and raw arc flows.
    """
    if exact_cfg is not None and model.n_arcs > exact_cfg.max_arcs:
        raise ValueError(
            f"Model size exceeds maximum allowed arcs: {model.n_arcs} > {exact_cfg.max_arcs}"
        )

    from ortools.graph.python import min_cost_flow

    smcf = min_cost_flow.SimpleMinCostFlow()

    # Add arcs
    for i in range(model.n_arcs):
        smcf.add_arc_with_capacity_and_unit_cost(
            model.start_nodes[i],
            model.end_nodes[i],
            int(model.capacities[i]),
            int(model.unit_costs[i]),
        )

    # Set supplies
    for nid, sup in model.supplies.items():
        smcf.set_node_supply(nid, int(sup))

    t0 = time.perf_counter()
    status = smcf.solve()
    solve_time_s = time.perf_counter() - t0

    status_str = "UNKNOWN"
    if status == smcf.OPTIMAL:
        status_str = "OPTIMAL"
    elif status == smcf.FEASIBLE:
        status_str = "FEASIBLE"
    elif status == smcf.INFEASIBLE:
        status_str = "INFEASIBLE"
    elif status == smcf.UNBALANCED:
        status_str = "UNBALANCED"
    elif status == smcf.BAD_RESULT:
        status_str = "BAD_RESULT"
    elif status == smcf.BAD_COST_RANGE:
        status_str = "BAD_COST_RANGE"

    if status != smcf.OPTIMAL:
        return ExactResult(
            n_agents=model.n_agents,
            n_safe=0,
            pct_safe_model=0.0,
            total_travel_steps=0,
            solve_time_s=solve_time_s,
            build_time_s=model.build_time_s,
            n_nodes=model.n_nodes,
            n_arcs=model.n_arcs,
            status=status_str,
            optimal_cost=0,
            arc_flows=[],
            rounding=model.rounding,
        )

    optimal_cost = smcf.optimal_cost()
    arc_flows = [int(smcf.flow(i)) for i in range(model.n_arcs)]

    # Count unsaved agents from unsaved arcs to sink
    n_unsaved = 0
    for i, meta in enumerate(model.arc_metadata):
        if meta.get("type") == "unsaved":
            n_unsaved += arc_flows[i]

    n_safe = model.n_agents - n_unsaved
    pct_safe = (100.0 * n_safe / model.n_agents) if model.n_agents > 0 else 0.0

    # Total travel steps = optimal_cost - (n_unsaved * big_m)
    total_travel_steps = int(optimal_cost - (n_unsaved * model.big_m))

    return ExactResult(
        n_agents=model.n_agents,
        n_safe=n_safe,
        pct_safe_model=pct_safe,
        total_travel_steps=total_travel_steps,
        solve_time_s=solve_time_s,
        build_time_s=model.build_time_s,
        n_nodes=model.n_nodes,
        n_arcs=model.n_arcs,
        status=status_str,
        optimal_cost=optimal_cost,
        arc_flows=arc_flows,
        rounding=model.rounding,
    )

