"""Flow decomposition for exact time-expanded min-cost flow solutions.

Decomposes the integral arc flow from an ExactModel (typically in 'ceil' mode)
into individual agent plans:
  agent_id -> AgentPlan | None

Each AgentPlan specifies:
- edges: list of (u, v, key) physical road edge traversals
- nodes: list of physical nodes visited [origin, ..., exit]
- planned_exit_s: discretized time instant when exit is reached
- planned_wait_steps: number of holdover (queueing/waiting) steps

Interchangeability & Ordering:
- Agents originating at the same node with identical departure step k_dep
  are interchangeable in the single-commodity flow formulation.
- Decomposed safe paths are assigned to agents in ascending agent ID order.
- Agents whose flow traverses an 'unsaved' arc receive None.

Validation & Guarantees:
- Total number of decomposed safe paths matches result.n_safe.
- Physical road edges connect consecutively: v_i == u_{i+1}.
- Each plan begins at the agent's origin and ends at a recognized safe exit.
- No network arc is consumed beyond its solved flow.
- All non-sink flow is fully consumed and accounted for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from src.exact.time_expanded import ExactModel, ExactResult
    from src.routing.base import Evacuee


@dataclass
class AgentPlan:
    """Planned evacuation route and timing decomposed from time-expanded min-cost flow.

    Parameters
    ----------
    edges : list[tuple[int | str, int | str, int]]
        Physical road edge sequence (u, v, key).
    nodes : list[int | str]
        Physical node sequence visited, starting from origin and ending at exit.
    planned_exit_s : float
        Discretized time instant (in seconds) when the agent reaches the exit in the model.
    planned_wait_steps : int
        Number of holdover (waiting) steps used along the space-time route.
    """

    edges: list[tuple[int | str, int | str, int]]
    nodes: list[int | str]
    planned_exit_s: float
    planned_wait_steps: int


def _trace_single_safe_path(
    start_nid: int,
    sink_id: int,
    out_arcs: dict[int, list[int]],
    model: "ExactModel",
    rem_flow: list[int],
    arc_ptr: dict[int, int],
    max_steps: int,
) -> list[int] | None:
    """Trace a single flow unit from start_nid to sink_id following positive-flow arcs.

    In the time-expanded network, time k is strictly monotonically increasing on all
    road and holdover arcs, so the graph is a Directed Acyclic Graph (DAG). A greedy
    forward traversal following positive flow arcs is guaranteed to reach an exit arc
    to the sink in O(horizon) steps without encountering directed cycles.
    """
    curr = start_nid
    path_arcs: list[int] = []
    steps = 0

    while curr != sink_id:
        steps += 1
        if steps > max_steps:
            raise RuntimeError(
                f"Path decomposition exceeded {max_steps} steps from node {start_nid}; infinite loop detected."
            )

        arcs = out_arcs.get(curr, [])
        ptr = arc_ptr.get(curr, 0)
        chosen_arc: int | None = None

        while ptr < len(arcs):
            a_idx = arcs[ptr]
            meta = model.arc_metadata[a_idx]
            # Skip unsaved arcs during safe path decomposition
            if meta.get("type") == "unsaved":
                ptr += 1
                continue
            if rem_flow[a_idx] > 0:
                chosen_arc = a_idx
                break
            ptr += 1

        arc_ptr[curr] = ptr

        if chosen_arc is None:
            # No reachable safe path from this node with remaining flow
            return None

        path_arcs.append(chosen_arc)
        curr = model.end_nodes[chosen_arc]

    return path_arcs


def decompose_flow(
    model: "ExactModel",
    result: "ExactResult",
    agents: Sequence["Evacuee"] | None = None,
) -> dict[int, AgentPlan | None]:
    """Decompose solved integral min-cost flow into per-agent evacuation plans.

    Parameters
    ----------
    model : ExactModel
        Built time-expanded network.
    result : ExactResult
        Solved min-cost flow result containing integral arc flows.
    agents : Sequence[Evacuee], optional
        Evacuating population. If None, retrieves `model.agents`.

    Returns
    -------
    dict[int, AgentPlan | None]
        Mapping of agent_id -> AgentPlan for safe agents, or None for unsaved agents.

    Raises
    ------
    ValueError
        If result is not OPTIMAL, flow conservation is violated, consecutive edges
        do not connect, or arcs are used beyond solved flow capacity.
    """
    if result.status != "OPTIMAL":
        raise ValueError(
            f"Cannot decompose non-optimal flow result: status={result.status}"
        )

    if agents is None:
        agents = getattr(model, "agents", None)
    if agents is None:
        raise ValueError("agents must be provided either directly or attached to model")

    rem_flow = list(result.arc_flows)
    if len(rem_flow) != model.n_arcs:
        raise ValueError(
            f"Result arc_flows length ({len(rem_flow)}) does not match model.n_arcs ({model.n_arcs})"
        )

    for i, f in enumerate(rem_flow):
        if f < 0:
            raise ValueError(f"Negative flow {f} detected on arc {i}")

    dt = model.dt_s
    K = model.K
    sink_id = model.sink_id
    max_steps = max(K + 10, 1000)

    # Build outgoing adjacency list
    out_arcs: dict[int, list[int]] = {}
    unsaved_arc_map: dict[tuple[int | str, int], int] = {}

    for i in range(model.n_arcs):
        u = model.start_nodes[i]
        out_arcs.setdefault(u, []).append(i)
        meta = model.arc_metadata[i]
        if meta.get("type") == "unsaved":
            unsaved_arc_map[(meta["origin"], meta["k_dep"])] = i

    # Group agents by (origin, k_dep)
    agent_groups: dict[tuple[int | str, int], list["Evacuee"]] = {}
    for a in agents:
        if model.rounding == "ceil":
            k_dep = min(K, math.ceil(a.depart_time_s / dt))
        else:
            k_dep = min(K, math.floor(a.depart_time_s / dt))
        agent_groups.setdefault((a.origin, k_dep), []).append(a)

    agent_plans: dict[int, AgentPlan | None] = {}
    arc_ptr: dict[int, int] = {}

    # Identify recognized exit nodes in the network
    network_exits: set[int | str] = set()
    for meta in model.arc_metadata:
        if meta.get("type") == "exit":
            network_exits.add(meta["exit"])

    # Process agent groups in chronological departure order so holdover flow
    # from earlier departures is decomposed before later departures at the same origin
    sorted_group_keys = sorted(agent_groups.keys(), key=lambda k: k[1])

    for st in sorted_group_keys:
        group = agent_groups[st]
        origin, k_dep = st
        sorted_group = sorted(group, key=lambda a: a.id)
        n_group = len(sorted_group)

        if st not in model.node_to_id:
            # Origin space-time node was not present in the network (infeasible from the start)
            for a in sorted_group:
                agent_plans[a.id] = None
            continue

        start_nid = model.node_to_id[st]
        safe_plans: list[AgentPlan] = []

        # Determine exact number of safe paths originating from this supply group
        unsaved_arc = unsaved_arc_map.get(st)
        n_unsaved_group = rem_flow[unsaved_arc] if unsaved_arc is not None else 0
        n_safe_group = n_group - n_unsaved_group

        # Extract exactly n_safe_group safe paths from start_nid to sink_id
        for _ in range(n_safe_group):
            path_arcs = _trace_single_safe_path(
                start_nid=start_nid,
                sink_id=sink_id,
                out_arcs=out_arcs,
                model=model,
                rem_flow=rem_flow,
                arc_ptr=arc_ptr,
                max_steps=max_steps,
            )
            if not path_arcs:
                raise ValueError(
                    f"Expected {n_safe_group} safe paths from {st}, but could only trace {len(safe_plans)}"
                )

            # Decrement flow along path
            for a_idx in path_arcs:
                rem_flow[a_idx] -= 1

            # Build AgentPlan from path arcs
            edges: list[tuple[int | str, int | str, int]] = []
            nodes: list[int | str] = [origin]
            planned_wait_steps = 0
            planned_exit_s = float(k_dep * dt)

            for a_idx in path_arcs:
                meta = model.arc_metadata[a_idx]
                atype = meta.get("type")
                if atype == "road":
                    edge = meta["edge"]
                    edges.append(edge)
                    nodes.append(edge[1])
                elif atype == "holdover":
                    planned_wait_steps += 1
                elif atype == "exit":
                    planned_exit_s = float(meta["k"] * dt)

            safe_plans.append(
                AgentPlan(
                    edges=edges,
                    nodes=nodes,
                    planned_exit_s=planned_exit_s,
                    planned_wait_steps=planned_wait_steps,
                )
            )

        # Sort safe plans by (planned_exit_s, planned_wait_steps) for ascending assignment
        safe_plans.sort(key=lambda p: (p.planned_exit_s, p.planned_wait_steps))

        n_safe_group = len(safe_plans)
        if n_safe_group > n_group:
            raise ValueError(
                f"More safe paths extracted ({n_safe_group}) than agents in group ({n_group}) at {st}"
            )

        # Assign safe plans to agents with lower IDs
        for idx in range(n_safe_group):
            agent_plans[sorted_group[idx].id] = safe_plans[idx]

        # Remaining agents get None and consume the unsaved arc
        n_unsaved_group = n_group - n_safe_group
        for idx in range(n_safe_group, n_group):
            agent_plans[sorted_group[idx].id] = None

        if n_unsaved_group > 0:
            unsaved_arc = unsaved_arc_map.get(st)
            if unsaved_arc is None:
                raise ValueError(f"Missing unsaved arc for {st} with {n_unsaved_group} unsaved agents")
            if rem_flow[unsaved_arc] < n_unsaved_group:
                raise ValueError(
                    f"Unsaved arc {unsaved_arc} at {st} had flow {rem_flow[unsaved_arc]} < required {n_unsaved_group}"
                )
            rem_flow[unsaved_arc] -= n_unsaved_group

    # -----------------------------------------------------------------------
    # Comprehensive Validations
    # -----------------------------------------------------------------------
    n_decomposed_safe = sum(1 for p in agent_plans.values() if p is not None)
    if n_decomposed_safe != result.n_safe:
        raise ValueError(
            f"Decomposition safe count mismatch: decomposed {n_decomposed_safe} safe paths, "
            f"but ExactResult reported {result.n_safe} safe agents."
        )

    agent_dict = {a.id: a for a in agents}
    for agent_id, plan in agent_plans.items():
        if plan is None:
            continue
        agent = agent_dict[agent_id]

        # Must start at agent's origin
        if plan.nodes[0] != agent.origin:
            raise ValueError(
                f"Agent {agent_id} plan starts at {plan.nodes[0]}, expected origin {agent.origin}"
            )
        if plan.edges and plan.edges[0][0] != agent.origin:
            raise ValueError(
                f"Agent {agent_id} first edge starts at {plan.edges[0][0]}, expected origin {agent.origin}"
            )

        # Consecutive edges must connect (v_i == u_{i+1})
        for i in range(len(plan.edges) - 1):
            curr_v = plan.edges[i][1]
            next_u = plan.edges[i + 1][0]
            if curr_v != next_u:
                raise ValueError(
                    f"Agent {agent_id} edge disconnect at step {i}: {plan.edges[i]} -> {plan.edges[i+1]}"
                )

        # Plan nodes must match edge traversal sequence
        expected_nodes = [agent.origin] + [e[1] for e in plan.edges]
        if plan.nodes != expected_nodes:
            raise ValueError(
                f"Agent {agent_id} nodes sequence {plan.nodes} does not match edges {expected_nodes}"
            )

        # Must end at an exit
        last_node = plan.nodes[-1]
        if network_exits and (last_node not in network_exits):
            raise ValueError(
                f"Agent {agent_id} plan ends at node {last_node}, which is not an exit"
            )

    # Flow conservation check: all non-sink flow must be 0
    for i, meta in enumerate(model.arc_metadata):
        atype = meta.get("type")
        if atype in ("road", "holdover"):
            if rem_flow[i] != 0:
                raise ValueError(
                    f"Unconsumed flow {rem_flow[i]} remains on non-sink arc {i} ({meta})"
                )
        elif atype in ("exit", "unsaved"):
            if rem_flow[i] != 0:
                raise ValueError(
                    f"Unconsumed flow {rem_flow[i]} remains on {atype} arc {i} ({meta})"
                )

    return agent_plans
