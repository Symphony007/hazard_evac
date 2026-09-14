"""
multi_agent_router.py -- Route many evacuees with congestion feedback.

This module is the central addition in Layer 2.  It takes a list of
(origin, destination) pairs representing multiple evacuees and routes
them **sequentially**, updating the congestion model after each agent
so that later agents naturally avoid roads already saturated by earlier
ones.

Two strategies are provided:

1. ``route_all_agents``       -- congestion-aware (BPR feedback loop)
2. ``route_all_agents_naive`` -- baseline without congestion (all agents
                                  get the same shortest path)

Comparing these two on the same input demonstrates the value of
congestion-aware routing: the aware version produces more diverse
routes and lower maximum edge loads.

Design notes (for viva)
-----------------------
* Sequential routing with incremental congestion updates is a form of
  **greedy marginal-cost routing**.  Each agent is routed on a graph
  whose edge costs reflect all previously routed agents.  This is
  simpler than a full System Optimal (SO) or User Equilibrium (UE)
  assignment but captures the essential congestion-diversification
  effect and is easy to explain in a viva.
* Hazard avoidance and congestion are combined by first penalising
  hazardous edges (length -> inf), then overlaying BPR congestion
  costs on top.  This guarantees hazardous edges are never used
  regardless of congestion.

References
----------
- Wardrop's principles (1952) for traffic assignment theory.
- BPR function: Bureau of Public Roads (1964).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .congestion import CongestionModel
from .hazard import HazardState
from .router import NoRouteError, compute_route

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RoutingResult:
    """
    Aggregated result of routing multiple agents.

    Attributes
    ----------
    agent_routes : dict[int, list[int]]
        Mapping from agent index to their route (node list).
    failed_agents : list[int]
        Indices of agents for whom no route could be found.
    avg_path_length_m : float
        Mean physical path length across all successfully routed agents.
    max_edge_load : int
        Highest load on any single edge after all agents are routed.
    num_hazard_reroutes : int
        Number of agents whose route was affected by hazard avoidance.
    num_unique_routes : int
        Number of distinct routes (by node sequence) used.
    congestion_stats : dict
        Summary stats from the congestion model after routing.
    """
    agent_routes: Dict[int, List[int]] = field(default_factory=dict)
    failed_agents: List[int] = field(default_factory=list)
    avg_path_length_m: float = 0.0
    max_edge_load: int = 0
    num_hazard_reroutes: int = 0
    num_unique_routes: int = 0
    failure_reasons: Dict[int, str] = field(default_factory=dict)
    congestion_stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper: compute physical path length
# ---------------------------------------------------------------------------

def _path_length_m(graph: nx.MultiDiGraph, route: List[int]) -> float:
    """Sum the ``length`` attributes along a route to get total metres."""
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        # Use the shortest edge if parallel edges exist
        edge_data = graph[u][v]
        min_len = min(
            float(d.get("length", 0)) for d in edge_data.values()
        )
        total += min_len
    return total


# ---------------------------------------------------------------------------
# Congestion-aware multi-agent routing
# ---------------------------------------------------------------------------

def route_all_agents(
    graph: nx.MultiDiGraph,
    od_pairs: List[Tuple[int, int]],
    hazard_state: Optional[HazardState] = None,
    congestion_model: Optional[CongestionModel] = None,
) -> RoutingResult:
    """
    Route all agents sequentially with congestion + hazard awareness.

    Each agent is routed on a graph whose edge costs reflect:
      1. BPR congestion from all previously routed agents.
      2. Infinite cost on hazardous edges (if a hazard_state is given).

    After each agent is routed, their route is registered with the
    congestion model so the next agent sees updated costs.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The original road-network graph (never mutated).
    od_pairs : list of (origin, destination)
        One tuple per agent, each containing OSM node IDs.
    hazard_state : HazardState or None
        If provided, hazardous edges are penalised to infinity.
    congestion_model : CongestionModel or None
        If None, a fresh one is created for this routing session.

    Returns
    -------
    RoutingResult
        Aggregated routing results and statistics.
    """
    if congestion_model is None:
        congestion_model = CongestionModel(graph)

    result = RoutingResult()
    num_reroutes = 0

    for idx, (origin, dest) in enumerate(od_pairs):
        # 1. Start with the original graph
        G_work = graph.copy()

        # 2. Penalise hazardous edges (if any)
        if hazard_state is not None:
            for u, v, key in hazard_state.hazardous_edges:
                if G_work.has_edge(u, v, key):
                    G_work[u][v][key]["length"] = float("inf")
            # Track if this agent's route would differ from no-hazard
            # (simplified: if origin or dest is near hazard, count it)

        # 3. Apply congestion costs onto the working graph
        G_work = congestion_model.apply_to_graph(G_work)

        # 4. Route this agent using congestion-adjusted costs
        try:
            route = compute_route(G_work, origin, dest, weight="congestion_cost")
        except NoRouteError:
            reason = f"No path exists between origin node {origin} and dest node {dest} (network disconnected or blocked by hazard)"
            logger.warning("[FAILED] Agent %d: %s", idx, reason)
            result.failed_agents.append(idx)
            result.failure_reasons[idx] = reason
            continue

        # 5. Check if hazard forced a different route than pure shortest
        if hazard_state is not None:
            try:
                pure_route = compute_route(graph, origin, dest, weight="length")
                if route != pure_route:
                    num_reroutes += 1
            except NoRouteError:
                num_reroutes += 1

        # 6. Register route with congestion model
        congestion_model.add_route(route)
        result.agent_routes[idx] = route

        logger.debug(
            "Agent %d routed: %d nodes, length=%.0f m",
            idx, len(route), _path_length_m(graph, route),
        )

    # --- Aggregate statistics ---
    if result.agent_routes:
        lengths = [
            _path_length_m(graph, r)
            for r in result.agent_routes.values()
        ]
        result.avg_path_length_m = sum(lengths) / len(lengths)

        # Count unique routes by their node tuple
        unique = set(tuple(r) for r in result.agent_routes.values())
        result.num_unique_routes = len(unique)

    stats = congestion_model.get_stats()
    result.max_edge_load = stats.max_load
    result.num_hazard_reroutes = num_reroutes
    result.congestion_stats = {
        "loaded_edges": stats.loaded_edges,
        "max_load": stats.max_load,
        "avg_load_on_used": round(stats.avg_load_on_used, 2),
        "total_volume": stats.total_volume,
    }

    return result


# ---------------------------------------------------------------------------
# Naive baseline (no congestion awareness)
# ---------------------------------------------------------------------------

def route_all_agents_naive(
    graph: nx.MultiDiGraph,
    od_pairs: List[Tuple[int, int]],
    hazard_state: Optional[HazardState] = None,
) -> RoutingResult:
    """
    Route all agents using plain shortest path (no congestion feedback).

    Every agent gets the same Dijkstra shortest path for their OD pair,
    regardless of how many other agents share the same edges.  This
    serves as the baseline to compare against congestion-aware routing.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The original road-network graph.
    od_pairs : list of (origin, destination)
        One tuple per agent.
    hazard_state : HazardState or None
        If provided, hazardous edges are penalised.

    Returns
    -------
    RoutingResult
        Routing results with congestion stats computed *after* all
        routes are assigned (for comparison purposes).
    """
    # Build a hazard-penalised graph (shared by all agents)
    if hazard_state is not None:
        G_work = graph.copy()
        for u, v, key in hazard_state.hazardous_edges:
            if G_work.has_edge(u, v, key):
                G_work[u][v][key]["length"] = float("inf")
    else:
        G_work = graph

    result = RoutingResult()
    # Track congestion post-hoc for comparison stats
    congestion_model = CongestionModel(graph)

    for idx, (origin, dest) in enumerate(od_pairs):
        try:
            route = compute_route(G_work, origin, dest, weight="length")
        except NoRouteError:
            reason = f"No path exists between origin node {origin} and dest node {dest} (network disconnected or blocked by hazard)"
            logger.warning("[FAILED] Agent %d (naive): %s", idx, reason)
            result.failed_agents.append(idx)
            result.failure_reasons[idx] = reason
            continue

        congestion_model.add_route(route)
        result.agent_routes[idx] = route

    # --- Aggregate statistics ---
    if result.agent_routes:
        lengths = [
            _path_length_m(graph, r)
            for r in result.agent_routes.values()
        ]
        result.avg_path_length_m = sum(lengths) / len(lengths)

        unique = set(tuple(r) for r in result.agent_routes.values())
        result.num_unique_routes = len(unique)

    stats = congestion_model.get_stats()
    result.max_edge_load = stats.max_load
    result.congestion_stats = {
        "loaded_edges": stats.loaded_edges,
        "max_load": stats.max_load,
        "avg_load_on_used": round(stats.avg_load_on_used, 2),
        "total_volume": stats.total_volume,
    }

    return result
