"""SUMO route sanitizer for open-loop evacuation replay (Task 8b fix).

Replanning routers (e.g. selfish, selfish_forecast) can occasionally produce
U-turns or disconnected edge transitions at junctions where SUMO's network has
no physical turnaround connection or turn lane.

This module sanitizes routes before the SUMO routes XML file is written:
1. Reversal removal: removes pairs (e_i, e_{i+1}) where e_{i+1} is the reverse
   direction of the same road and (e_i, e_{i+1}) is not in valid_connections.
   Repeats until no invalid reversals remain (handles nested reversals).
2. Other disconnected pairs: attempts to find a bridge path using sumolib's
   net.getShortestPath(e_i_obj, e_{i+1}_obj). If found, intermediate edges
   are inserted.
3. If reversals empty the route, or a disconnection cannot be bridged, the
   agent is marked ROUTE_DROPPED (excluded from route file, counted as UNSAFE).
4. Sanitizer statistics are tracked and logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class SanitizerStats:
    """Statistics collected during route sanitization."""

    n_uturn_pairs_removed: int = 0
    n_repaired: int = 0
    n_route_dropped: int = 0
    n_agents_affected: int = 0


class SumoNetCache:
    """Per-process cache of parsed SUMO network and connection topology."""

    _cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def get(cls, net_path: str | Path) -> dict[str, Any]:
        """Load and cache SUMO network data for net_path."""
        import sumolib

        resolved = str(Path(net_path).resolve())
        if resolved not in cls._cache:
            net = sumolib.net.readNet(resolved)
            valid_connections: set[tuple[str, str]] = {
                (e.getID(), out.getID())
                for e in net.getEdges()
                for out in e.getOutgoing()
            }
            edge_nodes: dict[str, tuple[str, str]] = {
                e.getID(): (e.getFromNode().getID(), e.getToNode().getID())
                for e in net.getEdges()
            }
            cls._cache[resolved] = {
                "net": net,
                "valid_connections": valid_connections,
                "edge_nodes": edge_nodes,
            }

        if "H_rev" not in cls._cache[resolved]:
            import networkx as nx

            net = cls._cache[resolved]["net"]
            H_rev = nx.DiGraph()
            for edge in net.getEdges():
                eid = edge.getID()
                fft = edge.getLength() / max(edge.getSpeed(), 0.1)
                H_rev.add_node(eid, fft=fft)

            for edge in net.getEdges():
                eid = edge.getID()
                for target in edge.getOutgoing():
                    target_id = target.getID()
                    target_fft = target.getLength() / max(target.getSpeed(), 0.1)
                    H_rev.add_edge(target_id, eid, weight=target_fft)

            cls._cache[resolved]["H_rev"] = H_rev

        return cls._cache[resolved]

    @classmethod
    def clear(cls) -> None:
        """Clear cache (primarily for testing)."""
        cls._cache.clear()


def is_reversal(e1_id: str, e2_id: str, edge_nodes: dict[str, tuple[str, str]]) -> bool:
    """Return True if e1 and e2 represent opposite directions of the same road segment."""
    u1, v1 = edge_nodes.get(e1_id, (None, None))
    u2, v2 = edge_nodes.get(e2_id, (None, None))
    return (u1 is not None) and (u2 is not None) and (u1 == v2) and (v1 == u2)


def sanitize_route(
    sumo_edges: list[str],
    net_data: dict[str, Any],
) -> tuple[list[str], bool, int, int]:
    """Sanitize a single vehicle's SUMO edge sequence.

    Parameters
    ----------
    sumo_edges : list[str]
        List of SUMO edge identifiers.
    net_data : dict
        Dict containing "net", "valid_connections", and "edge_nodes".

    Returns
    -------
    sanitized_edges : list[str]
        Cleaned edge list. Empty if dropped.
    dropped : bool
        True if the route could not be made valid.
    uturn_pairs_removed : int
        Number of illegal U-turn pairs removed.
    repaired : int
        Number of disconnected transitions bridged.
    """
    valid_connections: set[tuple[str, str]] = net_data["valid_connections"]
    edge_nodes: dict[str, tuple[str, str]] = net_data["edge_nodes"]
    net: Any = net_data["net"]

    # Deduplicate consecutive identical edges
    deduped: list[str] = []
    for eid in sumo_edges:
        if not deduped or deduped[-1] != eid:
            deduped.append(eid)

    if not deduped:
        return [], True, 0, 0

    uturn_pairs_removed = 0

    # Step 1: Reversal removal (loop until no invalid reversal pairs remain)
    changed = True
    while changed:
        changed = False
        if len(deduped) < 2:
            break
        i = 0
        while i < len(deduped) - 1:
            e1 = deduped[i]
            e2 = deduped[i + 1]
            if (e1, e2) not in valid_connections and is_reversal(e1, e2, edge_nodes):
                del deduped[i : i + 2]
                uturn_pairs_removed += 1
                changed = True
                # Collapse any adjacent duplicate edges resulting from deletion
                j = max(0, i - 1)
                while j < len(deduped) - 1:
                    if deduped[j] == deduped[j + 1]:
                        del deduped[j + 1]
                    else:
                        j += 1
                break  # Restart scan from beginning
            i += 1

    if not deduped:
        return [], True, uturn_pairs_removed, 0

    # Step 2: Other disconnected pairs
    repaired = 0
    route_broken = False
    i = 0
    while i < len(deduped) - 1:
        e1 = deduped[i]
        e2 = deduped[i + 1]
        if (e1, e2) not in valid_connections:
            # Check if sumolib can find a shortest-path bridge
            edge_obj1 = net.getEdge(e1) if hasattr(net, "getEdge") else None
            edge_obj2 = net.getEdge(e2) if hasattr(net, "getEdge") else None
            if edge_obj1 is None or edge_obj2 is None:
                route_broken = True
                break

            path, _ = net.getShortestPath(edge_obj1, edge_obj2)
            if path is None or len(path) < 2:
                route_broken = True
                break

            bridge = [e.getID() for e in path[1:-1]]
            if not bridge and (e1, e2) not in valid_connections:
                # getShortestPath returned 2 edges but connection still not in valid_connections
                route_broken = True
                break

            # Insert intermediate bridge edges between e1 and e2
            deduped[i + 1 : i + 1] = bridge
            repaired += 1
            # Advance index past the bridge to e2
            i += len(bridge) + 1
        else:
            i += 1

    if route_broken or not deduped:
        return [], True, uturn_pairs_removed, repaired

    return deduped, False, uturn_pairs_removed, repaired


def sanitize_agent_routes(
    net_path: str | Path,
    agents_with_routes: list[tuple[Any, list[Any]]],
    edge_map_g2s: dict[str, str] | None = None,
) -> tuple[list[tuple[Any, list[str]]], set[int], SanitizerStats]:
    """Sanitize all agent routes before SUMO replay.

    Parameters
    ----------
    net_path : str or Path
        Path to compiled SUMO net.net.xml.
    agents_with_routes : list of (agent, route_edges)
        Each route_edges can be a list of graph edge tuples (u, v, k) or SUMO edge IDs.
    edge_map_g2s : dict, optional
        Mapping "u,v,k" -> SUMO edge id. Required if route_edges contains graph tuples.

    Returns
    -------
    sanitized_pairs : list of (agent, list_of_sumo_edge_ids)
        Agents whose routes were successfully sanitized.
    dropped_agent_ids : set of int
        Agent IDs whose routes were dropped (invalid/unrepairable).
    stats : SanitizerStats
        Counts of removed U-turns, bridges, dropped routes, and affected agents.
    """
    net_data = SumoNetCache.get(net_path)
    stats = SanitizerStats()
    sanitized_pairs: list[tuple[Any, list[str]]] = []
    dropped_agent_ids: set[int] = set()

    for agent, route_edges in agents_with_routes:
        aid = int(agent.id)
        if not route_edges:
            dropped_agent_ids.add(aid)
            stats.n_route_dropped += 1
            stats.n_agents_affected += 1
            continue

        # Convert to SUMO edge IDs if needed
        if isinstance(route_edges[0], str):
            sumo_edges = list(route_edges)
        else:
            if edge_map_g2s is None:
                raise ValueError("edge_map_g2s must be provided when route_edges contains graph tuples")
            sumo_edges = []
            for item in route_edges:
                if isinstance(item, str):
                    sumo_edges.append(item)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    u, v = item[0], item[1]
                    k = item[2] if len(item) >= 3 else 0
                    seid = edge_map_g2s.get(f"{u},{v},{k}")
                    if seid is not None:
                        sumo_edges.append(seid)

        sanitized, dropped, u_removed, n_rep = sanitize_route(sumo_edges, net_data)
        stats.n_uturn_pairs_removed += u_removed
        stats.n_repaired += n_rep

        affected = (u_removed > 0) or (n_rep > 0) or dropped

        if dropped or not sanitized:
            dropped_agent_ids.add(aid)
            stats.n_route_dropped += 1
            stats.n_agents_affected += 1
        else:
            if affected:
                stats.n_agents_affected += 1
            sanitized_pairs.append((agent, sanitized))

    logger.info(
        f"[sanitizer] uturn_pairs_removed={stats.n_uturn_pairs_removed}, "
        f"repaired={stats.n_repaired}, "
        f"route_dropped={stats.n_route_dropped}, "
        f"agents_affected={stats.n_agents_affected}"
    )

    return sanitized_pairs, dropped_agent_ids, stats


def compute_sumo_fallback_routes(
    net_path: str | Path,
    agents: Sequence[Any],
    exits: Sequence[int | str] | set[int | str],
    edge_map_g2s: dict[str, str] | None = None,
    G: Any = None,
) -> dict[int, list[str]]:
    """Compute shortest free-flow-time path on SUMO net from agent origins to nearest exit.

    Parameters
    ----------
    net_path : str or Path
        Path to compiled SUMO net.net.xml.
    agents : Sequence[Evacuee]
        Evacuee agents that need fallback routes.
    exits : Sequence or set of int or str
        Exit node IDs in the road network.
    edge_map_g2s : dict, optional
        Mapping 'u,v,k' -> SUMO edge id.
    G : nx.MultiDiGraph, optional
        Graph to look up outgoing edges if origin node not found in SUMO net.

    Returns
    -------
    dict[int, list[str]]
        Mapping agent_id -> list of SUMO edge IDs for the fallback route.
    """
    if not agents:
        return {}

    import networkx as nx

    net_data = SumoNetCache.get(net_path)
    net = net_data["net"]
    H_rev = net_data["H_rev"].copy()

    exit_node_strs = {str(x) for x in exits}
    exit_edges = [
        e.getID() for e in net.getEdges()
        if e.getToNode().getID() in exit_node_strs
    ]

    if not exit_edges:
        logger.warning("No exit edges found in SUMO network for given exits: %s", exits)
        return {}

    for ee in exit_edges:
        H_rev.add_edge("__EXIT__", ee, weight=H_rev.nodes[ee]["fft"])

    distances, paths = nx.single_source_dijkstra(H_rev, "__EXIT__", weight="weight")

    fallback_routes: dict[int, list[str]] = {}
    for a in agents:
        orig_id_str = str(a.origin)
        cand_edges: list[str] = []
        if hasattr(net, "_id2node") and orig_id_str in net._id2node:
            orig_node = net.getNode(orig_id_str)
            cand_edges.extend(e.getID() for e in orig_node.getOutgoing())
        elif hasattr(net, "getNode"):
            try:
                orig_node = net.getNode(orig_id_str)
                if orig_node is not None:
                    cand_edges.extend(e.getID() for e in orig_node.getOutgoing())
            except Exception:
                pass

        if not cand_edges and G is not None and hasattr(G, "out_edges") and a.origin in G:
            for _, v, k in G.out_edges(a.origin, keys=True):
                if edge_map_g2s:
                    seid = edge_map_g2s.get(f"{a.origin},{v},{k}")
                    if seid:
                        cand_edges.append(seid)

        valid_cands = [e for e in cand_edges if e in distances]
        if valid_cands:
            best_edge = min(valid_cands, key=lambda e: distances[e])
            # Reverse path and exclude '__EXIT__'
            fwd_path = paths[best_edge][::-1][:-1]
            if fwd_path:
                fallback_routes[int(a.id)] = fwd_path

    return fallback_routes
