"""
test_layer1.py — End-to-end tests for the Layer 1 baseline system.

Uses a small (~1.5 km radius) road-network extract around central Kolkata
to keep test execution fast while still exercising real OSM data.

Run with:
    .\\venv\\Scripts\\python.exe -m pytest tests/test_layer1.py -v

Test structure
--------------
1. test_graph_loading        — fetch / cache round-trip
2. test_basic_routing        — Dijkstra on real nodes
3. test_hazard_simulation    — hazard grows over time
4. test_reroute_avoids_hazard — rerouted path avoids danger zone
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `src` is importable as a package
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import load_graph, CACHE_DIR
from src.hazard import HazardSimulation
from src.router import compute_route, reroute_if_needed, NoRouteError


# ---------------------------------------------------------------------------
# Shared fixture — load graph once for the whole module
# ---------------------------------------------------------------------------

# Central Kolkata: 22.5726° N, 88.3639° E — 1500 m radius, "drive" network
_KOLKATA_POINT = (22.5726, 88.3639)
_KOLKATA_DIST = 1500


@pytest.fixture(scope="module")
def kolkata_graph():
    """Load (or cache-load) a small road-network graph around Kolkata."""
    G = load_graph(point=_KOLKATA_POINT, dist=_KOLKATA_DIST, network_type="drive")
    return G


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGraphLoading:
    """Verify graph fetching and caching behaviour."""

    def test_graph_has_nodes_and_edges(self, kolkata_graph):
        """A real OSM extract should have a non-trivial number of nodes/edges."""
        G = kolkata_graph
        assert len(G.nodes) > 50, f"Expected >50 nodes, got {len(G.nodes)}"
        assert len(G.edges) > 50, f"Expected >50 edges, got {len(G.edges)}"

    def test_cache_file_created(self):
        """After the first load, a .graphml file should exist in the cache."""
        slug = f"pt_{_KOLKATA_POINT[0]}_{_KOLKATA_POINT[1]}_d{int(_KOLKATA_DIST)}_drive"
        cached = CACHE_DIR / f"{slug}.graphml"
        assert cached.exists(), f"Cache file not found: {cached}"

    def test_reload_from_cache(self, kolkata_graph):
        """Loading the same graph again should use the cache (no network)."""
        G2 = load_graph(point=_KOLKATA_POINT, dist=_KOLKATA_DIST, network_type="drive")
        # Same number of nodes/edges confirms we got the same graph
        assert len(G2.nodes) == len(kolkata_graph.nodes)
        assert len(G2.edges) == len(kolkata_graph.edges)


class TestBasicRouting:
    """Verify shortest-path computation on real nodes."""

    def test_compute_route_returns_valid_path(self, kolkata_graph):
        """Pick two real nodes and check the route is a valid node sequence."""
        G = kolkata_graph
        nodes = list(G.nodes)
        origin, dest = nodes[0], nodes[-1]

        route = compute_route(G, origin, dest)

        # Route should start and end at the right nodes
        assert route[0] == origin
        assert route[-1] == dest
        # Every consecutive pair should be a real edge
        for u, v in zip(route[:-1], route[1:]):
            assert G.has_edge(u, v), f"Edge ({u}, {v}) not in graph"

    def test_route_has_multiple_nodes(self, kolkata_graph):
        """The route between distant nodes should have more than 2 hops."""
        G = kolkata_graph
        nodes = list(G.nodes)
        route = compute_route(G, nodes[0], nodes[-1])
        assert len(route) >= 2, "Route should have at least 2 nodes"


class TestHazardSimulation:
    """Verify hazard expansion over time steps."""

    def test_hazard_at_t0_contains_origin(self, kolkata_graph):
        """At t=0, only the origin node (at minimum) should be hazardous."""
        G = kolkata_graph
        origin = list(G.nodes)[len(G.nodes) // 2]

        sim = HazardSimulation(G, origin_node=origin, spread_rate_m=200)
        state = sim.get_hazard_state(0)

        assert origin in state.hazardous_nodes
        assert state.radius_m == 0.0

    def test_hazard_grows_over_time(self, kolkata_graph):
        """More nodes should become hazardous as time_step increases."""
        G = kolkata_graph
        origin = list(G.nodes)[len(G.nodes) // 2]

        sim = HazardSimulation(G, origin_node=origin, spread_rate_m=200)
        state_t1 = sim.get_hazard_state(1)
        state_t3 = sim.get_hazard_state(3)

        assert state_t3.radius_m > state_t1.radius_m
        assert len(state_t3.hazardous_nodes) >= len(state_t1.hazardous_nodes)

    def test_hazard_edges_are_consistent(self, kolkata_graph):
        """Every hazardous edge should have at least one hazardous endpoint."""
        G = kolkata_graph
        origin = list(G.nodes)[len(G.nodes) // 2]

        sim = HazardSimulation(G, origin_node=origin, spread_rate_m=300)
        state = sim.get_hazard_state(3)

        for u, v, _k in state.hazardous_edges:
            assert (
                u in state.hazardous_nodes or v in state.hazardous_nodes
            ), f"Edge ({u},{v}) is hazardous but neither endpoint is"


class TestRerouting:
    """Verify that rerouting avoids hazardous edges."""

    def test_reroute_produces_different_path(self, kolkata_graph):
        """
        When the hazard intersects the initial route, the rerouted
        path should differ and avoid hazardous edges.
        """
        G = kolkata_graph
        nodes = list(G.nodes)

        # Pick two nodes that are reasonably apart
        origin, dest = nodes[0], nodes[-1]
        initial_route = compute_route(G, origin, dest)

        # Place the hazard on a node in the *middle* of the route
        mid_idx = len(initial_route) // 2
        hazard_origin = initial_route[mid_idx]

        sim = HazardSimulation(G, origin_node=hazard_origin, spread_rate_m=150)

        # Advance time until the hazard actually intersects the route
        rerouted = False
        for t in range(1, 20):
            state = sim.get_hazard_state(t)
            new_route, did_reroute = reroute_if_needed(G, initial_route, state)
            if did_reroute:
                rerouted = True
                # The new route must not share the exact same sequence
                assert new_route != initial_route, (
                    "Rerouted path is identical to the original"
                )
                # Verify no edge on the new route is hazardous
                for u, v in zip(new_route[:-1], new_route[1:]):
                    assert (u, v, 0) not in state.hazardous_edges, (
                        f"Rerouted path still uses hazardous edge ({u},{v})"
                    )
                break

        assert rerouted, (
            "Hazard never intersected the route — test may need "
            "a higher spread_rate or more time steps"
        )

    def test_no_reroute_when_safe(self, kolkata_graph):
        """When the hazard is far from the route, no rerouting should occur."""
        G = kolkata_graph
        nodes = list(G.nodes)

        origin, dest = nodes[0], nodes[-1]
        route = compute_route(G, origin, dest)

        # Place hazard on a node NOT on the route, with tiny radius
        non_route_nodes = [n for n in nodes if n not in set(route)]
        if not non_route_nodes:
            pytest.skip("All nodes are on the route (very small graph)")

        sim = HazardSimulation(G, origin_node=non_route_nodes[0], spread_rate_m=1)
        state = sim.get_hazard_state(1)  # 1 m radius — basically nothing

        same_route, did_reroute = reroute_if_needed(G, route, state)
        assert not did_reroute
        assert same_route == route
