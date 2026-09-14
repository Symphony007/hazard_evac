"""
test_layer2.py -- Tests for the Layer 2 multi-agent congestion system.

Verifies:
  1. BPR cost function increases with load.
  2. add_route / remove_route correctly update edge loads.
  3. Multiple agents with overlapping OD pairs produce diverse routes
     under congestion-aware routing.
  4. Agents still avoid hazardous edges even with congestion active.
  5. Congestion-aware routing produces lower max edge load than naive.

Run with:
    .\\venv\\Scripts\\python.exe -m pytest layer2/tests/test_layer2.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure layer2/ root is on sys.path so `src` is importable
# ---------------------------------------------------------------------------
LAYER2_ROOT = Path(__file__).resolve().parent.parent
if str(LAYER2_ROOT) not in sys.path:
    sys.path.insert(0, str(LAYER2_ROOT))

from src.graph import load_graph, CACHE_DIR
from src.hazard import HazardSimulation, HazardState
from src.router import compute_route, NoRouteError
from src.congestion import CongestionModel, DEFAULT_ALPHA, DEFAULT_BETA
from src.multi_agent_router import (
    route_all_agents,
    route_all_agents_naive,
    RoutingResult,
)


# ---------------------------------------------------------------------------
# Shared fixture -- small Kolkata graph
# ---------------------------------------------------------------------------

_KOLKATA_POINT = (22.5726, 88.3639)
_KOLKATA_DIST = 1500


@pytest.fixture(scope="module")
def kolkata_graph():
    """Load (or cache-load) a small road-network graph around Kolkata."""
    G = load_graph(point=_KOLKATA_POINT, dist=_KOLKATA_DIST, network_type="drive")
    return G


@pytest.fixture(scope="module")
def sample_route(kolkata_graph):
    """Compute a sample route across the graph for test reuse."""
    nodes = list(kolkata_graph.nodes)
    return compute_route(kolkata_graph, nodes[0], nodes[-1])


# ---------------------------------------------------------------------------
# Test: BPR congestion model
# ---------------------------------------------------------------------------

class TestBPRCostFunction:
    """Verify that the BPR cost formula behaves correctly."""

    def test_cost_increases_with_load(self, kolkata_graph, sample_route):
        """Adding load to an edge should increase its BPR-adjusted cost."""
        cm = CongestionModel(kolkata_graph)
        u, v = sample_route[0], sample_route[1]

        cost_empty = cm.get_adjusted_cost(u, v, 0)
        cm.add_route(sample_route)
        cost_loaded = cm.get_adjusted_cost(u, v, 0)

        assert cost_loaded > cost_empty, (
            f"BPR cost should increase with load: "
            f"empty={cost_empty:.2f}, loaded={cost_loaded:.2f}"
        )

    def test_cost_grows_nonlinearly(self, kolkata_graph, sample_route):
        """Cost should grow faster at higher loads (beta=4 -> steep curve)."""
        cm = CongestionModel(kolkata_graph)
        u, v = sample_route[0], sample_route[1]

        costs = []
        for i in range(5):
            costs.append(cm.get_adjusted_cost(u, v, 0))
            cm.add_route(sample_route)

        # Each successive cost increase should be larger than the previous
        deltas = [costs[i+1] - costs[i] for i in range(len(costs)-1)]
        for i in range(len(deltas)-1):
            assert deltas[i+1] >= deltas[i], (
                f"BPR cost growth should be non-linear: deltas={deltas}"
            )

    def test_zero_load_equals_base_length(self, kolkata_graph):
        """With zero load, BPR cost should equal the base edge length."""
        cm = CongestionModel(kolkata_graph)
        # Pick any edge
        u, v, key, data = next(iter(kolkata_graph.edges(keys=True, data=True)))
        base = float(data.get("length", 1.0))
        cost = cm.get_adjusted_cost(u, v, key, base_length=base)
        assert abs(cost - base) < 1e-6, (
            f"Zero-load cost should equal base: base={base}, cost={cost}"
        )


class TestLoadTracking:
    """Verify add_route and remove_route update loads correctly."""

    def test_add_route_increments_load(self, kolkata_graph, sample_route):
        """Each call to add_route should increment edge loads by 1."""
        cm = CongestionModel(kolkata_graph)
        u, v = sample_route[0], sample_route[1]

        assert cm.get_load(u, v) == 0
        cm.add_route(sample_route)
        assert cm.get_load(u, v) == 1
        cm.add_route(sample_route)
        assert cm.get_load(u, v) == 2

    def test_remove_route_decrements_load(self, kolkata_graph, sample_route):
        """remove_route should reverse the effect of add_route."""
        cm = CongestionModel(kolkata_graph)
        u, v = sample_route[0], sample_route[1]

        cm.add_route(sample_route)
        cm.add_route(sample_route)
        assert cm.get_load(u, v) == 2

        cm.remove_route(sample_route)
        assert cm.get_load(u, v) == 1

    def test_remove_does_not_go_negative(self, kolkata_graph, sample_route):
        """Removing from an empty edge should not produce negative load."""
        cm = CongestionModel(kolkata_graph)
        u, v = sample_route[0], sample_route[1]

        cm.remove_route(sample_route)  # no prior add
        assert cm.get_load(u, v) == 0

    def test_max_load_property(self, kolkata_graph, sample_route):
        """max_load should reflect the highest edge load."""
        cm = CongestionModel(kolkata_graph)
        cm.add_route(sample_route)
        cm.add_route(sample_route)
        assert cm.max_load >= 2

    def test_stats_are_consistent(self, kolkata_graph, sample_route):
        """CongestionStats fields should be internally consistent."""
        cm = CongestionModel(kolkata_graph)
        cm.add_route(sample_route)
        stats = cm.get_stats()
        assert stats.loaded_edges > 0
        assert stats.max_load >= 1
        assert stats.total_volume >= stats.loaded_edges


# ---------------------------------------------------------------------------
# Test: Multi-agent route diversity
# ---------------------------------------------------------------------------

class TestMultiAgentDiversity:
    """Verify that congestion-aware routing produces diverse routes."""

    def test_overlapping_od_pairs_produce_diverse_routes(self, kolkata_graph):
        """
        When many agents share the same OD pair, congestion-aware routing
        should produce more than one unique route.
        """
        nodes = list(kolkata_graph.nodes)
        origin, dest = nodes[0], nodes[-1]

        # 10 agents all going from the same origin to the same destination
        od_pairs = [(origin, dest)] * 10

        result = route_all_agents(kolkata_graph, od_pairs)

        assert len(result.agent_routes) == 10, "All 10 agents should be routed"
        assert result.num_unique_routes > 1, (
            f"Expected route diversity: got {result.num_unique_routes} unique "
            f"routes for 10 agents with same OD pair"
        )

    def test_naive_produces_fewer_unique_routes(self, kolkata_graph):
        """
        Naive routing (no congestion) should produce fewer unique routes
        than congestion-aware routing for the same OD pairs.
        """
        nodes = list(kolkata_graph.nodes)
        origin, dest = nodes[0], nodes[-1]
        od_pairs = [(origin, dest)] * 10

        naive_result = route_all_agents_naive(kolkata_graph, od_pairs)
        aware_result = route_all_agents(kolkata_graph, od_pairs)

        # Naive should give all agents the same route (1 unique)
        assert naive_result.num_unique_routes == 1, (
            f"Naive routing should produce 1 unique route, got "
            f"{naive_result.num_unique_routes}"
        )
        # Aware should produce more
        assert aware_result.num_unique_routes > naive_result.num_unique_routes


# ---------------------------------------------------------------------------
# Test: Hazard avoidance under congestion
# ---------------------------------------------------------------------------

class TestHazardAvoidanceWithCongestion:
    """Verify that hazard avoidance still works when congestion is active."""

    def test_agents_avoid_hazardous_edges(self, kolkata_graph, sample_route):
        """
        Even with congestion active, no agent's route should use a
        hazardous edge.
        """
        # Place hazard at the midpoint of a sample route
        mid_node = sample_route[len(sample_route) // 2]
        sim = HazardSimulation(kolkata_graph, origin_node=mid_node, spread_rate_m=200)
        state = sim.get_hazard_state(2)  # 400 m radius

        nodes = list(kolkata_graph.nodes)
        origin, dest = nodes[0], nodes[-1]
        od_pairs = [(origin, dest)] * 5

        result = route_all_agents(
            kolkata_graph, od_pairs, hazard_state=state
        )

        for idx, route in result.agent_routes.items():
            for u, v in zip(route[:-1], route[1:]):
                assert (u, v, 0) not in state.hazardous_edges, (
                    f"Agent {idx}'s route uses hazardous edge ({u}, {v})"
                )


# ---------------------------------------------------------------------------
# Test: Congestion-aware beats naive on max edge load
# ---------------------------------------------------------------------------

class TestCongestionVsNaive:
    """Compare congestion-aware vs naive routing on the same input."""

    def test_aware_has_lower_max_edge_load(self, kolkata_graph):
        """
        Congestion-aware routing should spread load more evenly,
        producing a lower (or equal) maximum edge load than naive.
        """
        nodes = list(kolkata_graph.nodes)
        origin, dest = nodes[0], nodes[-1]
        od_pairs = [(origin, dest)] * 15

        naive = route_all_agents_naive(kolkata_graph, od_pairs)
        aware = route_all_agents(kolkata_graph, od_pairs)

        assert aware.max_edge_load <= naive.max_edge_load, (
            f"Congestion-aware max load ({aware.max_edge_load}) should be "
            f"<= naive max load ({naive.max_edge_load})"
        )
