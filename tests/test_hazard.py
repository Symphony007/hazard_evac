"""Tests for the flood hazard simulator."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HazardConfig, StudyAreaConfig, load_config
from src.graph.loader import get_or_build_graph
from src.hazard.flood import FloodHazard
from src.hazard.terrain import generate_terrain

GRAPH_CONFIG_PATH = PROJECT_ROOT / "configs" / "study_area.yaml"
HAZARD_CONFIG_PATH = PROJECT_ROOT / "configs" / "hazard.yaml"


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def graph_cfg() -> StudyAreaConfig:
    return load_config(GRAPH_CONFIG_PATH)


@pytest.fixture(scope="module")
def graph(graph_cfg: StudyAreaConfig) -> nx.MultiDiGraph:
    return get_or_build_graph(graph_cfg)


@pytest.fixture(scope="module")
def hazard_cfg() -> HazardConfig:
    return load_config(HAZARD_CONFIG_PATH, HazardConfig)


@pytest.fixture(scope="module")
def hazard(graph: nx.MultiDiGraph, hazard_cfg: HazardConfig) -> FloodHazard:
    return FloodHazard(graph, hazard_cfg)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_sources_arrival_zero(hazard: FloodHazard) -> None:
    """All source nodes have arrival time 0."""
    for s in hazard.sources:
        assert hazard.node_arrival_time(s) == 0.0, (
            f"Source {s} has arrival time {hazard.node_arrival_time(s)}"
        )


def test_finite_arrivals_nonneg(hazard: FloodHazard, graph: nx.MultiDiGraph) -> None:
    """All finite node arrival times are >= 0."""
    for n in graph.nodes:
        t = hazard.node_arrival_time(n)
        if np.isfinite(t):
            assert t >= 0.0, f"Node {n} arrival time {t} < 0"


def test_deterministic_same_seed(
    graph: nx.MultiDiGraph, hazard_cfg: HazardConfig
) -> None:
    """Same config → identical arrival times."""
    h1 = FloodHazard(graph, hazard_cfg)
    h2 = FloodHazard(graph, hazard_cfg)
    sample = list(graph.nodes)[:200]
    for n in sample:
        assert h1.node_arrival_time(n) == h2.node_arrival_time(n), (
            f"Node {n}: {h1.node_arrival_time(n)} != {h2.node_arrival_time(n)}"
        )


def test_different_seed_different_arrivals(
    graph: nx.MultiDiGraph, hazard_cfg: HazardConfig
) -> None:
    """Different seed → different arrival times (at least for some nodes)."""
    h1 = FloodHazard(graph, hazard_cfg)
    h2 = FloodHazard(graph, replace(hazard_cfg, seed=999))
    sample = list(graph.nodes)[:200]
    any_diff = any(
        h1.node_arrival_time(n) != h2.node_arrival_time(n) for n in sample
    )
    assert any_diff, "Expected at least one different arrival time"


def test_terrain_independent_of_flood_seed(
    graph: nx.MultiDiGraph, hazard_cfg: HazardConfig
) -> None:
    """Same terrain_seed with different flood seeds → identical terrain."""
    h1 = FloodHazard(graph, hazard_cfg)
    h2 = FloodHazard(graph, replace(hazard_cfg, seed=999))
    sample = list(graph.nodes)[:200]
    for n in sample:
        assert h1.elevation[n] == h2.elevation[n], (
            f"Node {n}: elev {h1.elevation[n]} != {h2.elevation[n]}"
        )


def test_closure_monotonic(
    hazard: FloodHazard, graph: nx.MultiDiGraph
) -> None:
    """Once an edge is closed it stays closed (checked on a sample)."""
    sample_edges = list(graph.edges(keys=True))[:50]
    times = np.arange(0, 3601, 60)  # 0 to 60 min in 1-min steps
    for u, v, k in sample_edges:
        was_closed = False
        for t in times:
            closed = hazard.is_edge_closed(u, v, k, float(t))
            if was_closed:
                assert closed, (
                    f"Edge ({u},{v},{k}) reopened at t={t}s"
                )
            if closed:
                was_closed = True


def test_edge_arrival_is_min_of_nodes(
    hazard: FloodHazard, graph: nx.MultiDiGraph
) -> None:
    """edge_arrival_time(u,v,k) == min(T(u), T(v))."""
    for u, v, k in list(graph.edges(keys=True))[:200]:
        expected = min(hazard.node_arrival_time(u), hazard.node_arrival_time(v))
        actual = hazard.edge_arrival_time(u, v, k)
        assert actual == expected, (
            f"Edge ({u},{v},{k}): arrival {actual} != min({hazard.node_arrival_time(u)}, "
            f"{hazard.node_arrival_time(v)}) = {expected}"
        )


def test_speed_factor_bounds(
    hazard: FloodHazard, graph: nx.MultiDiGraph, hazard_cfg: HazardConfig
) -> None:
    """edge_speed_factor is in [0, 1]; 1.0 well before; 0.0 at/after arrival."""
    ramp_s = hazard_cfg.ramp_min * 60.0
    sample_edges = list(graph.edges(keys=True))[:50]
    for u, v, k in sample_edges:
        at = hazard.edge_arrival_time(u, v, k)
        if not np.isfinite(at):
            assert hazard.edge_speed_factor(u, v, k, 0.0) == 1.0
            continue

        # Well before arrival
        t_safe = max(0.0, at - ramp_s - 500.0)
        assert hazard.edge_speed_factor(u, v, k, t_safe) == 1.0

        # At arrival → 0.0
        assert hazard.edge_speed_factor(u, v, k, at) == 0.0

        # After arrival → 0.0
        assert hazard.edge_speed_factor(u, v, k, at + 100.0) == 0.0

        # Sweep: always in [0, 1]
        for t in np.linspace(0.0, at + 100.0, 30):
            sf = hazard.edge_speed_factor(u, v, k, float(t))
            assert 0.0 <= sf <= 1.0, (
                f"Edge ({u},{v},{k}) speed_factor={sf} out of [0,1] at t={t}"
            )


def test_closure_schedule_sorted(
    hazard: FloodHazard, graph: nx.MultiDiGraph
) -> None:
    """closure_schedule is sorted and has one entry per finite-arrival edge."""
    schedule = hazard.closure_schedule()

    # Sorted by time
    times = [t for t, _ in schedule]
    assert times == sorted(times), "closure_schedule is not sorted"

    # Count matches edges with finite arrival
    finite_count = sum(
        1 for u, v, k in graph.edges(keys=True)
        if np.isfinite(hazard.edge_arrival_time(u, v, k))
    )
    assert len(schedule) == finite_count, (
        f"schedule length {len(schedule)} != finite-arrival edges {finite_count}"
    )


def test_closed_edges_nondecreasing(
    hazard: FloodHazard, hazard_cfg: HazardConfig
) -> None:
    """Number of closed edges is non-decreasing over time."""
    times_min = [0, 5, 10, 15, 20, 30, 45, 60]
    prev_count = 0
    for t_min in times_min:
        t_sec = t_min * 60.0
        count = len(hazard.closed_edges(t_sec))
        assert count >= prev_count, (
            f"|closed_edges| dropped from {prev_count} to {count} "
            f"between t={times_min[times_min.index(t_min)-1]} and t={t_min} min"
        )
        prev_count = count

