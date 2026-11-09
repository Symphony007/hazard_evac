"""Tests for the road-network graph loader."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, StudyAreaConfig
from src.graph.loader import get_or_build_graph


# Use the default study-area config for tests.
CONFIG_PATH = PROJECT_ROOT / "configs" / "study_area.yaml"


@pytest.fixture(scope="module")
def cfg() -> StudyAreaConfig:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def graph(cfg: StudyAreaConfig) -> nx.MultiDiGraph:
    """Load the graph (from cache if available, otherwise build)."""
    return get_or_build_graph(cfg)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_graph_loads(graph: nx.MultiDiGraph) -> None:
    """Graph loads or builds without error and is non-empty."""
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0


def test_strongly_connected(graph: nx.MultiDiGraph) -> None:
    """Graph is a single strongly-connected component."""
    assert nx.is_strongly_connected(graph)


def test_edge_attributes_present(graph: nx.MultiDiGraph) -> None:
    """Every edge has the required attributes with positive values."""
    required_float = ("length", "speed_kph", "travel_time")
    for u, v, k, data in graph.edges(keys=True, data=True):
        for attr in required_float:
            assert attr in data, f"Edge ({u},{v},{k}) missing '{attr}'"
            assert float(data[attr]) > 0, (
                f"Edge ({u},{v},{k}) '{attr}' = {data[attr]} (expected > 0)"
            )
        assert "capacity" in data, f"Edge ({u},{v},{k}) missing 'capacity'"
        assert int(data["capacity"]) > 0, (
            f"Edge ({u},{v},{k}) capacity = {data['capacity']} (expected > 0)"
        )


def test_capacity_minimum(graph: nx.MultiDiGraph, cfg: StudyAreaConfig) -> None:
    """Every edge capacity is an int >= min_edge_capacity."""
    for u, v, k, data in graph.edges(keys=True, data=True):
        cap = data["capacity"]
        assert isinstance(cap, int), f"Edge ({u},{v},{k}) capacity type {type(cap)}"
        assert cap >= cfg.min_edge_capacity, (
            f"Edge ({u},{v},{k}) capacity {cap} < min {cfg.min_edge_capacity}"
        )

