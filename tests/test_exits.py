"""Unit tests for sector-based boundary exit placement and legacy fallback."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import networkx as nx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ScenarioConfig
from src.scenario import ExitList, build_exits


class StubHazard:
    """Minimal HazardModel stub providing predetermined node arrival times."""

    def __init__(
        self,
        arrival_times: dict[int, float],
        sources: list[int] | None = None,
    ) -> None:
        self._arrival_times = arrival_times
        self.sources = sources or []

    def node_arrival_time(self, node: int) -> float:
        return self._arrival_times.get(node, float("inf"))


def _build_toy_ring_graph(
    n_nodes: int = 24,
    radius_m: float = 1500.0,
    center_lat: float = 22.4985,
    center_lon: float = 88.3450,
) -> nx.MultiDiGraph:
    """Construct a toy circular graph with evenly spaced ring nodes."""
    G = nx.MultiDiGraph()
    lat0_rad = math.radians(center_lat)
    cos_lat0 = math.cos(lat0_rad)

    for i in range(n_nodes):
        angle_deg = i * (360.0 / n_nodes)
        theta = math.radians(angle_deg)
        dx_m = radius_m * math.cos(theta)
        dy_m = radius_m * math.sin(theta)

        lat = center_lat + dy_m / 111320.0
        lon = center_lon + dx_m / (111320.0 * cos_lat0)

        G.add_node(i, y=lat, x=lon)

    return G


def test_sector_mode_all_safe():
    """24 ring nodes with late arrival produce 6 exits, 1 per sector, 0 dropped."""
    G = _build_toy_ring_graph(24)
    cfg = ScenarioConfig(
        n_exits=6,
        exit_min_dist_from_center_m=1200.0,
        exit_mode="sector",
        exit_safe_margin_min=15.0,
        min_exits_required=3,
        depart_window_min=20.0,
    )
    # safe_time_s = (20 + 15) * 60 = 2100 s
    hazard = StubHazard(arrival_times={n: 5000.0 for n in range(24)})

    exits, info = build_exits(G, hazard, cfg)

    assert isinstance(exits, ExitList)
    assert len(exits) == 6
    assert info["n_exits_effective"] == 6
    assert info["sectors_with_exit"] == [0, 1, 2, 3, 4, 5]
    assert info["sectors_dropped"] == []
    assert info["used_fallback_exits"] is False
    assert info["n_fallback_exits"] == 0
    assert info["safe_time_s"] == pytest.approx(2100.0)


def test_sector_mode_two_sectors_dropped():
    """Make all nodes in two adjacent sectors flood before safe_time_s -> 2 dropped, 4 returned."""
    G = _build_toy_ring_graph(24)
    cfg = ScenarioConfig(
        n_exits=6,
        exit_min_dist_from_center_m=1200.0,
        exit_mode="sector",
        exit_safe_margin_min=15.0,
        min_exits_required=3,
        depart_window_min=20.0,
    )
    # Sector 1 has nodes 4..7; Sector 2 has nodes 8..11
    # Set their arrival to 1000 s (< 2100 s), others to 5000 s (> 2100 s)
    arrival_times = {n: 5000.0 for n in range(24)}
    for n in range(4, 12):
        arrival_times[n] = 1000.0

    hazard = StubHazard(arrival_times=arrival_times)
    exits, info = build_exits(G, hazard, cfg)

    assert len(exits) == 4
    assert info["n_exits_effective"] == 4
    assert info["sectors_with_exit"] == [0, 3, 4, 5]
    assert info["sectors_dropped"] == [1, 2]
    assert info["used_fallback_exits"] is False
    assert info["n_fallback_exits"] == 0


def test_sector_mode_fallback_when_under_min_required(capsys):
    """When fewer than min_exits_required sectors qualify, trigger fallback and warn."""
    G = _build_toy_ring_graph(24)
    cfg = ScenarioConfig(
        n_exits=6,
        exit_min_dist_from_center_m=1200.0,
        exit_mode="sector",
        exit_safe_margin_min=15.0,
        min_exits_required=3,
        depart_window_min=20.0,
    )
    # Only sector 0 (nodes 0..3) has safe nodes; sectors 1..5 flood early
    arrival_times = {n: 500.0 for n in range(24)}
    arrival_times[0] = 3000.0

    hazard = StubHazard(arrival_times=arrival_times)
    exits, info = build_exits(G, hazard, cfg)

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert info["used_fallback_exits"] is True
    assert len(exits) == 6
    assert info["n_exits_effective"] == 6
    # 5 exits did not meet the safe_time_s threshold
    assert info["n_fallback_exits"] == 5


def test_sector_mode_determinism():
    """Identical graph and hazard yield identical exits and info."""
    G = _build_toy_ring_graph(24)
    cfg = ScenarioConfig(
        n_exits=6,
        exit_min_dist_from_center_m=1200.0,
        exit_mode="sector",
    )
    arrival_times = {n: (n * 200.0) % 3600.0 for n in range(24)}
    hazard = StubHazard(arrival_times=arrival_times)

    exits1, info1 = build_exits(G, hazard, cfg)
    exits2, info2 = build_exits(G, hazard, cfg)

    assert exits1 == exits2
    assert info1 == info2


def test_legacy_latest_mode_reproducibility():
    """Legacy mode uses farthest-point sampling on unflooded nodes."""
    G = _build_toy_ring_graph(24)
    cfg = ScenarioConfig(
        n_exits=6,
        exit_min_dist_from_center_m=1200.0,
        exit_mode="legacy_latest",
    )
    arrival_times = {n: 4000.0 for n in range(24)}
    hazard = StubHazard(arrival_times=arrival_times)

    exits, info = build_exits(G, hazard, cfg, horizon_min=60.0)

    assert len(exits) == 6
    assert isinstance(exits, ExitList)
    assert info["used_fallback_exits"] is True or info["used_fallback_exits"] is False
    assert info["n_exits_effective"] == 6

