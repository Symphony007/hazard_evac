"""Kolkata road-network graph loader using OSMnx.

Builds, caches, and loads a drivable road graph with per-edge attributes
needed by the hazard and routing layers:
  length (m), speed_kph (km/h), travel_time (s), free_flow_time (s),
  capacity (vehicles), lanes_n (int), flow_capacity (veh/min),
  and node/edge coordinates.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import osmnx as ox

if TYPE_CHECKING:
    from src.config import StudyAreaConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_lanes(lanes_raw: str | list | int | float | None, default: int) -> int:
    """Parse the OSM 'lanes' tag into an integer lane count.

    The tag may be:
      - an int or float already,
      - a string like "2",
      - a semicolon-separated string like "2;3" (take max),
      - a list of values,
      - or missing / None  (fall back to *default*).
    """
    if lanes_raw is None:
        return default

    # Already numeric
    if isinstance(lanes_raw, (int, float)):
        return max(int(lanes_raw), 1)

    # List of values → take the max
    if isinstance(lanes_raw, list):
        parsed = [_parse_lanes(v, default) for v in lanes_raw]
        return max(parsed) if parsed else default

    # String – may contain semicolons
    if isinstance(lanes_raw, str):
        parts = lanes_raw.replace(",", ";").split(";")
        values: list[int] = []
        for p in parts:
            p = p.strip()
            try:
                values.append(max(int(float(p)), 1))
            except (ValueError, TypeError):
                continue
        return max(values) if values else default

    return default


def _add_lanes_n(G: nx.MultiDiGraph, cfg: "StudyAreaConfig") -> None:
    """Store the parsed lane count as *lanes_n* (int) on every edge."""
    for u, v, k, data in G.edges(keys=True, data=True):
        data["lanes_n"] = _parse_lanes(data.get("lanes"), cfg.default_lanes)


def _add_capacity(G: nx.MultiDiGraph, cfg: "StudyAreaConfig") -> None:
    """Add an integer *capacity* attribute to every edge.

    capacity = lanes_n × (length_m / 1000) × capacity_per_lane_veh_per_km,
    rounded, and never below min_edge_capacity.
    Requires *lanes_n* to be set first (see :func:`_add_lanes_n`).
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        length_m: float = data["length"]
        raw_cap = data["lanes_n"] * (length_m / 1000.0) * cfg.capacity_per_lane_veh_per_km
        capacity = max(round(raw_cap), cfg.min_edge_capacity)
        data["capacity"] = capacity


def _add_flow_capacity(G: nx.MultiDiGraph, cfg: "StudyAreaConfig") -> None:
    """Add *flow_capacity* (veh/min) to every edge.

    flow_capacity = lanes_n × flow_rate_veh_per_min_per_lane.
    Requires *lanes_n* to be set first.
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        data["flow_capacity"] = float(
            data["lanes_n"] * cfg.flow_rate_veh_per_min_per_lane
        )


def _add_free_flow_time(G: nx.MultiDiGraph) -> None:
    """Copy travel_time → free_flow_time for every edge."""
    for u, v, k, data in G.edges(keys=True, data=True):
        data["free_flow_time"] = data["travel_time"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(cfg: "StudyAreaConfig") -> nx.MultiDiGraph:
    """Download, clean, and enrich the road graph.

    Steps:
      1. Download from OSM via ``ox.graph_from_point``.
      2. Keep only the largest strongly-connected component.
      3. Add edge speeds and travel times (OSMnx helpers).
      4. Add *lanes_n*, *capacity*, *flow_capacity*.
      5. Add *free_flow_time* = travel_time.

    Parameters
    ----------
    cfg : StudyAreaConfig
        Study-area configuration (center, radius, capacity params, …).

    Returns
    -------
    nx.MultiDiGraph
        The enriched road graph.
    """
    G = ox.graph_from_point(
        (cfg.center_lat, cfg.center_lon),
        dist=cfg.dist_m,
        network_type=cfg.network_type,
        simplify=True,
    )

    # Largest strongly connected component
    G = ox.truncate.largest_component(G, strongly=True)

    # Speeds & travel times
    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)

    # Lane count, storage capacity, flow capacity, free-flow snapshot
    _add_lanes_n(G, cfg)
    _add_capacity(G, cfg)
    _add_flow_capacity(G, cfg)
    _add_free_flow_time(G)

    return G


def save_graph(G: nx.MultiDiGraph, path: str | Path) -> Path:
    """Persist graph as GraphML. Creates parent dirs if needed.

    Returns the resolved path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.io.save_graphml(G, filepath=path)
    return path.resolve()


def load_graph(path: str | Path) -> nx.MultiDiGraph:
    """Load a cached GraphML and coerce numeric edge attributes.

    GraphML round-trips may store numbers as strings; this function
    ensures all custom numeric attributes come back as proper types.
    """
    path = Path(path)
    G = ox.io.load_graphml(filepath=path)

    _FLOAT_ATTRS = ("length", "speed_kph", "travel_time", "free_flow_time",
                     "flow_capacity")
    _INT_ATTRS = ("capacity", "lanes_n")

    for u, v, k, data in G.edges(keys=True, data=True):
        for attr in _FLOAT_ATTRS:
            if attr in data:
                data[attr] = float(data[attr])
        for attr in _INT_ATTRS:
            if attr in data:
                data[attr] = int(round(float(data[attr])))

    return G


def get_or_build_graph(cfg: "StudyAreaConfig") -> nx.MultiDiGraph:
    """Return the cached graph if it exists, otherwise build and save it.

    Parameters
    ----------
    cfg : StudyAreaConfig
        Study-area configuration.

    Returns
    -------
    nx.MultiDiGraph
        The road graph with all required edge attributes.
    """
    cache = Path(cfg.cache_path)
    if cache.exists():
        print(f"[loader] Loading cached graph from {cache}")
        return load_graph(cache)

    print("[loader] Building graph from OSM …")
    G = build_graph(cfg)
    saved = save_graph(G, cache)
    print(f"[loader] Graph saved to {saved}")
    return G
