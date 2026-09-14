"""
visualize.py -- Folium map rendering for multi-agent evacuation (Layer 2).

Extended from Layer 1 to support:
  - Multiple agent routes rendered in distinct colours.
  - Road segments coloured/weighted by congestion load (heat-map style).
  - Hazard zone overlay (same as Layer 1).
  - Legend showing colour assignments.

Design notes (for viva)
-----------------------
* Each agent gets a colour from a perceptually-distinct palette.  With
  ~20 agents, we cycle through a curated 10-colour palette.
* Congestion is shown by drawing *all edges that carry load > 0* with
  thickness and colour proportional to their load.  Unloaded edges are
  not drawn to keep the map readable.
* The map uses OpenStreetMap tiles (no API key required).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import folium
import networkx as nx

from .congestion import CongestionModel
from .hazard import HazardState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette for agents (10 perceptually distinct colours)
# ---------------------------------------------------------------------------

AGENT_COLOURS = [
    "#2980b9",  # blue
    "#27ae60",  # green
    "#8e44ad",  # purple
    "#e67e22",  # orange
    "#1abc9c",  # teal
    "#e84393",  # pink
    "#00b894",  # mint
    "#6c5ce7",  # indigo
    "#fdcb6e",  # gold
    "#00cec9",  # cyan
]


def _agent_colour(idx: int) -> str:
    """Return a colour for agent `idx`, cycling through the palette."""
    return AGENT_COLOURS[idx % len(AGENT_COLOURS)]


# ---------------------------------------------------------------------------
# Congestion colour scale (green -> yellow -> red)
# ---------------------------------------------------------------------------

def _congestion_colour(load: int, max_load: int) -> str:
    """Map a load value to a green-yellow-red colour string."""
    if max_load <= 0:
        return "#2ecc71"  # green (no congestion)
    ratio = min(load / max_load, 1.0)
    # Interpolate: green (0) -> yellow (0.5) -> red (1.0)
    if ratio <= 0.5:
        # green -> yellow
        r = int(46 + (255 - 46) * (ratio / 0.5))
        g = int(204 + (215 - 204) * (ratio / 0.5))
        b = int(113 + (0 - 113) * (ratio / 0.5))
    else:
        # yellow -> red
        t = (ratio - 0.5) / 0.5
        r = int(255)
        g = int(215 - 215 * t)
        b = int(0)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Single-agent map (Layer 1 compatible)
# ---------------------------------------------------------------------------

def render_map(
    graph: nx.MultiDiGraph,
    route: List[int],
    hazard_state: HazardState,
    filename: Optional[str] = None,
) -> folium.Map:
    """
    Render a single-agent evacuation map (Layer 1 interface preserved).
    """
    mid_node = route[len(route) // 2]
    centre_lat = graph.nodes[mid_node]["y"]
    centre_lon = graph.nodes[mid_node]["x"]

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=15,
        tiles="OpenStreetMap",
    )

    _draw_hazardous_edges(m, graph, hazard_state)
    _draw_hazard_circle(m, graph, hazard_state)
    _draw_route(m, graph, route, colour="#2980b9", label="Evacuation route")
    _add_hazard_marker(m, graph, hazard_state)

    if filename:
        out = Path(filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out))
        logger.info("Map saved -> %s", out)

    return m


# ---------------------------------------------------------------------------
# Multi-agent map (Layer 2)
# ---------------------------------------------------------------------------

def render_multi_agent_map(
    graph: nx.MultiDiGraph,
    agent_routes: Dict[int, List[int]],
    hazard_state: HazardState,
    congestion_model: Optional[CongestionModel] = None,
    filename: Optional[str] = None,
) -> folium.Map:
    """
    Render a multi-agent evacuation map with congestion visualization.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The road-network graph.
    agent_routes : dict[int, list[int]]
        Mapping from agent index to their route (node ID list).
    hazard_state : HazardState
        Current hazard snapshot.
    congestion_model : CongestionModel or None
        If provided, edges are coloured by congestion load.
    filename : str or None
        If provided, save to this HTML path.

    Returns
    -------
    folium.Map
    """
    # --- Centre the map on the graph's mean coordinates ---
    all_lats = [graph.nodes[n]["y"] for n in graph.nodes]
    all_lons = [graph.nodes[n]["x"] for n in graph.nodes]
    centre_lat = sum(all_lats) / len(all_lats)
    centre_lon = sum(all_lons) / len(all_lons)

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    # --- 1. Congestion heat layer (below routes) ---
    if congestion_model is not None:
        _draw_congestion_layer(m, graph, congestion_model)

    # --- 2. Hazard zone ---
    _draw_hazardous_edges(m, graph, hazard_state)
    _draw_hazard_circle(m, graph, hazard_state)
    _add_hazard_marker(m, graph, hazard_state)

    # --- 3. Agent routes (each in a distinct colour) ---
    for idx, route in agent_routes.items():
        colour = _agent_colour(idx)
        _draw_route(
            m, graph, route,
            colour=colour,
            label=f"Agent {idx}",
            weight=3,
            opacity=0.7,
        )

    # --- 4. Add origin/destination markers for a few representative agents ---
    _add_agent_markers(m, graph, agent_routes)

    # --- Save ---
    if filename:
        out = Path(filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out))
        logger.info("Multi-agent map saved -> %s", out)

    return m


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_congestion_layer(
    m: folium.Map,
    graph: nx.MultiDiGraph,
    congestion_model: CongestionModel,
) -> None:
    """Draw edges coloured by congestion load (only edges with load > 0)."""
    stats = congestion_model.get_stats()
    max_load = stats.max_load
    if max_load == 0:
        return

    for u, v, key, data in graph.edges(keys=True, data=True):
        load = congestion_model.get_load(u, v, key)
        if load <= 0:
            continue
        colour = _congestion_colour(load, max_load)
        weight = 2 + 4 * (load / max_load)  # thicker = more congested
        coords = [
            [graph.nodes[u]["y"], graph.nodes[u]["x"]],
            [graph.nodes[v]["y"], graph.nodes[v]["x"]],
        ]
        folium.PolyLine(
            coords,
            color=colour,
            weight=weight,
            opacity=0.6,
            tooltip=f"Load: {load} / cap: {congestion_model._capacity.get((u,v,key), '?')}",
        ).add_to(m)


def _draw_hazardous_edges(
    m: folium.Map,
    graph: nx.MultiDiGraph,
    hazard_state: HazardState,
) -> None:
    """Draw each hazardous edge as a red polyline."""
    for u, v, _key in hazard_state.hazardous_edges:
        if not graph.has_node(u) or not graph.has_node(v):
            continue
        coords = [
            [graph.nodes[u]["y"], graph.nodes[u]["x"]],
            [graph.nodes[v]["y"], graph.nodes[v]["x"]],
        ]
        folium.PolyLine(
            coords,
            color="#e74c3c",
            weight=3,
            opacity=0.7,
            tooltip="Hazardous",
        ).add_to(m)


def _draw_hazard_circle(
    m: folium.Map,
    graph: nx.MultiDiGraph,
    hazard_state: HazardState,
) -> None:
    """Draw a translucent circle showing the hazard spread radius."""
    origin_data = graph.nodes[hazard_state.origin_node]
    folium.Circle(
        location=[origin_data["y"], origin_data["x"]],
        radius=hazard_state.radius_m,
        color="#c0392b",
        fill=True,
        fill_color="#e74c3c",
        fill_opacity=0.15,
        weight=2,
        tooltip=f"Hazard zone -- radius {hazard_state.radius_m:.0f} m",
    ).add_to(m)


def _draw_route(
    m: folium.Map,
    graph: nx.MultiDiGraph,
    route: List[int],
    colour: str = "#2980b9",
    label: str = "Route",
    weight: int = 5,
    opacity: float = 0.9,
) -> None:
    """Draw a route as a coloured polyline."""
    coords = [
        [graph.nodes[n]["y"], graph.nodes[n]["x"]]
        for n in route
    ]
    folium.PolyLine(
        coords,
        color=colour,
        weight=weight,
        opacity=opacity,
        tooltip=label,
    ).add_to(m)


def _add_hazard_marker(
    m: folium.Map,
    graph: nx.MultiDiGraph,
    hazard_state: HazardState,
) -> None:
    """Add a marker at the hazard origin."""
    node = hazard_state.origin_node
    folium.Marker(
        location=[graph.nodes[node]["y"], graph.nodes[node]["x"]],
        popup=f"Hazard origin (node {node})",
        icon=folium.Icon(color="darkred", icon="exclamation-triangle", prefix="fa"),
    ).add_to(m)


def _add_agent_markers(
    m: folium.Map,
    graph: nx.MultiDiGraph,
    agent_routes: Dict[int, List[int]],
) -> None:
    """
    Add origin/destination markers for agents.

    To avoid visual clutter with many agents, we deduplicate by
    origin and destination node -- each unique endpoint gets one marker.
    """
    origins_seen = set()
    dests_seen = set()

    for idx, route in agent_routes.items():
        origin = route[0]
        dest = route[-1]

        if origin not in origins_seen:
            origins_seen.add(origin)
            folium.CircleMarker(
                location=[graph.nodes[origin]["y"], graph.nodes[origin]["x"]],
                radius=6,
                color="#27ae60",
                fill=True,
                fill_color="#27ae60",
                fill_opacity=0.8,
                tooltip=f"Origin (Agent {idx})",
            ).add_to(m)

        if dest not in dests_seen:
            dests_seen.add(dest)
            folium.CircleMarker(
                location=[graph.nodes[dest]["y"], graph.nodes[dest]["x"]],
                radius=8,
                color="#e74c3c",
                fill=True,
                fill_color="#e74c3c",
                fill_opacity=0.9,
                tooltip=f"Destination (safe zone)",
            ).add_to(m)
