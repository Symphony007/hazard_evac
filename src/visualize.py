"""
visualize.py — Folium-based map rendering for evacuation routes.

Produces an interactive Leaflet map (via folium) showing:
  • The road network's hazardous edges highlighted in red.
  • The current evacuation route drawn as a bold blue polyline.
  • A translucent red circle representing the hazard radius.
  • Markers for origin (green), destination (red), and hazard centre.

Design notes (for viva)
-----------------------
* We intentionally do **not** draw every edge of the graph — on a city-
  scale network that would be tens of thousands of polylines and very
  slow to render.  Instead we draw only *hazardous* edges (the danger
  zone) plus the route itself.
* The map is returned as a ``folium.Map`` object so it works both in
  Jupyter notebooks (inline display) and as a standalone HTML file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import folium
import networkx as nx

from .hazard import HazardState

logger = logging.getLogger(__name__)


def render_map(
    graph: nx.MultiDiGraph,
    route: List[int],
    hazard_state: HazardState,
    filename: Optional[str] = None,
) -> folium.Map:
    """
    Render an interactive evacuation map.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The road-network graph (unprojected WGS-84).
    route : list of int
        Ordered node IDs forming the evacuation route.
    hazard_state : HazardState
        Current hazard snapshot (defines the danger zone).
    filename : str or None
        If provided, save the map as an HTML file at this path.
        Parent directories are created automatically.

    Returns
    -------
    folium.Map
        The rendered map object.
    """
    # --- Compute map centre from the route midpoint ---
    mid_node = route[len(route) // 2]
    centre_lat = graph.nodes[mid_node]["y"]
    centre_lon = graph.nodes[mid_node]["x"]

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=15,
        tiles="OpenStreetMap",
    )

    # --- 1. Draw hazardous edges in red ---
    _draw_hazardous_edges(m, graph, hazard_state)

    # --- 2. Draw the hazard radius circle ---
    _draw_hazard_circle(m, graph, hazard_state)

    # --- 3. Draw the evacuation route ---
    _draw_route(m, graph, route)

    # --- 4. Add origin / destination markers ---
    _add_endpoint_markers(m, graph, route)

    # --- 5. Add hazard origin marker ---
    _add_hazard_marker(m, graph, hazard_state)

    # --- Save to file if requested ---
    if filename:
        out = Path(filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out))
        logger.info("Map saved → %s", out)

    return m


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_hazardous_edges(
    m: folium.Map,
    graph: nx.MultiDiGraph,
    hazard_state: HazardState,
) -> None:
    """Draw each hazardous edge as a red polyline on the map."""
    for u, v, _key in hazard_state.hazardous_edges:
        if not graph.has_node(u) or not graph.has_node(v):
            continue
        coords = [
            [graph.nodes[u]["y"], graph.nodes[u]["x"]],
            [graph.nodes[v]["y"], graph.nodes[v]["x"]],
        ]
        folium.PolyLine(
            coords,
            color="#e74c3c",      # vibrant red
            weight=3,
            opacity=0.7,
            tooltip="⚠ Hazardous",
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
        tooltip=f"Hazard zone — radius {hazard_state.radius_m:.0f} m",
    ).add_to(m)


def _draw_route(
    m: folium.Map,
    graph: nx.MultiDiGraph,
    route: List[int],
) -> None:
    """Draw the evacuation route as a bold blue polyline."""
    coords = [
        [graph.nodes[n]["y"], graph.nodes[n]["x"]]
        for n in route
    ]
    folium.PolyLine(
        coords,
        color="#2980b9",      # strong blue
        weight=5,
        opacity=0.9,
        tooltip="Evacuation route",
    ).add_to(m)


def _add_endpoint_markers(
    m: folium.Map,
    graph: nx.MultiDiGraph,
    route: List[int],
) -> None:
    """Add green (origin) and red (destination) markers."""
    origin = route[0]
    dest = route[-1]

    folium.Marker(
        location=[graph.nodes[origin]["y"], graph.nodes[origin]["x"]],
        popup="🟢 Origin (evacuation start)",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)

    folium.Marker(
        location=[graph.nodes[dest]["y"], graph.nodes[dest]["x"]],
        popup="🔴 Destination (safe zone)",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
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
        popup=f"☢ Hazard origin (node {node})",
        icon=folium.Icon(color="darkred", icon="exclamation-triangle", prefix="fa"),
    ).add_to(m)
