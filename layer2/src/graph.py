"""
graph.py -- Road-network graph loading with local caching (Layer 2 copy).

Identical to the Layer 1 version.  Copied here so that layer2/ is fully
self-contained and can run without layer1/ present.

This module provides a single entry-point, `load_graph`, that:
  1. Checks a local GraphML cache under `data/graph_cache/`.
  2. If cached, loads and returns the graph instantly (no network).
  3. If not cached, fetches from OpenStreetMap via osmnx, saves to
     the cache directory, and returns the graph.

Supports two modes:
  - **Place-name mode**: pass a geocodable string (e.g. "Kolkata, India").
  - **Point+radius mode**: pass a (lat, lon) tuple and a distance in metres.

References
----------
- osmnx docs: https://osmnx.readthedocs.io/en/stable/
- OSM Nominatim usage policy: https://operations.osmfoundation.org/policies/nominatim/
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Tuple, Union

import osmnx as ox

# ---------------------------------------------------------------------------
# Module-level config
# ---------------------------------------------------------------------------

# Resolve project root relative to this file: layer2/src/ -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = _PROJECT_ROOT / "data" / "graph_cache"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """
    Convert an arbitrary string into a safe, lowercase filename slug.

    Example
    -------
    >>> _slugify("Kolkata, West Bengal, India")
    'kolkata_west_bengal_india'
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)   # remove punctuation
    text = re.sub(r"[\s-]+", "_", text)     # collapse whitespace / hyphens
    return text


def _cache_path(slug: str) -> Path:
    """Return the full path to a cached .graphml file for a given slug."""
    return CACHE_DIR / f"{slug}.graphml"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_graph(
    place: Optional[str] = None,
    *,
    point: Optional[Tuple[float, float]] = None,
    dist: float = 2000,
    network_type: str = "drive",
) -> ox.graph.nx.MultiDiGraph:
    """
    Load a road-network graph, using a local cache when available.

    Parameters
    ----------
    place : str, optional
        A geocodable place name (e.g. "Kolkata, West Bengal, India").
        Mutually exclusive with `point`.
    point : tuple of (lat, lon), optional
        Centre point for a radius-based query.  Requires `dist`.
    dist : float, default 2000
        Radius in metres around `point`.  Ignored when `place` is used.
    network_type : str, default "drive"
        OSM network type -- one of "drive", "walk", "bike", "all", etc.

    Returns
    -------
    networkx.MultiDiGraph
        The road-network graph with node/edge attributes from OSM.

    Raises
    ------
    ValueError
        If neither `place` nor `point` is provided, or both are.
    """
    # --- Validate arguments ---
    if place and point:
        raise ValueError("Provide either `place` or `point`, not both.")
    if not place and not point:
        raise ValueError("Provide at least one of `place` or `point`.")

    # --- Build a cache key ---
    if place:
        slug = _slugify(place)
    else:
        # e.g. "pt_22.5726_88.3639_d2000_drive"
        slug = f"pt_{point[0]}_{point[1]}_d{int(dist)}_{network_type}"

    cached = _cache_path(slug)

    # --- Try the cache first ---
    if cached.exists():
        logger.info("Cache HIT  -> loading %s", cached.name)
        G = ox.load_graphml(filepath=cached)
        return G

    # --- Cache miss -- fetch from OSM ---
    logger.info("Cache MISS -> fetching from OpenStreetMap ...")

    if place:
        G = ox.graph_from_place(place, network_type=network_type)
    else:
        G = ox.graph_from_point(point, dist=dist, network_type=network_type)

    # Ensure the cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Persist to disk
    ox.save_graphml(G, filepath=cached)
    logger.info("Saved to cache -> %s", cached.name)

    return G
