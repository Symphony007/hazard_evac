"""Synthetic terrain generator and elevation-provider protocol.

The terrain assigns an elevation (metres) to every node in the road
graph.  The synthetic generator uses random Gaussian bumps; a real DEM
provider can be substituted later by implementing the same
:class:`ElevationProvider` protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import networkx as nx
import numpy as np


@runtime_checkable
class ElevationProvider(Protocol):
    """Callable that assigns an elevation (m) to every graph node.

    Any callable ``(G: nx.MultiDiGraph) -> dict[int, float]`` satisfies
    this protocol.
    """

    def __call__(self, G: nx.MultiDiGraph) -> dict[int, float]: ...


# ---------------------------------------------------------------------------
# Synthetic terrain
# ---------------------------------------------------------------------------

def generate_terrain(
    G: nx.MultiDiGraph,
    seed: int,
    n_bumps: int,
    range_m: float,
) -> dict[int, float]:
    """Generate a synthetic elevation field over the graph nodes.

    Uses *n_bumps* random 2-D Gaussian bumps in normalised
    ``[0, 1]²`` coordinate space (derived from node lat/lon).
    The result is min-max scaled to ``[0, range_m]``.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road graph whose nodes carry ``'x'`` (lon) and ``'y'`` (lat).
    seed : int
        RNG seed – same seed produces identical terrain.
    n_bumps : int
        Number of Gaussian bumps to superimpose.
    range_m : float
        Elevation range in metres (output spans ``[0, range_m]``).

    Returns
    -------
    dict[int, float]
        Mapping ``node_id -> elevation_m``.
    """
    rng = np.random.default_rng(seed)

    nodes = list(G.nodes)
    xs = np.array([G.nodes[n]["x"] for n in nodes], dtype=np.float64)
    ys = np.array([G.nodes[n]["y"] for n in nodes], dtype=np.float64)

    # Normalise to [0, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    xn = (xs - x_min) / (x_max - x_min) if x_max != x_min else np.zeros_like(xs)
    yn = (ys - y_min) / (y_max - y_min) if y_max != y_min else np.zeros_like(ys)

    elevation = np.zeros(len(nodes), dtype=np.float64)
    for _ in range(n_bumps):
        cx = rng.uniform(0, 1)
        cy = rng.uniform(0, 1)
        width = rng.uniform(0.15, 0.4)
        amplitude = rng.uniform(-1.0, 1.0)
        dist_sq = (xn - cx) ** 2 + (yn - cy) ** 2
        elevation += amplitude * np.exp(-dist_sq / (2.0 * width ** 2))

    # Min-max scale to [0, range_m]
    e_min, e_max = elevation.min(), elevation.max()
    if e_max > e_min:
        elevation = (elevation - e_min) / (e_max - e_min) * range_m
    else:
        elevation[:] = range_m / 2.0

    return {n: float(elevation[i]) for i, n in enumerate(nodes)}


def make_synthetic_provider(
    seed: int, n_bumps: int, range_m: float
) -> ElevationProvider:
    """Create a synthetic :class:`ElevationProvider` with fixed parameters.

    This is a convenience wrapper so that external code can pass an
    ``ElevationProvider`` without knowing the synthetic function's
    signature.
    """

    def _provider(G: nx.MultiDiGraph) -> dict[int, float]:
        return generate_terrain(G, seed, n_bumps, range_m)

    return _provider  # type: ignore[return-value]

