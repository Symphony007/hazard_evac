"""Flood hazard simulator.

Water starts at a set of source nodes and spreads along roads.  The
result is a deterministic (given seed) arrival-time field T(n) for every
node.  An edge closes at ``min(T(u), T(v))`` and degrades gradually
during a configurable ramp window before closure.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import osmnx as ox

from src.hazard.base import HazardModel
from src.hazard.terrain import generate_terrain

if TYPE_CHECKING:
    from src.config import HazardConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    R = 6_371_009.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# FloodHazard
# ---------------------------------------------------------------------------

class FloodHazard(HazardModel):
    """Flood hazard model: computes per-node arrival times and edge status.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road graph (must have ``length`` on edges and ``x``/``y`` on nodes).
    cfg : HazardConfig
        Hazard parameters loaded from YAML.
    elevation : dict[int, float] or None
        Pre-computed node elevations.  If *None*, synthetic terrain is
        generated from ``cfg.terrain_seed``.
    """

    def __init__(
        self,
        G: nx.MultiDiGraph,
        cfg: "HazardConfig",
        elevation: dict[int, float] | None = None,
    ) -> None:
        self._G = G
        self._cfg = cfg
        self._ramp_s: float = cfg.ramp_min * 60.0
        self._min_speed_factor: float = cfg.min_speed_factor

        # 1. Elevation ---------------------------------------------------
        if elevation is not None:
            self.elevation: dict[int, float] = dict(elevation)
        else:
            self.elevation = generate_terrain(
                G, cfg.terrain_seed, cfg.terrain_n_bumps, cfg.terrain_range_m
            )

        # 2. Select flood sources ----------------------------------------
        rng = np.random.default_rng(cfg.seed)
        self.sources: list[int] = self._select_sources(G, cfg, rng)

        # 3-4. Compute arrival times -------------------------------------
        self._node_times: dict[int, float] = self._compute_arrival_times(
            G, cfg, rng
        )

        # 5. Edge arrival times ------------------------------------------
        self._edge_times: dict[tuple[int, int, int], float] = {}
        for u, v, k in G.edges(keys=True):
            self._edge_times[(u, v, k)] = min(
                self._node_times[u], self._node_times[v]
            )

        # Lazy cache
        self._closure_schedule_cache: list[tuple[float, tuple[int, int, int]]] | None = None

    # -----------------------------------------------------------------
    # Source selection
    # -----------------------------------------------------------------

    def _select_sources(
        self,
        G: nx.MultiDiGraph,
        cfg: "HazardConfig",
        rng: np.random.Generator,
    ) -> list[int]:
        nodes = list(G.nodes)

        if cfg.source_selection == "explicit":
            sources: list[int] = []
            for lat, lon in cfg.explicit_sources:
                node = int(ox.distance.nearest_nodes(G, lon, lat))
                sources.append(node)
            return sources

        if cfg.source_selection == "lowest_elevation":
            ordered = sorted(nodes, key=lambda n: self.elevation[n])
        elif cfg.source_selection == "random":
            indices = rng.permutation(len(nodes))
            ordered = [nodes[int(i)] for i in indices]
        else:
            raise ValueError(f"Unknown source_selection: {cfg.source_selection!r}")

        sources = []
        for n in ordered:
            if len(sources) >= cfg.n_sources:
                break
            # Enforce minimum separation
            if all(
                _haversine_m(
                    G.nodes[n]["y"], G.nodes[n]["x"],
                    G.nodes[s]["y"], G.nodes[s]["x"],
                ) >= cfg.min_source_separation_m
                for s in sources
            ):
                sources.append(n)

        if not sources:
            raise RuntimeError(
                "No flood sources could be selected "
                f"(n_sources={cfg.n_sources}, "
                f"min_separation={cfg.min_source_separation_m} m)"
            )
        if len(sources) < cfg.n_sources:
            warnings.warn(
                f"Only {len(sources)}/{cfg.n_sources} sources found "
                f"with min separation {cfg.min_source_separation_m} m",
                stacklevel=2,
            )
        return sources

    # -----------------------------------------------------------------
    # Arrival-time computation
    # -----------------------------------------------------------------

    def _compute_arrival_times(
        self,
        G: nx.MultiDiGraph,
        cfg: "HazardConfig",
        rng: np.random.Generator,
    ) -> dict[int, float]:
        elev_vals = np.array([self.elevation[n] for n in G.nodes])
        e_ref = float(elev_vals.mean())
        e_range = float(elev_vals.max() - elev_vals.min())

        # -- Build undirected edge base costs --
        edge_costs: dict[tuple[int, int], float] = {}
        for u, v, k, data in G.edges(keys=True, data=True):
            pair = (min(u, v), max(u, v))
            e_edge = (self.elevation[u] + self.elevation[v]) / 2.0
            if e_range > 0:
                f = float(np.clip(
                    1.0 + cfg.terrain_gamma * (e_ref - e_edge) / e_range,
                    0.3, 2.0,
                ))
            else:
                f = 1.0
            base_cost = (data["length"] / (cfg.spread_speed_m_per_min * f)) * 60.0
            if pair not in edge_costs or base_cost < edge_costs[pair]:
                edge_costs[pair] = base_cost

        # -- Apply lognormal noise in sorted (deterministic) order --
        sorted_pairs = sorted(edge_costs.keys())
        noise = rng.lognormal(mean=0.0, sigma=cfg.noise_sigma, size=len(sorted_pairs))

        H = nx.Graph()
        H.add_nodes_from(G.nodes)
        for (a, b), n_val in zip(sorted_pairs, noise):
            H.add_edge(a, b, cost=edge_costs[(a, b)] * float(n_val))

        # -- Multi-source Dijkstra on H --
        lengths = dict(
            nx.multi_source_dijkstra_path_length(
                H, set(self.sources), weight="cost"
            )
        )

        node_times: dict[int, float] = {}
        for n in G.nodes:
            node_times[n] = lengths.get(n, np.inf)
        for s in self.sources:
            node_times[s] = 0.0

        return node_times

    # =================================================================
    # HazardModel interface
    # =================================================================

    def node_arrival_time(self, n: int) -> float:
        """Seconds until flood reaches node *n*; ``np.inf`` if never."""
        return self._node_times[n]

    def edge_arrival_time(self, u: int, v: int, k: int) -> float:
        """Seconds until flood reaches edge *(u, v, k)*."""
        return self._edge_times[(u, v, k)]

    def is_edge_closed(self, u: int, v: int, k: int, t: float) -> bool:
        arrival = self._edge_times[(u, v, k)]
        if not np.isfinite(arrival):
            return False
        return t >= arrival

    def edge_speed_factor(self, u: int, v: int, k: int, t: float) -> float:
        arrival = self._edge_times[(u, v, k)]
        if not np.isfinite(arrival):
            return 1.0
        if t >= arrival:
            return 0.0
        ramp_start = arrival - self._ramp_s
        if t <= ramp_start:
            return 1.0
        # Linear ramp from 1.0 → min_speed_factor
        progress = (t - ramp_start) / self._ramp_s
        return 1.0 - progress * (1.0 - self._min_speed_factor)

    def is_node_flooded(self, n: int, t: float) -> bool:
        arrival = self._node_times[n]
        if not np.isfinite(arrival):
            return False
        return t >= arrival

    def closed_edges(self, t: float) -> set[tuple[int, int, int]]:
        return {
            key for key, arrival in self._edge_times.items()
            if np.isfinite(arrival) and t >= arrival
        }

    def closure_schedule(self) -> list[tuple[float, tuple[int, int, int]]]:
        if self._closure_schedule_cache is None:
            entries = [
                (arrival, key)
                for key, arrival in self._edge_times.items()
                if np.isfinite(arrival)
            ]
            entries.sort()
            self._closure_schedule_cache = entries
        return list(self._closure_schedule_cache)

