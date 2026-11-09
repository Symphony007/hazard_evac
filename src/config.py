"""Configuration loader for YAML config files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StudyAreaConfig:
    """Configuration for the study area road network."""

    name: str = "tollygunge_test"
    center_lat: float = 22.4985
    center_lon: float = 88.3450
    dist_m: int = 1500
    network_type: str = "drive"
    default_lanes: int = 1
    capacity_per_lane_veh_per_km: int = 40
    min_edge_capacity: int = 2
    flow_rate_veh_per_min_per_lane: int = 12
    cache_path: str = "data/graph.graphml"


@dataclass
class HazardConfig:
    """Configuration for the flood hazard simulator."""

    terrain_seed: int = 7
    terrain_n_bumps: int = 6
    terrain_range_m: float = 3.0
    seed: int = 42
    n_sources: int = 2
    source_selection: str = "lowest_elevation"
    explicit_sources: list[list[float]] = field(default_factory=list)
    min_source_separation_m: float = 600.0
    spread_speed_m_per_min: float = 30.0
    terrain_gamma: float = 1.0
    noise_sigma: float = 0.15
    ramp_min: float = 3.0
    min_speed_factor: float = 0.2
    horizon_min: float = 60.0


@dataclass
class ScenarioConfig:
    """Configuration for evacuation scenario generation and experiment sweep."""

    scenario_seed: int = 11
    n_exits: int = 6
    exit_min_dist_from_center_m: float = 1200.0
    n_evacuees: int = 300
    depart_window_min: float = 20.0
    vehicle_speed_scale: float = 0.5
    sf_floor: float = 0.05
    flood_speed_sweep_m_per_min: list[float] = field(
        default_factory=lambda: [30.0, 60.0, 120.0, 240.0]
    )
    example_flood_speed_m_per_min: float = 120.0


def load_config(path: str | Path, config_cls: type = StudyAreaConfig):
    """Load a YAML config file into a dataclass.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.
    config_cls : type
        Dataclass to instantiate (default: StudyAreaConfig).

    Returns
    -------
    dataclass instance
        Parsed configuration.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return config_cls(**raw)
