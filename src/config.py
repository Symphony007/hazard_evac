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
    cache_path: str = "data/graph.graphml"


def load_config(path: str | Path) -> StudyAreaConfig:
    """Load a YAML config file and return a StudyAreaConfig dataclass.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    StudyAreaConfig
        Parsed configuration.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return StudyAreaConfig(**raw)

