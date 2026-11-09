"""Base routing definitions and abstract Router interface.

All routing methods (static, reactive, time-dependent, CCRP, RL) operate
behind the Router interface defined here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import networkx as nx
    from src.hazard.base import HazardModel


@dataclass(frozen=True)
class Evacuee:
    """An evacuating agent.

    Parameters
    ----------
    id : int
        Unique agent identifier.
    origin : int | str
        Node ID where the evacuee begins evacuation.
    depart_time_s : float
        Departure time in seconds from simulation start.
    """

    id: int
    origin: int | str
    depart_time_s: float


@dataclass
class Route:
    """A planned evacuation path from origin/current node to an exit.

    Parameters
    ----------
    edges : list[tuple[int | str, int | str, int]]
        Sequence of MultiDiGraph edge keys (u, v, k).
    nodes : list[int | str]
        Sequence of nodes traversed, including start and exit.
    length_m : float
        Total physical road length of the route in metres.
    planned_arrival_s : float | None
        Estimated or planned arrival time at the exit in seconds.
    """

    edges: list[tuple[int | str, int | str, int]]
    nodes: list[int | str]
    length_m: float
    planned_arrival_s: float | None = None


@dataclass(frozen=True)
class RoutingParams:
    """Parameters governing edge traversal time computation.

    Parameters
    ----------
    vehicle_speed_scale : float
        Multiplier on edge free-flow speed (e.g. 0.5 for urban traffic/crawl).
    sf_floor : float
        Minimum hazard speed factor to prevent division by zero or infinite
        traversal time on degraded edges prior to closure (default: 0.05).
    """

    vehicle_speed_scale: float = 1.0
    sf_floor: float = 0.05


class Router(ABC):
    """Abstract router interface.

    Attributes
    ----------
    name : str
        Human-readable name of the routing method.
    replans : bool
        Whether this router triggers dynamic replanning during simulation
        upon detecting closed downstream edges.

    Notes
    -----
    In Task 5 (multi-agent routing with edge capacities), this interface
    will gain a `plan_batch(G, hazard, evacuees, t_now_s, states, exits, params)`
    method with a default implementation that loops over agents.
    """

    name: str
    replans: bool = False

    @abstractmethod
    def plan(
        self,
        G: "nx.MultiDiGraph",
        hazard: "HazardModel",
        evacuee: Evacuee,
        t_now_s: float,
        current_node: int,
        exits: set[int] | Sequence[int],
        params: RoutingParams,
    ) -> Route | None:
        """Compute a route from *current_node* at *t_now_s* to any exit.

        Parameters
        ----------
        G : nx.MultiDiGraph
            Road network graph.
        hazard : HazardModel
            Hazard model providing closure and speed degradation queries.
        evacuee : Evacuee
            Agent metadata (id, original origin, departure time).
        t_now_s : float
            Current simulation/departure time in seconds.
        current_node : int
            Current location of the agent.
        exits : set[int] or Sequence[int]
            Set or collection of safe exit node IDs.
        params : RoutingParams
            Speed scaling and floor parameters.

        Returns
        -------
        Route or None
            Feasible earliest-arrival route to an exit, or None if no route
            exists.
        """
