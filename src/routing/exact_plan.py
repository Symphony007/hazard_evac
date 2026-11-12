"""ExactPlanRouter replaying precomputed optimal plans in the multi-agent simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import networkx as nx

from src.hazard.base import HazardModel
from src.routing.base import Evacuee, Route, Router, RoutingParams

if TYPE_CHECKING:
    from src.exact.decompose import AgentPlan


class ExactPlanRouter(Router):
    """Router that plays back precomputed exact plans from flow decomposition.

    Parameters
    ----------
    plans : dict[int, AgentPlan | None], optional
        Pre-assigned plans for agents indexed by agent ID.
        Agents with None or missing from the dict receive None (no_route).

    Notes
    -----
    Planned waiting (holdover arcs in the discrete time-expanded network) is not
    reproduced as explicit pausing in the simulator. Instead, agents move forward
    immediately onto their designated sequence of physical road edges and experience
    realistic continuous-time FIFO point-queueing at bottlenecks.

    Consequently, an exact plan can occasionally fail in the continuous-time
    simulator due to timing mismatches between discrete time steps and continuous
    hazard encroachment, or queue-induced entry delays (recorded as trapped).
    """

    name = "exact_plan"
    replans = False

    def __init__(self, plans: dict[int, AgentPlan | None] | None = None) -> None:
        self.plans: dict[int, AgentPlan | None] = dict(plans) if plans is not None else {}

    def prepare(
        self,
        G: nx.MultiDiGraph,
        hazard: HazardModel,
        agents: Sequence[Evacuee],
        exits: set[int] | Sequence[int],
        params: RoutingParams,
        traffic_cfg: Any = None,
        plans: dict[int, AgentPlan | None] | None = None,
    ) -> None:
        """Store or update pre-assigned plans prior to simulation."""
        if plans is not None:
            self.plans = dict(plans)

    def plan(
        self,
        G: nx.MultiDiGraph,
        hazard: HazardModel,
        evacuee: Evacuee,
        t_now_s: float,
        current_node: int | str,
        exits: set[int] | Sequence[int],
        params: RoutingParams,
        traffic: Any = None,
    ) -> Route | None:
        """Return the precomputed route for the agent, or None if unsaved/no route."""
        plan = self.plans.get(evacuee.id)
        if plan is None:
            return None

        # Compute total physical length in metres
        length_m = 0.0
        for u, v, k in plan.edges:
            if G.has_edge(u, v, key=k):
                length_m += float(G[u][v][k].get("length", 0.0))
            else:
                data = G.get_edge_data(u, v, default={})
                if k in data:
                    length_m += float(data[k].get("length", 0.0))
                elif data:
                    first_k = next(iter(data))
                    length_m += float(data[first_k].get("length", 0.0))

        return Route(
            edges=list(plan.edges),
            nodes=list(plan.nodes),
            length_m=length_m,
            planned_arrival_s=plan.planned_exit_s,
        )

