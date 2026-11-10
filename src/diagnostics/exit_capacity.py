"""Exit bottleneck capacity diagnostics.

Calculates the aggregate flow capacity of incoming roads directly feeding
the designated boundary exits, and computes theoretical lower bounds on
evacuation duration based on maximum exit throughput.

NOTE: This is a simple access-capacity bottleneck bound (sum of flow
capacities of edges directly entering exit nodes), not a proven
multi-commodity min-cut.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd

if TYPE_CHECKING:
    import networkx as nx


def exit_capacity_report(
    G: "nx.MultiDiGraph",
    exits: set[int] | Sequence[int],
    demand_levels: Sequence[int],
    ceiling_pct: float,
) -> dict[str, Any]:
    """Calculate aggregate exit capacity and throughput lower bounds.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road graph with edge attribute 'flow_capacity' (veh/min).
    exits : set[int] or Sequence[int]
        Target exit node IDs.
    demand_levels : Sequence[int]
        Demand levels (number of evacuees) to evaluate.
    ceiling_pct : float
        Percentage of agents savable under single-agent time-dependent routing.

    Returns
    -------
    dict[str, Any]
        Dictionary with per-exit capacity, total exit capacity, and lower bounds.
    """
    exits_list = sorted(list(set(exits)))
    per_exit: dict[int, dict[str, Any]] = {}
    total_capacity = 0.0

    for e in exits_list:
        in_edges = list(G.in_edges(e, keys=True, data=True))
        cap = sum(float(d.get("flow_capacity", 0.0)) for _, _, _, d in in_edges)
        per_exit[int(e)] = {
            "flow_capacity_veh_per_min": cap,
            "n_incoming_edges": len(in_edges),
        }
        total_capacity += cap

    demand_bounds: dict[int, dict[str, float]] = {}
    for n in sorted(demand_levels):
        n_savable = float(n) * (float(ceiling_pct) / 100.0)
        t_min = (n_savable / max(total_capacity, 1e-6)) if total_capacity > 0 else float("inf")
        demand_bounds[int(n)] = {
            "demand": int(n),
            "n_savable": n_savable,
            "t_min_minutes": t_min,
        }

    return {
        "ceiling_pct": float(ceiling_pct),
        "total_exit_capacity_veh_per_min": total_capacity,
        "n_exits": len(exits_list),
        "per_exit": per_exit,
        "demand_bounds": demand_bounds,
        "note": "Simple access-capacity bound (sum of capacities of edges entering exits), not a proven min-cut.",
    }


def save_exit_capacity_csv(
    report: dict[str, Any],
    out_path: str | Path,
) -> pd.DataFrame:
    """Save the throughput lower bounds table to CSV.

    Parameters
    ----------
    report : dict[str, Any]
        Output of exit_capacity_report.
    out_path : str or Path
        Target CSV file path.

    Returns
    -------
    pd.DataFrame
        DataFrame of throughput lower bounds.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    c_exit = report["total_exit_capacity_veh_per_min"]
    ceil = report["ceiling_pct"]
    for demand, b in report["demand_bounds"].items():
        rows.append(
            {
                "demand": int(demand),
                "ceiling_pct": ceil,
                "n_savable": b["n_savable"],
                "total_exit_flow_capacity_veh_per_min": c_exit,
                "t_min_minutes": b["t_min_minutes"],
                "bound_type": "access_capacity_throughput_bound",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df

