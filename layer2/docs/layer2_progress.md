# Layer 2 Progress: Multi-Agent Congestion-Aware Evacuation Routing

## Overview

Layer 2 extends the Layer 1 baseline from a single-agent system to a **multi-agent** evacuation routing system. Where Layer 1 routes one evacuee around a spreading hazard, Layer 2 handles **20+ simultaneous evacuees** competing for the same road network. The key innovation is a **congestion model** that makes roads progressively more expensive as more evacuees are assigned to them, causing the routing algorithm to naturally diversify paths across the network rather than funneling everyone onto one shortest route.

## What was built

### New modules

1. **`congestion.py` -- BPR Volume-Delay Model**

   Implements the Bureau of Public Roads (BPR) cost function, the standard traffic engineering model used worldwide. For each road segment, the traversal cost is computed as:

   ```
   cost = base_length * (1 + 0.15 * (volume / capacity)^4)
   ```

   At low traffic (volume << capacity), cost is essentially the road's physical length. As traffic approaches capacity, the cost curve rises sharply -- a road carrying twice its capacity is nearly 4x more expensive to traverse. This creates a powerful incentive for the router to distribute agents across alternative paths.

   Edge capacity is derived from OpenStreetMap's `lanes` attribute when available (e.g. a 4-lane road has higher capacity than a 1-lane road). When lane data is missing, a sensible default is used.

2. **`multi_agent_router.py` -- Sequential Congestion-Aware Routing**

   Routes agents one at a time in sequence. After each agent is routed, their chosen path is registered with the congestion model, updating the costs on every edge they use. The *next* agent then sees higher costs on those edges and is naturally pushed toward alternative routes. This is a form of greedy marginal-cost routing.

   Two functions are provided:
   - `route_all_agents()` -- congestion-aware (the main algorithm)
   - `route_all_agents_naive()` -- baseline without congestion, for comparison

   Both return a `RoutingResult` dataclass with per-agent routes and aggregate statistics (average path length, max edge load, number of unique routes, hazard reroute count).

3. **`visualize.py` -- Multi-Agent Map Rendering**

   Extended from Layer 1 to render multiple agent routes in distinct colours (10-colour perceptually-distinct palette, cycling for >10 agents). Road segments are also coloured by congestion load on a green-yellow-red scale, with line thickness proportional to traffic volume. The hazard zone overlay is preserved from Layer 1.

### Preserved modules (copied from Layer 1)

- `graph.py` -- road network loading with GraphML caching (unchanged)
- `hazard.py` -- radius-based hazard spread simulation (unchanged)
- `router.py` -- Dijkstra shortest path (extended with a configurable `weight` parameter so it can route on either `"length"` or `"congestion_cost"`)

## What the test suite verifies

The test suite (`tests/test_layer2.py`) contains 12 tests across 5 categories:

1. **BPR cost function correctness** (3 tests): Cost increases with load; cost grows non-linearly (steeper at higher loads, confirming beta=4); zero load produces base length exactly.

2. **Load tracking** (5 tests): `add_route` increments edge loads; `remove_route` decrements them; loads never go negative; `max_load` property is correct; statistics are internally consistent.

3. **Multi-agent route diversity** (2 tests): When 10 agents share the same origin-destination pair, congestion-aware routing produces *more than 1 unique route* (proving diversification works); naive routing produces exactly 1 (all agents get the same shortest path).

4. **Hazard avoidance under congestion** (1 test): Even with congestion active, no agent's route uses a hazardous edge. Safety always takes precedence over congestion optimization.

5. **Congestion-aware beats naive** (1 test): On 15 agents with the same OD pair, congestion-aware routing achieves a lower (or equal) maximum edge load than naive routing.

## Demo results (`main.py`)

### Evacuee Scenario & Scale Adjustment

To create a realistic emergency evacuation scenario that matches Layer 1's geographic scale:
- The road network graph covers a 2 km radius query (4 km diameter bounding box) around central Kolkata.
- **Constrained Origin Sampling**: Rather than scattering evacuees across the entire 4 km bounding box, all 20 evacuee origin nodes are sampled within a **1,000 m radius of the hazard epicenter** (outside the 600 m active hazard zone). This realistically models a crowd of people in the danger area needing to escape outward.
- **Periphery Safe Zones**: 3 safe-zone destinations are positioned at the network periphery, forcing all evacuees to travel away from the center to safety.
- **Tightened Map Framing**: The interactive map (`layer2/output/evacuation_multi_agent.html`) centers directly on the hazard epicenter and uses `zoom_start=15` (matching Layer 1) rather than a wide zoom.

### Comparison Results

Under this realistic bottleneck scenario, the value of congestion-aware routing is even more pronounced:

| Metric               | Naive | Congestion-Aware | Delta / Impact |
|----------------------|------:|------------------:|---------------:|
| Agents routed        |    20 |                20 | All routed (0 failed) |
| Failed agents        |     0 |                 0 | 100% success rate |
| Unique routes        |    20 |                20 | Full route diversity |
| **Max edge load**    |  **12** |             **8** | **-33% peak load reduction** |
| Avg path length (m)  | 3,089 |             3,180 | +2.9% minor detour |
| Loaded edges         |   229 |               288 | +59 alternative streets used |
| Total edge volume    |   660 |               717 | Distributed traffic |

**Key takeaway:** When 20 evacuees all flee the same danger zone at once, naive shortest-path routing funnels everyone down the exact same primary arterial roads, causing peak edge load to spike to **12 vehicles**. Congestion-aware routing dynamically diverts later evacuees onto secondary streets, **slashing the peak edge bottleneck by 33% (12 → 8)** and spreading traffic across 59 additional road segments, with only a negligible 2.9% (+91 m) average detour.

## What the rendered map shows

When you open `layer2/output/evacuation_multi_agent.html`:

- **High-contrast coloured polylines**: Each agent's route is drawn using a curated, colorblind-friendly, high-contrast palette (royal blue, purple, pumpkin orange, dark magenta, pine teal, electric indigo, warm brown, cobalt, goldenrod, slate navy). Pale colors (such as pale gold and light cyan) and green shades have been removed to ensure crisp visibility against OpenStreetMap's light background.
- **Prominent route line weight**: Routes are rendered at weight 4 and 85% opacity, ensuring they are immediately visible as they emerge from origin markers.
- **Congestion heat layer**: Road segments carry color and thickness proportional to traffic load (green for light volume through yellow to red for bottlenecks).
- **Hazard zone**: Translucent red circular perimeter (600 m radius at $t=4$) with internal hazardous edges highlighted in red.
- **Green origin markers**: 20 evacuee start points clustered within 1,000 m of the hazard epicenter, bordered in dark green.
- **Red destination markers**: 3 safe-zone destinations at the outer network boundary.
- **Dark red hazard marker**: Hazard epicenter at the core.
- **Failure logging**: If an agent cannot find a path due to complete network blockage or hazard cutoff, `multi_agent_router.py` explicitly catches and logs the agent ID, origin, destination, and exact failure reason instead of silently dropping it.
