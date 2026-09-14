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

The demo routes 20 evacuees across a 2km radius of central Kolkata with a 600m-radius hazard zone. The comparison table shows:

| Metric               | Naive | Congestion-Aware |
|----------------------|------:|------------------:|
| Agents routed        |    20 |                20 |
| Failed agents        |     0 |                 0 |
| Unique routes        |    20 |                20 |
| **Max edge load**    |   **8** |             **7** |
| Avg path length (m)  | 2,606 |             2,608 |
| Loaded edges         |   300 |               320 |

**Key takeaway:** Congestion-aware routing reduced the maximum edge load by 12%, spreading traffic more evenly across 320 edges (vs 300 for naive), at virtually the same average path length (+2m, or 0.08%). This demonstrates the core value proposition: better load distribution with negligible cost increase.

## What the rendered map shows

When you open `layer2/output/evacuation_multi_agent.html`:

- **Coloured polylines**: Each agent's route is drawn in a distinct colour (blue, green, purple, orange, etc.). You should see routes fanning out around the hazard zone.
- **Congestion heat layer**: Edges with traffic appear as thicker lines on a green-to-red scale. Green = light load, red = heavy load. You should see the heaviest congestion on edges near the safe zone destinations.
- **Red circle + red edges**: The hazard zone, identical to Layer 1. No routes should pass through it.
- **Green dots**: Evacuee origin points (scattered around the network).
- **Red dots**: Safe zone destinations (3 points at the graph periphery).
- **Dark red marker**: Hazard epicentre.
