# Task 2 Audit Report: Flow Capacity & Flood Hazard Simulator

## 1. Verdict: Task 2 COMPLETE

**Justification:**
All components for Part A (flow capacity) and Part B (flood hazard simulator) were authored before the previous session was interrupted, but verification (rebuilding the graph cache, running the demo, and executing the test suite) had not yet been executed in that session. During this audit:
- `python -m compileall src scripts tests` passed with 0 errors across all 15 Python files.
- `python scripts/build_graph.py --config configs/study_area.yaml --rebuild` successfully rebuilt and cached the graph with `lanes_n` and `flow_capacity`.
- `python scripts/run_hazard_demo.py --graph-config configs/study_area.yaml --hazard-config configs/hazard.yaml` executed cleanly, printing expected hazard summary statistics and generating all 3 requested plots.
- `pytest -q` ran 15 tests (5 graph tests + 10 hazard tests) and all 15 passed with 0 failures.
- All four output images (`graph_overview.png`, `terrain.png`, `hazard_snapshots.png`, `hazard_curve.png`) exist, are non-empty, and valid.

---

## 2. Checklist Table

### PART A: Flow Capacity

| Item | Requirement | Status | Evidence |
|---|---|---|---|
| A.1 | `configs/study_area.yaml` has `flow_rate_veh_per_min_per_lane: 12` | **done** | `configs/study_area.yaml:9` |
| A.2 | `src/config.py` `StudyAreaConfig` has `flow_rate_veh_per_min_per_lane: int = 12` | **done** | `src/config.py:24` |
| A.3.a | Store parsed lane count as `lanes_n` (int) on every edge | **done** | `src/graph/loader.py:61-64` (`_add_lanes_n`) |
| A.3.b | Add `flow_capacity` (float, veh/min) = `lanes_n * flow_rate_veh_per_min_per_lane` | **done** | `src/graph/loader.py:79-87` (`_add_flow_capacity`) |
| A.3.c | Retain existing storage `capacity` attribute unchanged | **done** | `src/graph/loader.py:67-76` (`_add_capacity`) |
| A.3.d | `load_graph` casts `lanes_n` to int and `flow_capacity` to float | **done** | `src/graph/loader.py:164-173` |
| A.4 | `scripts/build_graph.py` adds `--rebuild` flag and prints flow_capacity stats | **done** | `scripts/build_graph.py:36-40, 48-54, 77-80` |
| A.5 | `tests/test_graph.py` tests `flow_capacity > 0` and equals `lanes_n * flow_rate` | **done** | `tests/test_graph.py:76-90` (`test_flow_capacity`) |
| A.6 | Graph cache rebuilt via `--rebuild` | **done** | Command output: `data/graph.graphml` written (2,601,559 bytes) |

### PART B: Flood Hazard Simulator

| Item | Requirement | Status | Evidence |
|---|---|---|---|
| B.1 | `configs/hazard.yaml` with all 14 parameters | **done** | `configs/hazard.yaml:1-14` |
| B.2 | `src/hazard/__init__.py` exporting public symbols | **done** | Exports `HazardModel`, `FloodHazard`, `ElevationProvider`, `generate_terrain` |
| B.3 | `src/hazard/base.py` abstract `HazardModel` declaring 7 required methods | **done** | `src/hazard/base.py:13-51` |
| B.4.a | `src/hazard/terrain.py` `generate_terrain` normalized to `[0,1]^2`, Gaussian bumps, min-max scaled to `[0, range_m]` | **done** | `src/hazard/terrain.py:32-89` |
| B.4.b | `ElevationProvider` Protocol defined | **done** | `src/hazard/terrain.py:18-25` |
| B.5.a | `FloodHazard` accepts `elevation` or generates via `terrain_seed` | **done** | `src/hazard/flood.py:70-75` |
| B.5.b | Source selection: `lowest_elevation`, `random`, `explicit` with `min_source_separation_m` | **done** | `src/hazard/flood.py:100-149` (`_select_sources`) |
| B.5.c | Undirected graph H with speed factor `f`, deterministic sorted lognormal noise | **done** | `src/hazard/flood.py:163-189` |
| B.5.d | Node arrival times via `nx.multi_source_dijkstra_path_length` | **done** | `src/hazard/flood.py:191-201` |
| B.5.e | Edge arrival times precomputed as `min(T(u), T(v))` | **done** | `src/hazard/flood.py:88-91` |
| B.5.f | All 7 `HazardModel` methods implemented with fast dict lookups | **done** | `src/hazard/flood.py:209-257` |
| B.6.a | `scripts/run_hazard_demo.py` CLI `--graph-config` and `--hazard-config` | **done** | `scripts/run_hazard_demo.py:204-213` |
| B.6.b | Prints source nodes, first closure time, % road length closed at milestones | **done** | `scripts/run_hazard_demo.py:225-260` |
| B.6.c | Saves `outputs/hazard_snapshots.png` (2x3 grid, LineCollection, custom colors) | **done** | `outputs/hazard_snapshots.png` (1,369,660 bytes) |
| B.6.d | Saves `outputs/hazard_curve.png` (% closed & % degraded-or-closed vs time) | **done** | `outputs/hazard_curve.png` (55,959 bytes) |
| B.6.e | Saves `outputs/terrain.png` (node elevations colored) | **done** | `outputs/terrain.png` (471,752 bytes) |
| B.7 | `tests/test_hazard.py` with all 10 unit test cases passing | **done** | `tests/test_hazard.py:53-202` (10 tests, all pass) |

---

## 3. Files Changed or Added Since Task 1

### Modified Existing Files (5)
- `configs/study_area.yaml` (added `flow_rate_veh_per_min_per_lane: 12`)
- `src/config.py` (added `flow_rate_veh_per_min_per_lane` to `StudyAreaConfig`, added `HazardConfig`, updated `load_config`)
- `src/graph/loader.py` (added `_add_lanes_n`, `_add_flow_capacity`, updated `build_graph` and `load_graph` numeric casting)
- `scripts/build_graph.py` (added `--rebuild` CLI flag, added flow_capacity statistics calculation and printout)
- `tests/test_graph.py` (added `test_flow_capacity`)

### New Files Created (6)
- `configs/hazard.yaml` (hazard parameters)
- `src/hazard/__init__.py` (package exports)
- `src/hazard/base.py` (`HazardModel` abstract interface)
- `src/hazard/terrain.py` (`generate_terrain`, `ElevationProvider` protocol, `make_synthetic_provider`)
- `src/hazard/flood.py` (`FloodHazard` class)
- `scripts/run_hazard_demo.py` (demo script for hazard simulation and visualization)
- `tests/test_hazard.py` (10 unit tests for hazard mechanics)

---

## 4. Broken, Half-Written, or Suspicious Items

1. **Direct `pytest` executable on PATH (Windows)**:
   - When calling bare `pytest` in PowerShell, Windows threw `CommandNotFoundException` because Python user-site scripts (`C:\Users\Deepmalya\AppData\Roaming\Python\Python314\Scripts`) were not in system `PATH`.
   - Running via `python -m pytest -q` or temporarily prepending the scripts path works without issue.
2. **Pre-existing cache vs rebuild**:
   - The file `data/graph.graphml` was originally written during Task 1 and lacked `flow_capacity` and `lanes_n`. Running `python scripts/build_graph.py --config configs/study_area.yaml --rebuild` successfully updated the cache.
3. **No syntax errors, placeholders, or incomplete blocks**:
   - Verified via `compileall` and regex scan. No `TODO`, `FIXME`, `NotImplementedError`, or incomplete functions exist.

---

## 5. Verification Commands and Exact Output

### a. `python scripts/build_graph.py --config configs/study_area.yaml --rebuild`
```
Study area : tollygunge_test
Center     : (22.4985, 88.345)
Radius     : 1500 m

[build] Rebuilding graph (ignoring cache) …
[build] Graph saved to D:\Side project\Dev folders\hazard_evac\data\graph.graphml
==================================================
GRAPH STATISTICS
==================================================
Nodes             : 1958
Edges             : 4813
Total road length : 282.33 km
Capacity  mean    : 2.9
Capacity  median  : 2.0
Travel-time min   : 0.19 s
Travel-time max   : 53.07 s
Flow-cap   mean   : 12.3 veh/min
Flow-cap   median : 12.0 veh/min
Flow-cap   min    : 12.0 veh/min
Flow-cap   max    : 48.0 veh/min
==================================================

Plot saved to D:\Side project\Dev folders\hazard_evac\outputs\graph_overview.png
```

### b. `python scripts/run_hazard_demo.py --graph-config configs/study_area.yaml --hazard-config configs/hazard.yaml`
```
[loader] Loading cached graph from data\graph.graphml
Building flood hazard …

=======================================================
FLOOD HAZARD SUMMARY
=======================================================

Source nodes (2):
  Node 9567745934  lat=22.4891  lon=88.3372  elev=0.00 m
  Node 7222827111  lat=22.4934  lon=88.3411  elev=0.28 m

First edge closure   : 0.00 min

Road length closed:
  t = 10 min :   4.5%
  t = 20 min :  13.7%
  t = 30 min :  22.1%
  t = 45 min :  37.7%
  t = 60 min :  60.2%
=======================================================

Snapshots saved to D:\Side project\Dev folders\hazard_evac\outputs\hazard_snapshots.png
Hazard curve saved to D:\Side project\Dev folders\hazard_evac\outputs\hazard_curve.png
Terrain plot saved to D:\Side project\Dev folders\hazard_evac\outputs\terrain.png
```

### c. `python -m pytest -q`
```
...............                                                          [100%]
============================== warnings summary ===============================
C:\Users\Deepmalya\AppData\Roaming\Python\Python314\site-packages\anyio\pytest_plugin.py:14
  C:\Users\Deepmalya\AppData\Roaming\Python\Python314\site-packages\anyio\pytest_plugin.py:14: PytestRemovedIn10Warning: _pytest.python.CallSpec2 has been renamed to CallSpec.
  The CallSpec2 alias will be removed in pytest 10.
  Update imports to use CallSpec instead.
  See https://docs.pytest.org/en/stable/deprecations.html#callspec2-renamed
    from _pytest.python import CallSpec2

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
15 passed, 1 warning in 10.77s
```
Failing tests: **None (0 failed)**.

---

## 6. Output Files Verification

| File | Size (bytes) | Timestamp | Status |
|---|---|---|---|
| `outputs/graph_overview.png` | 316,384 | 2026-11-10 00:34:17 | Valid, non-empty |
| `outputs/hazard_curve.png` | 55,959 | 2026-11-10 00:34:32 | Valid, non-empty |
| `outputs/hazard_snapshots.png` | 1,369,660 | 2026-11-10 00:34:32 | Valid, non-empty |
| `outputs/terrain.png` | 471,752 | 2026-11-10 00:34:33 | Valid, non-empty |

---

## 7. Remaining Work for Task 2

**Nothing remains.** Task 2 is complete.
The project is ready to advance to **Task 3: Static vs. Time-Dependent Dijkstra**.

---

## 8. Assumptions Made During Audit

1. **Python invocation for pytest**: Invoked via `python -m pytest -q` (or with user-site scripts added to `$env:PATH`) because Python was installed with `--user` on Windows, leaving `pytest.exe` outside the default system PATH.
2. **Rebuilding cache during verification**: To ensure step 3 verification ran on genuine data rather than a stale cache from Task 1, step 3.a was executed first to refresh `data/graph.graphml`.
3. **No source code modification constraint**: Strictly followed. No source files under `src/`, `configs/`, `scripts/`, or `tests/` were touched.

