"""
Evacuation Routing Project — Requirement Validation Script
Run this on your own machine (needs real internet access) to confirm
every tool/API/library actually works before committing to the project.

Install first:
    pip install osmnx networkx folium requests matplotlib

Run:
    python validate_requirements.py
"""

import sys

results = {}

def check(name, fn):
    print(f"\n{'='*50}\nTesting: {name}\n{'='*50}")
    try:
        fn()
        results[name] = "PASS"
        print(f"[PASS] {name}")
    except Exception as e:
        results[name] = f"FAIL: {e}"
        print(f"[FAIL] {name} — {e}")


# ---------------------------------------------------------
# 1. Core libraries import correctly
# ---------------------------------------------------------
def test_imports():
    import osmnx as ox
    import networkx as nx
    import folium
    import requests
    print("osmnx:", ox.__version__)
    print("networkx:", nx.__version__)
    print("folium:", folium.__version__)

check("Library imports", test_imports)


# ---------------------------------------------------------
# 2. Pull a REAL road network from OpenStreetMap
#    (This is the most important test — confirms osmnx + OSM
#    access works, since this is your core data source)
# ---------------------------------------------------------
def test_osm_pull():
    import osmnx as ox
    # Small, fast-to-download area — swap for your target city later
    G = ox.graph_from_place("Howrah, West Bengal, India", network_type="drive")
    print(f"Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
    assert len(G.nodes) > 0, "Graph came back empty"

check("OpenStreetMap live pull (osmnx)", test_osm_pull)


# ---------------------------------------------------------
# 3. Run Dijkstra / shortest path on the REAL pulled graph
# ---------------------------------------------------------
def test_routing_on_real_graph():
    import osmnx as ox
    import networkx as nx
    G = ox.graph_from_place("Howrah, West Bengal, India", network_type="drive")
    nodes = list(G.nodes)
    orig, dest = nodes[0], nodes[len(nodes)//2]
    path = nx.shortest_path(G, orig, dest, weight="length")
    print(f"Path found with {len(path)} nodes")
    assert len(path) > 1

check("Routing (Dijkstra/A*) on real OSM graph", test_routing_on_real_graph)


# ---------------------------------------------------------
# 4. Dynamic re-routing when an edge becomes "unsafe"
#    (the actual core mechanism of the project)
# ---------------------------------------------------------
def test_dynamic_reroute():
    import osmnx as ox
    import networkx as nx
    G = ox.graph_from_place("Howrah, West Bengal, India", network_type="drive")
    nodes = list(G.nodes)
    orig, dest = nodes[0], nodes[len(nodes)//2]

    path1 = nx.shortest_path(G, orig, dest, weight="length")

    # Simulate hazard: block an edge on the path
    u, v = path1[3], path1[4]
    original_weight = G.edges[u, v, 0]["length"]
    G.edges[u, v, 0]["length"] = float("inf")

    path2 = nx.shortest_path(G, orig, dest, weight="length")
    print("Path changed after simulated hazard:", path1 != path2)
    assert path1 != path2, "Rerouting did not change the path — check logic"

check("Dynamic re-routing on hazard (core mechanism)", test_dynamic_reroute)


# ---------------------------------------------------------
# 5. USGS earthquake API access (real hazard reference data)
# ---------------------------------------------------------
def test_usgs_api():
    import requests
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {"format": "geojson", "limit": 5, "orderby": "time"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    print(f"Fetched {len(data['features'])} recent earthquake records")
    assert len(data["features"]) > 0

check("USGS Earthquake API", test_usgs_api)


# ---------------------------------------------------------
# 6. Folium map visualization (renders route + hazard)
# ---------------------------------------------------------
def test_folium_render():
    import folium
    m = folium.Map(location=[22.5958, 88.2636], zoom_start=13)  # Howrah coords
    folium.PolyLine([[22.59, 88.26], [22.60, 88.27]], color="red").add_to(m)
    m.save("test_map_output.html")
    print("Map saved to test_map_output.html — open it in a browser to confirm it renders")

check("Folium visualization output", test_folium_render)


# ---------------------------------------------------------
# 7. Kaggle dataset access check (for flood/wildfire reference data)
#    Requires kaggle.json API credentials configured
# ---------------------------------------------------------
def test_kaggle_access():
    import os
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        raise Exception(
            "kaggle.json not found — go to kaggle.com/settings, "
            "create an API token, place it at ~/.kaggle/kaggle.json"
        )
    print("Kaggle credentials found locally")

check("Kaggle API credentials (optional — only if using Kaggle datasets)", test_kaggle_access)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
print("\n\n" + "="*50)
print("SUMMARY")
print("="*50)
for name, result in results.items():
    print(f"{result:8s} | {name}")

failed = [n for n, r in results.items() if r != "PASS"]
if failed:
    print(f"\n{len(failed)} check(s) failed — resolve these before committing fully.")
    sys.exit(1)
else:
    print("\nAll checks passed. Core requirements are confirmed working.")