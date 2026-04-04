import networkx as nx
import pickle
import json
import os
import glob

# Paths
graphs_dir = 'outputs/graphs'
output_json = 'web_app/data/network_data.json'

def generate_network_data():
    network_data = {}
    
    # We will process years 2010-2016 for deepcnl graphs
    for year in range(2010, 2017):
        pkl_path = os.path.join(graphs_dir, f'deepcnl_graph_{year}.pkl')
        if not os.path.exists(pkl_path):
            continue
            
        with open(pkl_path, 'rb') as f:
            g = pickle.load(f)
            
        nodes_list = []
        links_list = []
        
        # Add links
        for u, v, data in g.edges(data=True):
            weight = data.get('weight', 0.5)
            links_list.append({
                "source": u,
                "target": v,
                "weight": weight
            })
            
        # Add nodes
        for node in g.nodes():
            degree = g.degree(node)
            
            # Find top connections
            neighbors = list(g[node].items())
            # neighbors is a list of (neighbor_id, edge_data_dict)
            # sort by edge weight
            neighbors.sort(key=lambda x: x[1].get('weight', 0), reverse=True)
            
            top_connections = []
            for neighbor_id, edge_data in neighbors[:5]: # Top 5
                top_connections.append({
                    "ticker": neighbor_id,
                    "weight": edge_data.get('weight', 0.5)
                })
                
            nodes_list.append({
                "id": str(node),
                "name": str(node), # Just use ticker as name
                "degree": degree,
                "market_cap": degree * 1000, # Fake market cap for bubble sizes in UI
                "top_connections": top_connections
            })
            
        # Optional: We can trim to top N nodes to avoid overwhelming the frontend D3 visualization
        # The hardcoded data typically had ~50 nodes or 45 links
        # The actual graphs have 100-200 nodes and ~220 edges, which should be fine for D3.
            
        network_data[str(year)] = {
            "nodes": nodes_list,
            "links": links_list
        }
        print(f"Year {year}: {len(nodes_list)} nodes, {len(links_list)} links")
        
    with open(output_json, 'w') as f:
        json.dump(network_data, f, indent=2)
    print(f"Updated {output_json} successfully with actual pkl data.")

if __name__ == '__main__':
    generate_network_data()
