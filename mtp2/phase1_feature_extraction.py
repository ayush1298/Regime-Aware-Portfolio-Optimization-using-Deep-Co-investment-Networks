import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import entropy
import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtp2.utils import load_graph, get_lcc

YEARS = list(range(2010, 2017))  # 2010 to 2016 inclusive

def extract_features(g):
    """Extract 5 topology features from a networkx graph."""
    if g.number_of_nodes() == 0:
        return [0, 0, 0, 0, 0]
    
    # Feature 1: density
    density = nx.density(g)
    
    # Feature 2: average clustering
    avg_clustering = nx.average_clustering(g)
    
    # Feature 3: avg path length on LCC
    lcc = get_lcc(g)
    avg_path_length = nx.average_shortest_path_length(lcc) if lcc.number_of_nodes() > 1 else 0
    
    # Feature 4: hub concentration (top-10 degree / total degree)
    degrees = dict(g.degree())
    sorted_degrees = sorted(degrees.values(), reverse=True)
    total_degree = sum(sorted_degrees)
    top10_degree = sum(sorted_degrees[:10])
    hub_concentration = top10_degree / total_degree if total_degree > 0 else 0
    
    # Feature 5: assortativity
    try:
        assortativity = nx.degree_pearson_correlation_coefficient(g)
        if np.isnan(assortativity): assortativity = 0
    except:
        assortativity = 0
        
    # Feature 6: degree entropy
    degree_sequence = np.array(sorted_degrees)
    degree_entropy = entropy(degree_sequence + 1e-10)  # add epsilon to avoid log(0)
    
    return [density, avg_clustering, avg_path_length, hub_concentration, assortativity, degree_entropy]

def run():
    rows = []
    for year in YEARS:
        try:
            g = load_graph(year, 'deepcnl')
            features = extract_features(g)
            rows.append([year] + features)
        except Exception as e:
            print(f"Error processing year {year}: {e}")
    
    df = pd.DataFrame(rows, columns=['year', 'density', 'avg_clustering', 
                                      'avg_path_length', 'hub_concentration', 'assortativity', 'degree_entropy'])
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, 'outputs', 'features')
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, 'topology_features.csv')
    df.to_csv(out_path, index=False)
    print(df.to_string())
    return df

if __name__ == "__main__":
    run()
