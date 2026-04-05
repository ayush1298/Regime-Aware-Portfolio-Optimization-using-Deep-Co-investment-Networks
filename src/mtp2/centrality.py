import networkx as nx
import pandas as pd
import os
import sys
from scipy.stats import zscore

# Ensure project root is in python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.src.utils.core_utils import load_graph, YEARS

def compute_centrality(method):
    all_rows = []
    
    for year in YEARS:
        try:
            g = load_graph(year, method)
        except Exception as e:
            print(f"Skipping {method} graph for {year} due to error: {e}")
            continue
            
        nodes = list(g.nodes())
        if not nodes:
            continue
            
        # Check if graph is weighted
        is_weighted = False
        for _, _, data in g.edges(data=True):
            if 'weight' in data:
                is_weighted = True
                break
                
        # 1. Degree Centrality
        degree_dict = nx.degree_centrality(g)
        
        # 2. Betweenness Centrality
        if is_weighted:
            try:
                betweenness_dict = nx.betweenness_centrality(g, weight='weight')
            except Exception:
                betweenness_dict = nx.betweenness_centrality(g)
        else:
            betweenness_dict = nx.betweenness_centrality(g)
            
        # 3. Eigenvector Centrality
        try:
            if is_weighted:
                eigen_dict = nx.eigenvector_centrality(g, weight='weight', max_iter=1000)
            else:
                eigen_dict = nx.eigenvector_centrality(g, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            print(f"  Eigenvector failed to converge for {method} {year}. Falling back to degree_centrality")
            eigen_dict = {n: degree_dict[n] for n in nodes}
            
        for n in nodes:
            all_rows.append({
                'ticker': n,
                'year': year,
                'degree_centrality': degree_dict[n],
                'betweenness_centrality': betweenness_dict.get(n, 0),
                'eigenvector_centrality': eigen_dict.get(n, 0)
            })
            
    df = pd.DataFrame(all_rows)
    
    if len(df) == 0:
        print(f"No nodes found for {method}.")
        return df
        
    df['composite_score'] = (
        zscore(df['degree_centrality'].fillna(0)) + 
        zscore(df['betweenness_centrality'].fillna(0)) + 
        zscore(df['eigenvector_centrality'].fillna(0))
    ) / 3
    
    # Sort
    df = df.sort_values(by=['year', 'composite_score'], ascending=[True, False])
    
    # Save
    out_dir = os.path.join(project_root, 'outputs', 'centrality')
    os.makedirs(out_dir, exist_ok=True)
    
    filename = 'centrality_scores.csv' if method == 'deepcnl' else f'{method}_centrality_scores.csv'
    out_path = os.path.join(out_dir, filename)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} centrality rows for {method} to {filename}")
    return df

def run():
    print("Computing centrality for deepcnl graphs...")
    compute_centrality('deepcnl')
    
    print("\nComputing centrality for pcc graphs...")
    compute_centrality('pcc')
    
if __name__ == "__main__":
    run()
