import os
import sys
import json
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Colors for regimes
REGIME_COLORS = {
    'bull': 'rgba(16, 185, 129, 0.8)',      # Green
    'crisis': 'rgba(239, 68, 68, 0.8)',     # Red
    'recovery': 'rgba(245, 158, 11, 0.8)'   # Orange
}

KNOWN_EVENTS = {
    2010: "Post-2008 recovery, flash crash",
    2011: "European debt crisis, US downgrade",
    2012: "Markets stabilized",
    2013: "Strong bull run (+30% S&P)",
    2014: "Continued growth",
    2015: "China slowdown fears",
    2016: "Brexit, Trump election volatility"
}

EXPECTED_REGIMES = {
    2010: "recovery",
    2011: "crisis",
    2012: "bull",
    2013: "bull",
    2014: "bull",
    2015: "recovery",
    2016: "recovery"
}

def generate_web_data():
    ## 1. Portfolio Data
    port_csv = os.path.join(project_root, 'outputs', 'portfolios', 'regime_portfolio_results.csv')
    df_port = pd.read_csv(port_csv)
    
    annual_returns = {}
    for _, row in df_port.iterrows():
        year = str(int(row['year']))
        annual_returns[year] = {
            "deepcnl_hub_avoid": row['deepcnl_hub_avoid'],
            "deepcnl_hub_follow": row['deepcnl_hub_follow'],
            "pcc_hub_avoid": row['pcc_hub_avoid'],
            "sp500": row['sp500_equal_weight'],
            "regime_aware": row['regime_aware']
        }
        
    from src.src.mtp2.backtest import sharpe_ratio, max_drawdown, cumulative_return
    strategies = [
        ('deepcnl_hub_avoid', 'DeepCNL Hub-Avoid'),
        ('deepcnl_hub_follow', 'DeepCNL Hub-Follow'),
        ('pcc_hub_avoid', 'PCC Hub-Avoid'),
        ('sp500_equal_weight', 'S&P 500 Equal-Weight'),
        ('regime_aware', 'Regime-Aware (Adaptive)')
    ]
    
    summary_metrics = {}
    for col, name in strategies:
        returns = df_port[col].tolist()
        sr = sharpe_ratio(returns)
        cr = cumulative_return(returns)
        wealth_path = np.cumprod([1 + r for r in returns])
        md = max_drawdown(wealth_path)
        summary_metrics[col] = {
            "name": name,
            "sharpe": float(sr),
            "max_drawdown": float(md),
            "cumulative_return": float(cr)
        }
        
    portfolio_data = {
        "annual_returns": annual_returns,
        "summary_metrics": summary_metrics
    }
    
    with open(os.path.join(project_root, 'web_app', 'data', 'portfolio_data.json'), 'w') as f:
        json.dump(portfolio_data, f, indent=2)
        
    ## 2. Regime Data
    with open(os.path.join(project_root, 'outputs', 'regimes', 'regime_labels.json'), 'r') as f:
        labels_json = json.load(f)
        
    features_csv = os.path.join(project_root, 'outputs', 'features', 'topology_features.csv')
    df_feat = pd.read_csv(features_csv)
    
    regime_timeline = []
    matches = 0
    
    for _, row in df_feat.iterrows():
        year = int(row['year'])
        regime = labels_json['named'][str(year)]
        
        is_match = (regime == EXPECTED_REGIMES[year])
        if is_match: matches += 1
            
        regime_timeline.append({
            "year": year,
            "regime": regime,
            "color": REGIME_COLORS[regime],
            "known_event": KNOWN_EVENTS[year],
            "is_match": is_match
        })
        
    feature_vectors = {}
    for _, row in df_feat.iterrows():
        year = str(int(row['year']))
        feature_vectors[year] = {
            "density": row['density'],
            "avg_clustering": row['avg_clustering'],
            "avg_path_length": row['avg_path_length'],
            "assortativity": row['assortativity'],
            "degree_entropy": row['degree_entropy']
        }
        
    # Scale features 0-1 for radar chart
    from sklearn.preprocessing import MinMaxScaler
    X = df_feat[['density', 'avg_clustering', 'avg_path_length', 'assortativity', 'degree_entropy']]
    X_scaled = MinMaxScaler().fit_transform(X)
    df_feat[['density', 'avg_clustering', 'avg_path_length', 'assortativity', 'degree_entropy']] = X_scaled
    df_feat['regime'] = [labels_json['named'][str(int(y))] for y in df_feat['year']]
    
    # Calculate group averages for radar
    radar_data = {}
    grouped = df_feat.groupby('regime').mean(numeric_only=True)
    for regime in ['bull', 'crisis', 'recovery']:
        if regime in grouped.index:
            radar_data[regime] = grouped.loc[regime].to_dict()
            if 'year' in radar_data[regime]:
                del radar_data[regime]['year']
                
    regime_data = {
        "regime_timeline": regime_timeline,
        "feature_vectors": feature_vectors,
        "radar_averages": radar_data,
        "validation": {
            "matches": matches,
            "total_years": len(df_feat)
        }
    }
    
    with open(os.path.join(project_root, 'web_app', 'data', 'regime_data.json'), 'w') as f:
        json.dump(regime_data, f, indent=2)

    print("Phase 6 Web Data Generated Successfully.")
    
if __name__ == '__main__':
    generate_web_data()
