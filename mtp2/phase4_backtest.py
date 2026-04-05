import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from mtp2.utils import load_price_data, annual_return

sns.set_theme(style='whitegrid')

def sharpe_ratio(annual_returns, risk_free_rate=0.02):
    """Annualized Sharpe ratio. Risk-free rate = 2% (approx US Treasury)."""
    if len(annual_returns) == 0:
        return 0
    excess = np.array(annual_returns) - risk_free_rate
    std = np.std(excess)
    return np.mean(excess) / std if std > 0 else 0

def max_drawdown(wealth_path):
    """Maximum peak-to-trough decline over annual wealth path."""
    if len(wealth_path) == 0:
        return 0
    # prepending 1.0 establishes the baseline starting wealth
    path = np.array([1.0] + list(wealth_path))
    peak = np.maximum.accumulate(path)
    drawdown = (path - peak) / peak
    return drawdown.min()

def cumulative_return(annual_returns):
    if len(annual_returns) == 0:
        return 0
    return np.prod([1 + r for r in annual_returns]) - 1

def run():
    print("Loading price data, this may take a moment...")
    prices_df = load_price_data()
    
    deepcnl_cent = pd.read_csv(os.path.join(project_root, 'outputs', 'centrality', 'centrality_scores.csv'))
    pcc_cent = pd.read_csv(os.path.join(project_root, 'outputs', 'centrality', 'pcc_centrality_scores.csv'))
    
    # Evaluate over 2011-2016 to prevent look-ahead bias and accommodate dataset limits.
    backtest_years = list(range(2011, 2017))
    all_tickers = prices_df['symbol'].unique().tolist()
    
    results = []
    
    for test_year in backtest_years:
        base_year = test_year - 1 # Use centrality derived from Y-1
        print(f"Constructing portfolio for {test_year} using {base_year} graphs...")
        
        y_deepcnl = deepcnl_cent[deepcnl_cent['year'] == base_year].sort_values('composite_score', ascending=False)
        y_pcc = pcc_cent[pcc_cent['year'] == base_year].sort_values('composite_score', ascending=False)
        
        deepcnl_hub_follow = y_deepcnl.head(30)['ticker'].tolist()
        deepcnl_hub_avoid = y_deepcnl.tail(30)['ticker'].tolist()
        pcc_hub_avoid = y_pcc.tail(30)['ticker'].tolist()
        
        r_hub_follow = annual_return(deepcnl_hub_follow, prices_df, test_year)
        r_hub_avoid = annual_return(deepcnl_hub_avoid, prices_df, test_year)
        r_pcc_avoid = annual_return(pcc_hub_avoid, prices_df, test_year)
        r_sp500 = annual_return(all_tickers, prices_df, test_year)
        
        results.append({
            'year': test_year,
            'deepcnl_hub_avoid': r_hub_avoid,
            'deepcnl_hub_follow': r_hub_follow,
            'pcc_hub_avoid': r_pcc_avoid,
            'sp500_equal_weight': r_sp500
        })
        
    df_results = pd.DataFrame(results)
    
    out_dir = os.path.join(project_root, 'outputs', 'portfolios')
    os.makedirs(out_dir, exist_ok=True)
    df_results.to_csv(os.path.join(out_dir, 'backtest_results.csv'), index=False)
    
    # Calculate metrics
    strategies = ['deepcnl_hub_avoid', 'deepcnl_hub_follow', 'pcc_hub_avoid', 'sp500_equal_weight']
    metrics = []
    
    plt.figure(figsize=(10, 6))
    
    for strat in strategies:
        returns = df_results[strat].tolist()
        sr = sharpe_ratio(returns)
        cr = cumulative_return(returns)
        
        wealth_path = np.cumprod([1 + r for r in returns])
        md = max_drawdown(wealth_path)
        
        metrics.append({'Strategy': strat, 'Sharpe': sr, 'Cum_Return': cr, 'Max_Drawdown': md})
        
        plt.plot([2010] + backtest_years, [1.0] + list(wealth_path), label=strat, marker='o')
        
    plt.title('Cumulative Returns (Wealth Path) [2011-2016]')
    plt.xlabel('Year')
    plt.ylabel('Portfolio Value (Base 1.0)')
    plt.legend()
    plt.tight_layout()
    fig_dir = os.path.join(project_root, 'outputs', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'cumulative_returns.png'))
    plt.close()
    
    df_metrics = pd.DataFrame(metrics)
    print("\nPerformance Metrics:")
    print(df_metrics.to_string(index=False))
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Strategy', y='Sharpe', data=df_metrics)
    plt.title('Sharpe Ratio Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'sharpe_comparison.png'))
    plt.close()
    
    print("\nSaved backtest results and figures to outputs/portfolios/ and outputs/figures/")

if __name__ == "__main__":
    run()
