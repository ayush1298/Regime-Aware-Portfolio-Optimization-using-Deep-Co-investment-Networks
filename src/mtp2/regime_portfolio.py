import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import same standard metrics as Phase 4
from src.src.mtp2.backtest import sharpe_ratio, max_drawdown, cumulative_return

sns.set_theme(style='whitegrid')

def run():
    # 1. Load Phase 2 regimes (Detected Regimes)
    with open(os.path.join(project_root, 'outputs', 'regimes', 'regime_labels.json'), 'r') as f:
        regime_data = json.load(f)
    
    regimes = regime_data['named'] # e.g. {"2010": "recovery", ...}
    
    # 2. Load Phase 4 strict backtesting numbers
    backtest_df = pd.read_csv(os.path.join(project_root, 'outputs', 'portfolios', 'backtest_results.csv'))
    
    # 3. Build Regime-Aware Portfolio dynamically based on Phase 2 outputs
    regime_aware_returns = []
    
    for _, row in backtest_df.iterrows():
        test_year = int(row['year'])
        base_year = test_year - 1 # Prevent look-ahead bias!
        
        # Determine the detected regime from the base_year graph
        regime = regimes[str(base_year)]
        
        if regime == 'bull':
            # Follow market leaders in strong network environments
            ret = row['deepcnl_hub_follow']
        else:
            # crisis or recovery -> avoid hub contagion when network is fragmented or tight
            ret = row['deepcnl_hub_avoid']
            
        regime_aware_returns.append(ret)
        
    backtest_df['regime_aware'] = regime_aware_returns
    
    # Save the expanded results
    out_dir = os.path.join(project_root, 'outputs', 'portfolios')
    backtest_df.to_csv(os.path.join(out_dir, 'regime_portfolio_results.csv'), index=False)
    
    # 4. Compute metrics for all 5 strategies to expose the combined "Thesis Result"
    strategies = [
        'deepcnl_hub_avoid', 
        'deepcnl_hub_follow', 
        'pcc_hub_avoid', 
        'sp500_equal_weight',
        'regime_aware'
    ]
    
    metrics = []
    plt.figure(figsize=(10, 6))
    
    for strat in strategies:
        returns = backtest_df[strat].tolist()
        sr = sharpe_ratio(returns)
        cr = cumulative_return(returns)
        
        wealth_path = np.cumprod([1 + r for r in returns])
        md = max_drawdown(wealth_path)
        
        metrics.append({'Strategy': strat, 'Sharpe': sr, 'Cum_Return': cr, 'Max_Drawdown': md})
        
        years_list = [2010] + backtest_df['year'].tolist()
        
        # Emphasize our target strategy
        if strat == 'regime_aware':
            plt.plot(years_list, [1.0] + list(wealth_path), label='Regime-Aware (Adaptive)', marker='*', linewidth=3, markersize=10, color='purple')
        else:
            plt.plot(years_list, [1.0] + list(wealth_path), label=strat, marker='o', alpha=0.7)
            
    plt.title('Phase 5: Regime-Aware Adaptive Portfolio vs Static Strategies')
    plt.xlabel('Year')
    plt.ylabel('Portfolio Value (Base 1.0)')
    plt.legend()
    plt.tight_layout()
    fig_dir = os.path.join(project_root, 'outputs', 'figures')
    plt.savefig(os.path.join(fig_dir, 'regime_portfolio_comparison.png'))
    plt.close()
    
    # Regime Timeline Overlay specifically requested in plan.md
    plt.figure(figsize=(10, 5))
    colors = {'bull': 'green', 'crisis': 'red', 'recovery': 'orange'}
    
    x_pos = np.arange(len(backtest_df['year']))
    bar_colors = [colors[regimes[str(int(y) - 1)]] for y in backtest_df['year']]
    returns_pct = backtest_df['regime_aware'].values * 100 

    plt.bar(x_pos, returns_pct, color=bar_colors, edgecolor='black')
    
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=v, label=k.capitalize()) for k, v in colors.items()]
    plt.legend(handles=legend_handles, title="Detected Regime (Y-1)")
    
    plt.xticks(x_pos, backtest_df['year'])
    plt.axhline(0, color='black', linewidth=1)
    plt.ylabel('Annual Return (%)')
    plt.title('Regime-Aware Portfolio Switch History (Overlay)')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'regime_portfolio_overlay.png'))
    plt.close()
    
    df_metrics = pd.DataFrame(metrics)
    print("\n--- Final Phase 5 Performance Summary ---")
    print(df_metrics.to_string(index=False))
    print("\nPerfect regime orchestration completed successfully.")

if __name__ == "__main__":
    run()
