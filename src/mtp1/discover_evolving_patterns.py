"""
Standalone script to discover evolving co-investment patterns
Reproduces Figure 5 from the DeepCNL paper
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.mtp1.stock_network_analysis import *
from src.mtp1.data_util import Data_util

def main():
    print("=" * 70)
    print("EVOLVING CO-INVESTMENT PATTERN DISCOVERY")
    print("Reproducing Figure 5 from DeepCNL Paper")
    print("=" * 70)
    
    # Check data
    if not check_required_data():
        print("\n❌ Required data files missing!")
        print("Run: python download_market_data.py")
        return
    
    # Initialize
    datatool = Data_util(TICKER_NUM, WINDOW, FEATURE_NUM, DATA_PATH, SPY_PATH)
    experiment = Experimental_platform(datatool)
    
    # Run analysis
    results = experiment.evolving_coinvestment_patterns(
        start_year=2010,
        end_year=2013,
        rare_ratio=0.001,  # γ from paper
        compare_with_pcc=True
    )
    
    # Analyze results
    print("\n" + "=" * 70)
    print("ADDITIONAL ANALYSIS")
    print("=" * 70)
    
    persistent_nodes = experiment.analyze_temporal_stability(results)
    experiment.compare_dnl_vs_pcc_structure(results)
    
    print("\n✅ Analysis complete!")
    print(f"📁 Network visualizations saved in: network_figures/")
    print(f"\nFiles generated:")
    for year in range(2010, 2014):
        print(f"  - network_figures/{year}_DNL_network.png")
        print(f"  - network_figures/{year}_PCC_network.png")

if __name__ == "__main__":
    main()