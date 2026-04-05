"""
Phase 0 — Save DeepCNL and PCC graph outputs as pickle files.

Runs evolving_coinvestment_patterns() and saves each year's graph to
outputs/graphs/deepcnl_graph_{year}.pkl and outputs/graphs/pcc_graph_{year}.pkl.

Usage:
    python -m mtp2.phase0_save_graphs
    python -m mtp2.phase0_save_graphs --ticker_num 50   # local test with fewer tickers
"""
import sys
import os
import argparse
import time

# Add project root to path so stock_network_analysis imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.src.utils.core_utils import save_graph, GRAPHS_DIR, YEARS


def run_and_save_all_years(ticker_num=470, rare_ratio=0.002, start_year=2010, end_year=2016):
    """
    Run DeepCNL and PCC for each year and save graphs as pickle files.

    This imports stock_network_analysis at call time (not module level)
    to avoid triggering model init on import.
    """
    from src.mtp1.stock_network_analysis import (
        Experimental_platform, Data_util,
        WINDOW, FEATURE_NUM, DATA_PATH, SPY_PATH
    )

    print(f"{'='*70}")
    print(f"PHASE 0 — SAVING DEEPCNL & PCC GRAPHS")
    print(f"{'='*70}")
    print(f"Ticker count: {ticker_num}")
    print(f"Rare ratio:   {rare_ratio}")
    print(f"Years:        {start_year}–{end_year}")
    print(f"Output dir:   {GRAPHS_DIR}")
    print(f"{'='*70}\n")

    datatool = Data_util(ticker_num, WINDOW, FEATURE_NUM, DATA_PATH, SPY_PATH)
    experiment = Experimental_platform(datatool)

    saved_count = 0

    for year in range(start_year, end_year + 1):
        seed = year - 2010
        print(f"\n{'='*70}")
        print(f"Year {year} (seed={seed})")
        print(f"{'='*70}")

        train_period, test_period = experiment.period_generator(seed)

        try:
            train_x = datatool.load_x(train_period)
            train_y = datatool.load_y(train_period)

            if len(train_y) < 10:
                print(f"Skipping {year} — insufficient training data ({len(train_y)} labels)")
                continue

            # DeepCNL graph
            print(f"\nTraining DeepCNL for {year}...")
            g_deepcnl = experiment.deep_CNL('igo', train_x, train_y, rare_ratio)
            save_graph(g_deepcnl, year, 'deepcnl')

            # PCC graph (uses same loaded data via datatool.compare_data)
            print(f"Computing PCC for {year}...")
            g_pcc = experiment.Pearson_cor(rare_ratio)
            save_graph(g_pcc, year, 'pcc')

            saved_count += 1

        except Exception as e:
            print(f"Error processing {year}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print(f"PHASE 0 COMPLETE — Saved graphs for {saved_count}/{end_year - start_year + 1} years")
    print(f"Output: {GRAPHS_DIR}/")
    print(f"{'='*70}")


def run_single_year(year, ticker_num=470, rare_ratio=0.002):
    """Run and save graphs for a single year (for SLURM parallel jobs)."""
    run_and_save_all_years(
        ticker_num=ticker_num,
        rare_ratio=rare_ratio,
        start_year=year,
        end_year=year,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 0: Save DeepCNL & PCC graphs')
    parser.add_argument('--ticker_num', type=int, default=470,
                        help='Number of tickers (default: 470, use 50 for local testing)')
    parser.add_argument('--rare_ratio', type=float, default=0.002,
                        help='Edge density parameter (default: 0.002)')
    parser.add_argument('--start_year', type=int, default=2010)
    parser.add_argument('--end_year', type=int, default=2016)
    parser.add_argument('--year', type=int, default=None,
                        help='Run a single year only (for SLURM jobs)')
    args = parser.parse_args()

    start = time.time()

    if args.year is not None:
        run_single_year(args.year, args.ticker_num, args.rare_ratio)
    else:
        run_and_save_all_years(args.ticker_num, args.rare_ratio, args.start_year, args.end_year)

    elapsed = time.time() - start
    print(f"\nTime elapsed: {elapsed:.1f}s")
