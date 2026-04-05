import pickle
import os
import pandas as pd
import networkx as nx
import numpy as np

YEARS = list(range(2010, 2017))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'graphs')
DATA_PATH = os.path.join(PROJECT_ROOT, 'Data', 'prices-split-adjusted.csv')


def save_graph(g, year, method='deepcnl'):
    """Save a networkx graph as a pickle file."""
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    path = os.path.join(GRAPHS_DIR, f'{method}_graph_{year}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(g, f)
    print(f'Saved {method} graph for {year}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges')


def load_graph(year, method='deepcnl'):
    """Load a saved networkx graph."""
    path = os.path.join(GRAPHS_DIR, f'{method}_graph_{year}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Graph not found: {path}. Run Phase 0 first.')
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_price_data():
    """Load and preprocess the price CSV."""
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])
    return df


def get_lcc(g):
    """Return the largest connected component of a graph."""
    if g.number_of_nodes() == 0:
        return nx.Graph()
    components = sorted(nx.connected_components(g), key=len, reverse=True)
    return g.subgraph(components[0]).copy()


def annual_return(tickers, prices_df, year):
    """
    Compute equal-weight portfolio annual return.
    year: the calendar year to compute returns for (hold from Jan 1 to Dec 31).
    """
    start = pd.Timestamp(f'{year}-01-01')
    end = pd.Timestamp(f'{year}-12-31')
    year_prices = prices_df[(prices_df['date'] >= start) & (prices_df['date'] <= end)]

    returns = []
    for ticker in tickers:
        stock = year_prices[year_prices['symbol'] == ticker].sort_values('date')
        if len(stock) < 20:  # need at least 20 trading days
            continue
        ret = (stock['close'].iloc[-1] - stock['close'].iloc[0]) / stock['close'].iloc[0]
        returns.append(ret)

    if not returns:
        return 0.0
    return float(pd.Series(returns).mean())
