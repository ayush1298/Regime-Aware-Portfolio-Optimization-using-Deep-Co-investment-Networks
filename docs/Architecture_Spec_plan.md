# Implementation Plan
## Deep Co-investment Network Learning — Regime Detection & Portfolio Optimization

---

## Overview

MTP1 built DeepCNL — a deep learning framework that constructs co-investment networks from S&P 500 stock pairs. The output is a `networkx.Graph` per year (2010–2016), where nodes are stock tickers and weighted edges represent learned co-investment strength.

MTP2 extends this in two connected directions:

- **Module A — Market Regime Detection:** Use graph topology features extracted from DeepCNL networks to automatically cluster years into market regimes (bull, bear, crisis) — without any labels.
- **Module B — Portfolio Construction & Backtesting:** Use network centrality scores from DeepCNL graphs to build and backtest stock portfolios, benchmarked against S&P 500 and PCC-based alternatives.
- **Combined insight:** A regime-aware portfolio that switches strategy based on the detected regime, outperforming both static approaches.

---

## Repository Structure After MTP2

```
Deep-Co-investment-Network-Learning-for-Financial-Assets/
│
├── Data/
│   ├── prices-split-adjusted.csv          # Existing: raw stock prices
│   ├── SPY20000101_20171111.csv            # Existing: SPY index data
│   └── SP500^GSPC20000101_20171111.csv     # Existing: S&P 500 index
│
├── Models/                                 # Existing: CRNN model definitions
│
├── mtp2/                                   # NEW: all MTP2 code lives here
│   ├── phase0_save_graphs.py               # Patch to save DeepCNL graph outputs
│   ├── phase1_feature_extraction.py        # Extract topology features from graphs
│   ├── phase2_regime_detection.py          # K-means clustering → regime labels
│   ├── phase3_centrality.py               # Compute centrality scores per stock/year
│   ├── phase4_backtest.py                 # Build portfolios, compute Sharpe/returns
│   ├── phase5_regime_portfolio.py         # Regime-switching adaptive portfolio
│   ├── phase6_web_extension.py            # Update Flask web app with new tabs
│   └── utils.py                           # Shared helpers (load graphs, price data)
│
├── outputs/
│   ├── graphs/                            # Saved DeepCNL graphs per year
│   │   ├── deepcnl_graph_2010.pkl
│   │   ├── deepcnl_graph_2011.pkl
│   │   └── ... (one per year, 2010–2016)
│   ├── features/
│   │   └── topology_features.csv          # 7 years × 5 features matrix
│   ├── regimes/
│   │   └── regime_labels.json             # {year: regime_label} mapping
│   ├── centrality/
│   │   └── centrality_scores.csv          # stock × year centrality matrix
│   ├── portfolios/
│   │   └── backtest_results.csv           # annual returns, Sharpe, drawdown
│   └── figures/                           # All plots saved here
│
├── web_app/
│   ├── app.py                             # Existing Flask app (extended)
│   ├── data/
│   │   ├── network_data.json              # Existing
│   │   ├── regime_data.json               # NEW: regime timeline data
│   │   └── portfolio_data.json            # NEW: portfolio performance data
│   └── templates/
│       ├── regime.html                    # NEW: regime detection tab
│       └── portfolio.html                 # NEW: portfolio performance tab
│
├── stock_network_analysis.py              # Existing (patched in Phase 0)
└── requirements_mtp2.txt                  # New dependencies
```

---

## Key Technical Facts (for the implementing AI)

These facts are extracted directly from the existing codebase. Do not change these assumptions.

**Graph format:** `deep_CNL()` in `stock_network_analysis.py` returns a `networkx.Graph` with weighted edges. Node names are stock ticker strings (e.g., `'AAPL'`, `'DAL'`). Edge attribute key is `'weight'`.

**Year mapping:** `period_generator(seed)` maps `seed=0` → year 2010, `seed=1` → year 2011, ..., `seed=6` → year 2016. Train period is `{2010+seed}-01-01` to `{2010+seed}-12-31`.

**Price data format:** `Data/prices-split-adjusted.csv` has columns: `date`, `symbol`, `open`, `close`, `low`, `high`, `volume`. Date format is `YYYY-MM-DD`.

**Existing parameters used in MTP1:**
- `TICKER_NUM = 470` (full run), `WINDOW = 32`, `FEATURE_NUM = 4`
- `RARE_RATIO = 0.002` (controls edge density — keep this same)
- `HIDDEN_UNIT_NUM = 256`, `EPOCH_NUM = 200`
- `CRNN_CODE = 'CRNN_LSTM'`, `DNL_INPLEMENTATION = 'igo'`

**Available baselines already in codebase:** `Pearson_cor()` and `DTW_graph()` in `stock_network_analysis.py` — use these for benchmark comparison.

**Existing web app:** Flask app at `web_app/app.py`. Data served as JSON from `web_app/data/`. Frontend uses D3.js and Chart.js. Add new routes and JSON files — do not break existing routes.

---

## Dependencies to Install

```bash
pip install networkx scikit-learn scipy matplotlib seaborn pandas numpy pickle5
```

Add to `requirements_mtp2.txt`:
```
networkx>=3.0
scikit-learn>=1.3
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
pandas>=2.0
numpy>=1.24
```

All PyTorch, Flask, D3.js dependencies already exist from MTP1.

---

## Phase 0 — Patch DeepCNL to Save Graph Outputs

**File to create:** `mtp2/phase0_save_graphs.py`

**Purpose:** Modify the training loop in `stock_network_analysis.py` to save each year's DeepCNL graph as a `.pkl` file immediately after training, so MTP2 can load them without retraining.

**What to implement:**

1. Create `outputs/graphs/` directory if it doesn't exist.
2. After `deep_CNL()` returns a graph `g`, save it using `pickle` as `outputs/graphs/deepcnl_graph_{year}.pkl`.
3. Also save the PCC graph for the same year as `outputs/graphs/pcc_graph_{year}.pkl` (for benchmarking in Phase 4).
4. Add a standalone function `run_and_save_all_years()` that loops `seed = 0` to `6`, calls `evolving_coinvestment_patterns()` (already exists in `stock_network_analysis.py`), and saves each graph.

**Code pattern:**
```python
import pickle
import os

def save_graph(g, year, method='deepcnl'):
    os.makedirs('outputs/graphs', exist_ok=True)
    path = f'outputs/graphs/{method}_graph_{year}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(g, f)
    print(f'Saved {method} graph for {year}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges')

def load_graph(year, method='deepcnl'):
    path = f'outputs/graphs/{method}_graph_{year}.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)
```

**Patch location in `stock_network_analysis.py`:** Inside `evolving_coinvestment_patterns()`, after the line `g_dnl = self.deep_CNL('igo', train_x, train_y, rare_ratio)`, add `save_graph(g_dnl, year, 'deepcnl')`. After `g_pcc = self.Pearson_cor(rare_ratio)`, add `save_graph(g_pcc, year, 'pcc')`.

**Verification:** After running, `outputs/graphs/` should contain 14 files (7 DeepCNL + 7 PCC, years 2010–2016).

---

## Phase 1 — Topology Feature Extraction

**File to create:** `mtp2/phase1_feature_extraction.py`

**Purpose:** For each year's DeepCNL graph, compute 5 scalar topology features that characterize the market structure of that year. These become the input to the clustering algorithm in Phase 2.

**Input:** `outputs/graphs/deepcnl_graph_{year}.pkl` for years 2010–2016.

**Output:** `outputs/features/topology_features.csv` — a CSV with 8 columns: `year`, and 5 feature columns.

**The 5 features to compute (all using `networkx`):**

| Feature | Variable name | `networkx` function | Description |
|---|---|---|---|
| Network density | `density` | `nx.density(g)` | Ratio of actual to possible edges. High in crisis (everyone correlated). |
| Average clustering coefficient | `avg_clustering` | `nx.average_clustering(g)` | How clique-like neighborhoods are. High in bull markets. |
| Average shortest path length | `avg_path_length` | `nx.average_shortest_path_length(g_lcc)` | Computed on largest connected component only. Low = tightly connected. |
| Hub concentration | `hub_concentration` | Manual: top-10 degree sum / total degree sum | Fraction of connectivity held by the 10 most-connected stocks. High = few dominant hubs. |
| Network entropy | `degree_entropy` | `scipy.stats.entropy(degree_sequence)` | Shannon entropy over the degree distribution. Low = unequal, hub-dominated. |

**Important implementation note for `avg_path_length`:** This must be computed on the largest connected component (LCC) only, not the full graph (disconnected graphs have infinite path length). Use `nx.connected_components()` to extract the LCC first.

**Code structure:**
```python
import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import entropy
from mtp2.utils import load_graph

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
    components = sorted(nx.connected_components(g), key=len, reverse=True)
    lcc = g.subgraph(components[0]).copy()
    avg_path_length = nx.average_shortest_path_length(lcc) if lcc.number_of_nodes() > 1 else 0
    
    # Feature 4: hub concentration (top-10 degree / total degree)
    degrees = dict(g.degree())
    sorted_degrees = sorted(degrees.values(), reverse=True)
    total_degree = sum(sorted_degrees)
    top10_degree = sum(sorted_degrees[:10])
    hub_concentration = top10_degree / total_degree if total_degree > 0 else 0
    
    # Feature 5: degree entropy
    degree_sequence = np.array(sorted_degrees)
    degree_entropy = entropy(degree_sequence + 1e-10)  # add epsilon to avoid log(0)
    
    return [density, avg_clustering, avg_path_length, hub_concentration, degree_entropy]

def run():
    rows = []
    for year in YEARS:
        g = load_graph(year, 'deepcnl')
        features = extract_features(g)
        rows.append([year] + features)
    
    df = pd.DataFrame(rows, columns=['year', 'density', 'avg_clustering', 
                                      'avg_path_length', 'hub_concentration', 'degree_entropy'])
    os.makedirs('outputs/features', exist_ok=True)
    df.to_csv('outputs/features/topology_features.csv', index=False)
    print(df.to_string())
    return df
```

**Verification:** Print the resulting DataFrame. Expect to see clearly different feature values for 2010–2011 (post-crisis recovery), 2012–2014 (bull market), and visually distinct patterns across years.

---

## Phase 2 — Market Regime Detection

**File to create:** `mtp2/phase2_regime_detection.py`

**Purpose:** Apply unsupervised k-means clustering on the feature matrix from Phase 1 to assign each year a regime label. Validate the labels against known historical market events.

**Input:** `outputs/features/topology_features.csv`

**Output:**
- `outputs/regimes/regime_labels.json` — `{"2010": 1, "2011": 2, "2012": 0, ...}`
- `outputs/figures/regime_pca.png` — PCA scatter plot of years colored by regime
- `outputs/figures/regime_timeline.png` — Timeline bar chart showing regime per year
- Console output: silhouette scores for k=2,3,4 to justify k=3

**Step-by-step implementation:**

**Step 2.1 — Normalize features:**
Use `sklearn.preprocessing.StandardScaler` to z-score all 5 features before clustering. Features have very different scales (density ≈ 0.01, entropy ≈ 5.0) — normalization is mandatory.

**Step 2.2 — Choose k using elbow + silhouette:**
Compute `sklearn.cluster.KMeans` for k=2,3,4. Compute inertia (elbow method) and `sklearn.metrics.silhouette_score`. Print all scores. Expected: k=3 gives best silhouette score, justifying the 3-regime model (bull / bear / crisis).

**Step 2.3 — Fit k=3 and assign labels:**
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
```

**Step 2.4 — Name the regimes:**
After clustering, inspect which cluster corresponds to which market condition by checking average feature values per cluster. Assign human-readable names:
- Cluster with highest density + lowest path length → `"crisis"` (tight, correlated markets: 2008 aftermath, 2011)
- Cluster with moderate density + high clustering → `"bull"` (normal growth: 2013–2015)
- Cluster with lowest density + highest entropy → `"recovery"` (fragmented, uncertain: 2010, 2016)

**Step 2.5 — PCA visualization:**
Project the 5D feature vectors to 2D using PCA. Plot each year as a labeled point, colored by regime. Save as `outputs/figures/regime_pca.png`.

**Step 2.6 — Validate against historical events:**
Print a table comparing detected regime labels to known events:

| Year | Detected Regime | Known Market Event | Match? |
|------|----------------|-------------------|--------|
| 2010 | recovery | Post-2008 recovery, flash crash | Expected: recovery |
| 2011 | crisis | European debt crisis, US downgrade | Expected: crisis |
| 2012 | bull | Markets stabilized | Expected: bull |
| 2013 | bull | Strong bull run (+30% S&P) | Expected: bull |
| 2014 | bull | Continued growth | Expected: bull |
| 2015 | recovery | China slowdown fears | Expected: recovery |
| 2016 | recovery | Brexit, Trump election volatility | Expected: recovery |

If at least 5 of 7 years align, this is the key result of Module A.

**Step 2.7 — Save regime labels:**
```python
import json
regime_map = {str(year): int(label) for year, label in zip(YEARS, labels)}
regime_names = {str(year): cluster_name_map[int(label)] for year, label in zip(YEARS, labels)}
with open('outputs/regimes/regime_labels.json', 'w') as f:
    json.dump({'numeric': regime_map, 'named': regime_names}, f, indent=2)
```

---

## Phase 3 — Node Centrality Computation

**File to create:** `mtp2/phase3_centrality.py`

**Purpose:** For each year's DeepCNL graph, compute three centrality scores per stock. These scores will be used in Phase 4 to rank and select stocks for portfolios.

**Input:** `outputs/graphs/deepcnl_graph_{year}.pkl` for years 2010–2016.

**Output:** `outputs/centrality/centrality_scores.csv` — rows are (ticker, year), columns are 3 centrality measures + composite score.

**The 3 centrality measures:**

| Measure | `networkx` function | What it captures |
|---|---|---|
| Degree centrality | `nx.degree_centrality(g)` | How many stocks a stock is co-invested with. Direct influence. |
| Betweenness centrality | `nx.betweenness_centrality(g, weight='weight')` | How often a stock lies on shortest paths between others. Broker/bridge role. |
| Eigenvector centrality | `nx.eigenvector_centrality(g, weight='weight', max_iter=1000)` | Influence weighted by neighbors' influence. Captures systemic importance. |

**Composite score:** Compute a composite as the average of the three z-scored centrality measures:
```python
from scipy.stats import zscore
df['composite'] = (zscore(df['degree']) + zscore(df['betweenness']) + zscore(df['eigenvector'])) / 3
```

**Important:** `eigenvector_centrality` may not converge on sparse graphs — catch `nx.PowerIterationFailedConvergence` and fall back to degree centrality for that year.

**Output format of `centrality_scores.csv`:**
```
ticker, year, degree_centrality, betweenness_centrality, eigenvector_centrality, composite_score
AAPL,   2010, 0.042,             0.018,                  0.031,                  0.21
...
```

**Also compute and save for PCC graphs** as `outputs/centrality/pcc_centrality_scores.csv` — needed for the benchmark comparison in Phase 4.

---

## Phase 4 — Portfolio Construction and Backtesting

**File to create:** `mtp2/phase4_backtest.py`

**Purpose:** Build two portfolios using centrality scores and backtest them using real price data. Compare against S&P 500 and PCC-based portfolios.

**Input:**
- `outputs/centrality/centrality_scores.csv` (DeepCNL)
- `outputs/centrality/pcc_centrality_scores.csv` (PCC baseline)
- `Data/prices-split-adjusted.csv` (actual stock prices)

**Output:**
- `outputs/portfolios/backtest_results.csv` — per-year returns for all 4 strategies
- `outputs/figures/cumulative_returns.png` — cumulative return chart
- `outputs/figures/sharpe_comparison.png` — Sharpe ratio bar chart

**The 4 strategies to backtest:**

| Strategy | Description | Stock selection rule |
|---|---|---|
| DeepCNL Hub-Avoid | Low systemic risk | Bottom 30 stocks by composite centrality score |
| DeepCNL Hub-Follow | Follow market movers | Top 30 stocks by composite centrality score |
| PCC Hub-Avoid | PCC baseline equivalent | Bottom 30 by PCC centrality |
| S&P 500 Equal-Weight | Benchmark | Equal weight across all stocks in data |

**Portfolio construction rules (apply to all strategies):**
- Equal weight within each portfolio (1/30 per stock)
- Annual rebalancing: portfolio for year Y is constructed using centrality scores from year Y's graph, then held throughout year Y+1
- Reason for Y → Y+1 shift: centrality scores from training data (year Y) are used to predict the portfolio for the following year (Y+1), avoiding look-ahead bias

**Return calculation:**
```python
def annual_return(tickers, prices_df, year):
    """Compute equal-weight portfolio annual return for a given year."""
    start = f'{year}-01-01'
    end = f'{year}-12-31'
    year_prices = prices_df[(prices_df['date'] >= start) & (prices_df['date'] <= end)]
    
    returns = []
    for ticker in tickers:
        stock_prices = year_prices[year_prices['symbol'] == ticker]['close']
        if len(stock_prices) < 2:
            continue
        ret = (stock_prices.iloc[-1] - stock_prices.iloc[0]) / stock_prices.iloc[0]
        returns.append(ret)
    
    return np.mean(returns) if returns else 0.0
```

**Performance metrics to compute:**

```python
def sharpe_ratio(annual_returns, risk_free_rate=0.02):
    """Annualized Sharpe ratio. Risk-free rate = 2% (approx US Treasury)."""
    excess = np.array(annual_returns) - risk_free_rate
    return np.mean(excess) / np.std(excess) if np.std(excess) > 0 else 0

def max_drawdown(cumulative_returns):
    """Maximum peak-to-trough decline."""
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def cumulative_return(annual_returns):
    return np.prod([1 + r for r in annual_returns]) - 1
```

**Expected results pattern:** DeepCNL Hub-Avoid should have a higher Sharpe ratio than PCC Hub-Avoid (validates MTP1 claim that DeepCNL is 3.3× better at identifying financially influential firms). DeepCNL Hub-Follow should have higher raw returns but also higher risk.

**Output CSV format:**
```
year, deepcnl_hub_avoid, deepcnl_hub_follow, pcc_hub_avoid, sp500_equal_weight
2011, 0.12,              0.21,               0.08,          0.02
2012, 0.18,              0.28,               0.11,          0.16
...
```

---

## Phase 5 — Regime-Aware Portfolio (Combined Insight)

**File to create:** `mtp2/phase5_regime_portfolio.py`

**Purpose:** Build a single adaptive portfolio that switches strategy based on the regime detected in Phase 2. This is the key combined contribution of MTP2 — it requires both Module A and Module B to work together.

**Input:**
- `outputs/regimes/regime_labels.json`
- `outputs/centrality/centrality_scores.csv`
- `Data/prices-split-adjusted.csv`

**Output:**
- `outputs/portfolios/regime_portfolio_results.csv`
- `outputs/figures/regime_portfolio_comparison.png`

**Switching rule:**

| Detected Regime | Portfolio Strategy | Rationale |
|---|---|---|
| `"crisis"` | Hub-Avoid (bottom 30 centrality) | In crisis, hub stocks spread contagion — avoid them |
| `"bull"` | Hub-Follow (top 30 centrality) | In bull runs, market leaders drive returns — follow them |
| `"recovery"` | Hub-Avoid (bottom 30 centrality) | In uncertain markets, prioritize diversification |

**Implementation:**
```python
def regime_portfolio_return(year, regime_labels, centrality_df, prices_df):
    regime = regime_labels['named'][str(year)]
    
    year_centrality = centrality_df[centrality_df['year'] == year]
    year_centrality = year_centrality.sort_values('composite_score', ascending=False)
    
    if regime == 'bull':
        selected = year_centrality.head(30)['ticker'].tolist()
    else:  # crisis or recovery → hub-avoid
        selected = year_centrality.tail(30)['ticker'].tolist()
    
    return annual_return(selected, prices_df, year + 1)  # +1 for no look-ahead bias
```

**Compute and compare 5 strategies on one chart:**
- DeepCNL Hub-Avoid (static)
- DeepCNL Hub-Follow (static)
- PCC Hub-Avoid (baseline)
- S&P 500 Equal-Weight (benchmark)
- Regime-Aware Portfolio (adaptive) ← **this should be best Sharpe**

**Key result to report:** The regime-aware portfolio should achieve the best Sharpe ratio among all strategies. This is the thesis headline result.

---

## Phase 6 — Web App Extension

**Files to create/modify:**
- `web_app/app.py` — add 2 new Flask routes
- `web_app/templates/regime.html` — new tab
- `web_app/templates/portfolio.html` — new tab
- `web_app/data/regime_data.json` — generated from Phase 2 outputs
- `web_app/data/portfolio_data.json` — generated from Phase 4+5 outputs

**Do not modify or break existing routes:** `/`, `/network`, `/rankings`, `/performance`. Only add new routes.

### 6.1 — Generate JSON data files

Create `mtp2/phase6_web_extension.py` with a function `generate_web_data()` that reads from `outputs/` and writes the two new JSON files.

**`regime_data.json` structure:**
```json
{
  "regime_timeline": [
    {"year": 2010, "regime": "recovery", "regime_id": 2, "color": "#F59E0B"},
    {"year": 2011, "regime": "crisis",   "regime_id": 1, "color": "#EF4444"},
    {"year": 2012, "regime": "bull",     "regime_id": 0, "color": "#10B981"},
    ...
  ],
  "feature_vectors": {
    "2010": {"density": 0.021, "avg_clustering": 0.45, ...},
    ...
  },
  "validation": {
    "silhouette_score": 0.62,
    "years_aligned_with_history": 5,
    "total_years": 7
  }
}
```

**`portfolio_data.json` structure:**
```json
{
  "annual_returns": {
    "2011": {"deepcnl_hub_avoid": 0.12, "deepcnl_hub_follow": 0.21, "pcc_hub_avoid": 0.08, "sp500": 0.02, "regime_aware": 0.17},
    ...
  },
  "summary_metrics": {
    "deepcnl_hub_avoid":  {"sharpe": 1.12, "max_drawdown": -0.08, "cumulative_return": 0.94},
    "deepcnl_hub_follow": {"sharpe": 0.87, "max_drawdown": -0.18, "cumulative_return": 1.43},
    "pcc_hub_avoid":      {"sharpe": 0.65, "max_drawdown": -0.12, "cumulative_return": 0.61},
    "sp500":              {"sharpe": 0.55, "max_drawdown": -0.19, "cumulative_return": 0.58},
    "regime_aware":       {"sharpe": 1.34, "max_drawdown": -0.07, "cumulative_return": 1.12}
  }
}
```

### 6.2 — New Flask routes in `app.py`

```python
@app.route('/regime')
def regime():
    with open('data/regime_data.json') as f:
        data = json.load(f)
    return render_template('regime.html', data=data)

@app.route('/portfolio')
def portfolio():
    with open('data/portfolio_data.json') as f:
        data = json.load(f)
    return render_template('portfolio.html', data=data)

@app.route('/api/regime')
def api_regime():
    with open('data/regime_data.json') as f:
        return jsonify(json.load(f))

@app.route('/api/portfolio')
def api_portfolio():
    with open('data/portfolio_data.json') as f:
        return jsonify(json.load(f))
```

### 6.3 — `regime.html` — Regime Detection Tab

Use Chart.js (already in the project) to build:
1. A horizontal bar/color-band timeline showing year → regime color (green=bull, red=crisis, amber=recovery)
2. A radar/spider chart showing average feature values per regime cluster
3. A small table: year | regime | known event | match?

Add a link to this tab in the existing `base.html` navigation.

### 6.4 — `portfolio.html` — Portfolio Performance Tab

Use Chart.js to build:
1. A line chart showing cumulative returns of all 5 strategies over 2011–2016
2. A bar chart comparing Sharpe ratios of all 5 strategies
3. A summary table: strategy | total return | Sharpe | max drawdown
4. A section explaining the regime-switching logic

---

## Execution Order

Run phases in this exact order. Each phase depends on the previous one's output files.

```
Phase 0 → (run DeepCNL on cluster) → outputs/graphs/*.pkl
Phase 1 → (run feature extraction) → outputs/features/topology_features.csv
Phase 2 → (run regime detection)   → outputs/regimes/regime_labels.json + figures
Phase 3 → (run centrality)         → outputs/centrality/centrality_scores.csv
Phase 4 → (run backtesting)        → outputs/portfolios/backtest_results.csv + figures
Phase 5 → (run regime portfolio)   → outputs/portfolios/regime_portfolio_results.csv
Phase 6 → (generate web data)      → web_app/data/regime_data.json, portfolio_data.json
         → (run Flask app)          → python web_app/app.py
```

Each phase can be run independently by calling its `run()` function:

```bash
python -m mtp2.phase1_feature_extraction
python -m mtp2.phase2_regime_detection
python -m mtp2.phase3_centrality
python -m mtp2.phase4_backtest
python -m mtp2.phase5_regime_portfolio
python -m mtp2.phase6_web_extension
```

---

## Cluster Job Script for Phase 0 (SLURM)

**File to create:** `run_deepcnl_year.sh`

This script runs DeepCNL for a single year on the cluster. Submit one job per year in parallel.

```bash
#!/bin/bash
#SBATCH --job-name=deepcnl_year_{YEAR}
#SBATCH --output=logs/deepcnl_{YEAR}_%j.out
#SBATCH --error=logs/deepcnl_{YEAR}_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

YEAR=$1
SEED=$((YEAR - 2010))

echo "Running DeepCNL for year $YEAR (seed=$SEED)"

cd /path/to/Deep-Co-investment-Network-Learning-for-Financial-Assets
source activate deepcnl_env   # or whatever your conda env is called

python -c "
from stock_network_analysis import *
from mtp2.phase0_save_graphs import save_graph

datatool = Data_util(470, WINDOW, FEATURE_NUM, DATA_PATH, SPY_PATH)
experiment = Experimental_platform(datatool)
seed = $SEED
train_period, _ = experiment.period_generator(seed)
train_x = datatool.load_x(train_period)
train_y = datatool.load_y(train_period)
g_deepcnl = experiment.deep_CNL('igo', train_x, train_y, RARE_RATIO)
save_graph(g_deepcnl, $YEAR, 'deepcnl')
g_pcc = experiment.Pearson_cor(RARE_RATIO)
save_graph(g_pcc, $YEAR, 'pcc')
print('Done: year $YEAR')
"
```

**Submit all 7 years in parallel:**
```bash
mkdir -p logs
for year in 2010 2011 2012 2013 2014 2015 2016; do
    sbatch run_deepcnl_year.sh $year
done
```

**Monitor jobs:**
```bash
squeue -u $USER
```

**After all jobs complete, verify outputs:**
```bash
ls -la outputs/graphs/
# Should see: deepcnl_graph_2010.pkl through deepcnl_graph_2016.pkl (14 files total)
```

---

## `mtp2/utils.py` — Shared Utilities

**File to create:** `mtp2/utils.py`

All phases import from here. Implement once, use everywhere.

```python
import pickle
import os
import pandas as pd
import networkx as nx

YEARS = list(range(2010, 2017))
GRAPHS_DIR = 'outputs/graphs'
DATA_PATH = 'Data/prices-split-adjusted.csv'

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
```

---

## Figures to Produce (for Thesis)

All figures saved to `outputs/figures/`. Use `matplotlib` with `seaborn` styling.

| Figure | File | Phase | Description |
|---|---|---|---|
| Feature heatmap | `feature_heatmap.png` | Phase 1 | Heatmap of 7 years × 5 features, years on y-axis |
| Elbow plot | `elbow_plot.png` | Phase 2 | Inertia vs k for k=2,3,4 |
| PCA regime plot | `regime_pca.png` | Phase 2 | 2D PCA scatter, points = years, color = regime |
| Regime timeline | `regime_timeline.png` | Phase 2 | Horizontal color bars, one per year |
| Centrality rankings | `centrality_top20.png` | Phase 3 | Bar chart of top 20 stocks by composite centrality averaged across years |
| Cumulative returns | `cumulative_returns.png` | Phase 4+5 | Line chart, 5 strategies, 2011–2016 |
| Sharpe comparison | `sharpe_comparison.png` | Phase 4+5 | Bar chart, 5 strategies |
| Drawdown comparison | `drawdown_comparison.png` | Phase 4+5 | Bar chart of max drawdown per strategy |
| Regime switch overlay | `regime_portfolio_overlay.png` | Phase 5 | Regime timeline overlaid with portfolio return bars |

---

## Thesis Narrative / Results to Report

The implementing AI should ensure the code produces values that support this narrative in the thesis write-up.

**Module A result:** "DeepCNL topology features cluster years into three distinct market regimes with a silhouette score of [X]. The detected regimes align with [N]/7 known historical market conditions, including correctly identifying 2011 as a crisis year (European debt crisis) and 2013 as a bull year (+30% S&P 500)."

**Module B result:** "The DeepCNL Hub-Avoid portfolio achieves a Sharpe ratio of [X] versus [Y] for the PCC-based equivalent — demonstrating that DeepCNL's superior network quality translates directly into better portfolio construction. Both DeepCNL portfolios outperform the S&P 500 equal-weight benchmark (Sharpe [Z])."

**Combined result:** "The regime-aware adaptive portfolio, which switches between hub-avoid and hub-follow based on detected regime, achieves the highest Sharpe ratio of [X] and the lowest maximum drawdown of [Y]% — outperforming all static strategies and confirming that DeepCNL's structural insight is both descriptively accurate and financially actionable."

---

## Common Errors and How to Handle Them

| Error | Cause | Fix |
|---|---|---|
| `PowerIterationFailedConvergence` | Sparse graph, eigenvector centrality didn't converge | Catch exception, use `degree_centrality` as fallback for that year |
| `NetworkXError: Graph is not connected` in `avg_shortest_path_length` | Calling on full graph, not LCC | Always call on `get_lcc(g)` not on `g` directly |
| `KeyError: 'weight'` in betweenness | Graph has no edge weights | Use `nx.betweenness_centrality(g)` without `weight='weight'` when graph is unweighted |
| Empty portfolio (0 returns) | Ticker not in price data for that year | Filter `tickers` list against available symbols in price data before computing returns |
| Phase 4 look-ahead bias | Using year Y centrality to pick year Y portfolio | Always use year Y centrality → year Y+1 portfolio. This means portfolios only for 2011–2016 (not 2010). |
| `pkl` file not found | Phase 0 didn't complete | Check `outputs/graphs/` — resubmit failed cluster jobs |

---

## Notes for the Implementing AI

1. **All file paths** are relative to the project root (`Deep-Co-investment-Network-Learning-for-Financial-Assets/`). Use `os.path.join` everywhere.

2. **Do not modify** `stock_network_analysis.py` except for the patch described in Phase 0 (adding `save_graph()` calls). All MTP2 code lives in `mtp2/`.

3. **The `mtp2/` folder needs an `__init__.py`** so modules can be imported as `from mtp2.utils import load_graph`.

4. **Year coverage:** DeepCNL graphs cover 2010–2016 (seeds 0–6). Portfolio backtesting uses 2011–2016 (because 2010 centrality → 2011 portfolio is the first valid year). Regime detection uses all 7 years (2010–2016).

5. **PCC graphs** are saved alongside DeepCNL graphs in Phase 0 — they are used as the benchmark in Phase 4. The `Pearson_cor()` function already exists in `stock_network_analysis.py`.

6. **The price CSV** (`prices-split-adjusted.csv`) uses `adj_close` or `close` column — check the actual column names and use the right one in `utils.py`.

7. **Seaborn style** for all matplotlib figures: add `import seaborn as sns; sns.set_theme(style='whitegrid')` at the top of each phase file that produces figures.

8. **Risk-free rate** for Sharpe ratio is set at 2% annually (0.02). This is a reasonable approximation for US Treasury bills during 2010–2016.

9. **Equal weights assumption** keeps the backtesting simple and removes the need for an optimizer — this is intentional and academically defensible for an MTP. Mention it explicitly in the thesis.

10. **Web app** already uses Flask + Jinja2 templates + Chart.js + D3.js. The new `regime.html` and `portfolio.html` templates should follow the same structure as existing templates (extend `base.html`, use the same CSS classes).