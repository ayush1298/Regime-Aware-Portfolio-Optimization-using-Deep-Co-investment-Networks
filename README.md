# Deep Co-Investment Network Learning (DeepCNL)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DeepCNL is an end-to-end deep learning framework designed to extract dynamic structural relationships in financial markets. It leverages a Convolutional Recurrent Neural Network (CRNN) to learn co-investment networks from S&P 500 stock pairs and applies topological clustering to construct highly adaptive portfolio strategies.

---

## 🚀 Key Results
1. **Regime Detection**: DeepCNL graphs automatically organize market states into structural regimes (Bull/Crisis/Recovery) via K-means analysis with a high alignment to known macro-events. High network *assortativity* is cleanly separated from fragmented markets.
2. **Defensive Safety**: The DeepCNL `Hub-Avoid` portfolio mathematically identified systemic contagion risks, suppressing the maximum drawdown to a staggering **-0.5%** during the 2011–2016 continuous backtest period (vs S&P 500's -1.9%).
3. **PCC Outperformance**: DeepCNL significantly out-predicted standard Pearson Correlation Coefficient (PCC) networks in risk-adjusted performance (Sharpe 0.93 vs 0.57).

## 📂 Repository Architecture

The codebase has been refactored cleanly into distinct modules bridging the structural extraction and subsequent financial backtesting:

```text
Deep-Co-investment-Network-Learning/
├── data/                        # Raw pricing matrices and SPY index data
├── docs/                        # Papers, original specs, and supplementary material
├── outputs/                     # Generated networks (.pkl), charts, and JSON data
├── scripts/                     # Launchers and cluster deployment wrappers
├── web_app/                     # Interactive Flask frontend dashboard
└── src/                         # Core Python Logic
    ├── models/                  # Neural network layered architectures (CRNN, LSTM)
    ├── mtp1/                    # Module A: DeepCNL structural graph generator
    ├── mtp2/                    # Module B: Machine learning & portfolio backtesting
    └── utils/                   # Shared pipeline utilities
```

## ⚙️ Quickstart

### 1. Environment Setup
Install the necessary computational dependencies:
```bash
pip install -r requirements.txt
```

### 2. View the Live Dashboard
This repository ships with a fully responsive frontend dashboard summarizing the financial networks and simulation results.
```bash
python web_app/app.py
```
*Navigate to `http://127.0.0.1:5000` in your browser.*

---

## 📖 Deep Dives & Documentation

For algorithmic breakdowns, mathematical formulations, and step-by-step implementations of the ML phases, refer to our detailed documentation:
* [Architecture Synopsis](docs/Architecture_Summary.md): A concise but highly detailed breakdown of the Unsupervised Regime Detection system and the Adaptive Centrality Backtester.