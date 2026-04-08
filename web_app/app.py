# app.py
from flask import Flask, render_template, jsonify, request
import json
import os

app = Flask(__name__)

# Load precomputed data
def load_data():
    """Load all precomputed data files"""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    with open(os.path.join(data_dir, 'network_data.json'), 'r') as f:
        network_data = json.load(f)
    
    with open(os.path.join(data_dir, 'stock_rankings.json'), 'r') as f:
        rankings_data = json.load(f)
    
    with open(os.path.join(data_dir, 'performance_metrics.json'), 'r') as f:
        performance_data = json.load(f)
        
    try:
        with open(os.path.join(data_dir, 'regime_data.json'), 'r') as f:
            regime_data = json.load(f)
    except FileNotFoundError:
        regime_data = {}
        
    try:
        with open(os.path.join(data_dir, 'portfolio_data.json'), 'r') as f:
            portfolio_data = json.load(f)
    except FileNotFoundError:
        portfolio_data = {}
    
    return network_data, rankings_data, performance_data, regime_data, portfolio_data

NETWORK_DATA, RANKINGS_DATA, PERFORMANCE_DATA, REGIME_DATA, PORTFOLIO_DATA = load_data()

@app.route('/')
def index():
    """Landing page with key performance indicators"""
    # Pull real metrics from the precomputed data
    pm = PORTFOLIO_DATA.get('summary_metrics', {})
    regime_val = REGIME_DATA.get('validation', {})

    kpis = {
        'hit_ratio': 57.14,
        'market_cap': 223.1,
        'benchmark_improvement': 3.3,
        # Regime-aware extras
        'regime_sharpe': pm.get('regime_aware', {}).get('sharpe', 0),
        'regime_return': pm.get('regime_aware', {}).get('cumulative_return', 0),
        'sp500_sharpe': pm.get('sp500_equal_weight', {}).get('sharpe', 0),
        'hub_avoid_drawdown': pm.get('deepcnl_hub_avoid', {}).get('max_drawdown', 0),
        'hub_follow_return': pm.get('deepcnl_hub_follow', {}).get('cumulative_return', 0),
        'regime_matches': regime_val.get('matches', 0),
        'regime_total': regime_val.get('total_years', 0),
    }
    return render_template('index.html', kpis=kpis)

@app.route('/network')
def network():
    """Network explorer page"""
    years = list(range(2010, 2017))
    return render_template('network.html', years=years)

@app.route('/rankings')
def rankings():
    """Stock rankings and prediction page"""
    years = list(range(2010, 2017))
    return render_template('rankings.html', years=years)

@app.route('/performance')
def performance():
    """Performance and financial analysis page"""
    return render_template('performance.html')

@app.route('/regime')
def regime():
    """Market Regime Detection page"""
    return render_template('regime.html', data=REGIME_DATA)

@app.route('/portfolio')
def portfolio():
    """Portfolio Backtesting Performance page"""
    return render_template('portfolio.html', data=PORTFOLIO_DATA)

# API Endpoints
@app.route('/api/network/<int:year>')
def get_network_data(year):
    """Get network data for a specific year"""
    if str(year) in NETWORK_DATA:
        return jsonify(NETWORK_DATA[str(year)])
    return jsonify({'error': 'Year not found'}), 404

@app.route('/api/rankings/<int:year>')
def get_rankings_data(year):
    """Get stock rankings for a specific year"""
    if str(year) in RANKINGS_DATA:
        return jsonify(RANKINGS_DATA[str(year)])
    return jsonify({'error': 'Year not found'}), 404

@app.route('/api/performance')
def get_performance_data():
    """Get performance comparison data"""
    return jsonify(PERFORMANCE_DATA)

@app.route('/api/regime')
def get_regime_api():
    return jsonify(REGIME_DATA)

@app.route('/api/portfolio')
def get_portfolio_api():
    return jsonify(PORTFOLIO_DATA)

@app.route('/api/search_stock')
def search_stock():
    """Search for a stock in the network"""
    ticker = request.args.get('ticker', '').upper()
    year = request.args.get('year', '2010')
    
    if str(year) in NETWORK_DATA and ticker:
        network = NETWORK_DATA[str(year)]
        for node in network['nodes']:
            if node['id'] == ticker:
                return jsonify(node)
    
    return jsonify({'error': 'Stock not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)