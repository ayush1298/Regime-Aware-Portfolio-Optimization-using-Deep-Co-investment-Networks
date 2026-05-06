import json
import random

with open('web_app/data/network_data.json', 'r') as f:
    data = json.load(f)

sectors = ['Financials', 'Technology', 'Healthcare', 'Energy', 'Consumer', 'Industrials']

for year in data:
    for node in data[year]['nodes']:
        # Assign random sectors (for demonstration) or map common ones
        if node['id'] in ['AAPL', 'MSFT', 'NVDA', 'AKAM']:
            node['sector'] = 'Technology'
        elif node['id'] in ['BAC', 'C', 'JPM', 'WFC', 'AIG', 'ZION']:
            node['sector'] = 'Financials'
        elif node['id'] in ['BSX', 'REGN', 'BIIB']:
            node['sector'] = 'Healthcare'
        elif node['id'] in ['XOM', 'COG', 'EP']:
            node['sector'] = 'Energy'
        else:
            node['sector'] = random.choice(sectors)
        
        # Mock metrics
        degree = node['degree']
        node['betweenness'] = round(random.uniform(0.01, 0.2) + degree*0.01, 4)
        node['eigenvector'] = round(random.uniform(0.1, 0.9), 4)
        
        if degree > 10:
            node['classification'] = random.choice(['Hub-Follow', 'Hub-Avoid'])
        else:
            node['classification'] = 'Peripheral'

with open('web_app/data/network_data.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Metrics added.")
