import pandas as pd
import numpy as np

DATA_PATH = "/Users/ayushmunot/Deep-Co-investment-Network-Learning-for-Financial-Assets/Data/prices-split-adjusted.csv"

print("=" * 60)
print("DATA DIAGNOSTICS")
print("=" * 60)

# Load data
print("\n📂 Loading data...")
df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
print(f"✓ Loaded {len(df)} rows")

# Check for issues BEFORE filtering
print(f"\n⚠️  Data Quality Issues:")
print(f"  Duplicate dates: {df.index.duplicated().sum()}")
print(f"  Is sorted: {df.index.is_monotonic_increasing}")
print(f"  Missing values: {df.isnull().sum().sum()}")

# FIX: Sort and remove duplicates
print("\n🔧 Cleaning data...")
df = df.sort_index()
df = df[~df.index.duplicated(keep='first')]
print(f"✓ After cleaning: {len(df)} rows")

# Check date range
print(f"\n📅 Date Range:")
print(f"  Start: {df.index.min()}")
print(f"  End: {df.index.max()}")
print(f"  Total days: {(df.index.max() - df.index.min()).days}")

# Filter to 2011-2012 using boolean indexing (safer than .loc slicing)
print(f"\n📊 Filtering to 2011-2012...")
period_data = df[(df.index >= '2011-01-01') & (df.index <= '2012-12-31')]

if period_data.empty:
    print("❌ ERROR: No data found for 2011-2012!")
    print("\nAvailable date ranges by year:")
    for year in range(df.index.min().year, df.index.max().year + 1):
        year_data = df[df.index.year == year]
        if len(year_data) > 0:
            print(f"  {year}: {year_data.index.min()} to {year_data.index.max()} ({len(year_data)} rows)")
else:
    print(f"✓ Found {len(period_data)} rows")
    print(f"  Date range: {period_data.index.min()} to {period_data.index.max()}")
    print(f"  Unique dates: {len(period_data.index.unique())}")
    
    # Check if 'symbol' column exists
    if 'symbol' in period_data.columns:
        print(f"  Unique tickers: {len(period_data['symbol'].unique())}")
        
        # Check tickers
        print(f"\n🏢 Available Tickers in 2011-2012 (first 30):")
        tickers = sorted(period_data['symbol'].unique())
        for i, ticker in enumerate(tickers[:30]):
            ticker_data = period_data[period_data['symbol'] == ticker]
            print(f"  {i+1:2d}. {ticker:6s}: {len(ticker_data):4d} rows | "
                  f"{ticker_data.index.min().strftime('%Y-%m-%d')} to {ticker_data.index.max().strftime('%Y-%m-%d')}")
        
        if len(tickers) > 30:
            print(f"  ... and {len(tickers) - 30} more tickers")
        
        # Check data completeness
        print(f"\n📈 Data Completeness:")
        trading_days = len(period_data.index.unique())
        print(f"  Total trading days: {trading_days}")
        
        # Find tickers with full data
        complete_tickers = []
        for ticker in tickers[:20]:  # Check first 20
            ticker_data = period_data[period_data['symbol'] == ticker]
            coverage = len(ticker_data) / trading_days * 100
            if coverage > 90:
                complete_tickers.append(ticker)
            if len(tickers) <= 20 or ticker in tickers[:10]:  # Show details for first 10
                print(f"  {ticker:6s}: {len(ticker_data):4d}/{trading_days} days ({coverage:.1f}% coverage)")
        
        print(f"\n✓ Tickers with >90% coverage: {len(complete_tickers)}")
        
        # Check for required columns
        print(f"\n📋 Column Check:")
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol']
        for col in required_cols:
            status = "✓" if col in period_data.columns else "❌"
            print(f"  {status} {col}")
    else:
        print("\n❌ ERROR: 'symbol' column not found in data!")
        print(f"Available columns: {list(period_data.columns)}")

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

if period_data.empty:
    print("❌ No data available for 2011-2012")
    print("   → Check if DATA_PATH points to correct file")
    print("   → Verify CSV file contains data for these years")
else:
    if 'symbol' in period_data.columns:
        ticker_count = len(period_data['symbol'].unique())
        if ticker_count < 470:
            print(f"⚠️  Only {ticker_count} tickers available (requested 470)")
            print(f"   → Adjust TICKER_NUM in stock_network_analysis.py to {ticker_count}")
        else:
            print(f"✓ Sufficient tickers available ({ticker_count} tickers)")
        
        # Check if enough trading days
        if trading_days < 480:
            print(f"⚠️  Only {trading_days} trading days (need 480+ for WINDOW=32)")
            print(f"   → Consider reducing WINDOW size or using different date range")
        else:
            print(f"✓ Sufficient trading days ({trading_days} days)")
    else:
        print("❌ Data format issue: Missing 'symbol' column")
        print("   → Check CSV file structure")

print("\n" + "=" * 60)