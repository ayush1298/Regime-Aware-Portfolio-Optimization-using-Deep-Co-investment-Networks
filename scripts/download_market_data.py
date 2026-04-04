"""
Download missing market data files for DeepCNL analysis
Only downloads files that don't already exist in the Data folder
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def check_file_exists(filepath):
    """Check if file exists and has data"""
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            if len(df) > 0:
                return True, len(df)
        except:
            pass
    return False, 0

def download_spy_data(output_dir="Data"):
    """
    Download SPY (S&P 500 ETF) historical data if not present
    """
    file_path = os.path.join(output_dir, 'SPY20000101_20171111.csv')
    
    # Check if already exists
    exists, rows = check_file_exists(file_path)
    if exists:
        print(f"✅ SPY data already exists: {file_path} ({rows} rows)")
        return file_path
    
    print(f"📥 Downloading SPY data (2000-01-01 to 2017-11-12)...")
    
    # Create data directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Download data
        spy_data = yf.download("SPY", start="2000-01-01", end="2017-11-12", progress=True)
        
        # Validate data
        if spy_data.empty:
            raise ValueError("No data downloaded. Check internet connection.")
        
        # Save to CSV
        spy_data.to_csv(file_path)
        
        print(f"✅ SPY data saved!")
        print(f"   Location: {file_path}")
        print(f"   Rows: {len(spy_data)}")
        print(f"   Date range: {spy_data.index.min()} to {spy_data.index.max()}")
        
        return file_path
        
    except Exception as e:
        print(f"❌ Error downloading SPY data: {e}")
        return None


def download_sp500_index(output_dir="Data"):
    """
    Download S&P 500 Index (^GSPC) historical data if not present
    """
    file_path = os.path.join(output_dir, 'SP500^GSPC20000101_20171111.csv')
    
    # Check if already exists
    exists, rows = check_file_exists(file_path)
    if exists:
        print(f"✅ S&P 500 Index data already exists: {file_path} ({rows} rows)")
        return file_path
    
    print(f"📥 Downloading S&P 500 Index data (2000-01-01 to 2017-11-12)...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Download data
        sp500_data = yf.download("^GSPC", start="2000-01-01", end="2017-11-12", progress=True)
        
        if sp500_data.empty:
            raise ValueError("No data downloaded. Check internet connection.")
        
        # Save to CSV
        sp500_data.to_csv(file_path)
        
        print(f"✅ S&P 500 Index data saved!")
        print(f"   Location: {file_path}")
        print(f"   Rows: {len(sp500_data)}")
        print(f"   Date range: {sp500_data.index.min()} to {sp500_data.index.max()}")
        
        return file_path
        
    except Exception as e:
        print(f"❌ Error downloading S&P 500 data: {e}")
        return None


def verify_data_folder(output_dir="Data"):
    """Check status of all required data files"""
    print("\n🔍 Checking Data folder...")
    
    required_files = {
        'prices-split-adjusted.csv': 'Stock prices (DO NOT DOWNLOAD - already present)',
        'SPY20000101_20171111.csv': 'SPY ETF data',
        'SP500^GSPC20000101_20171111.csv': 'S&P 500 Index data'
    }
    
    status = {}
    
    for filename, description in required_files.items():
        filepath = os.path.join(output_dir, filename)
        exists, rows = check_file_exists(filepath)
        
        if exists:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            status[filename] = {
                'exists': True,
                'description': description,
                'rows': rows,
                'size': f"{size_mb:.2f} MB"
            }
            print(f"  ✅ {filename:<35} ({rows:,} rows, {size_mb:.2f} MB)")
        else:
            status[filename] = {
                'exists': False,
                'description': description,
                'rows': 0,
                'size': 'N/A'
            }
            if filename == 'prices-split-adjusted.csv':
                print(f"  ⚠️  {filename:<35} (NOT FOUND - Please add manually)")
            else:
                print(f"  ❌ {filename:<35} (MISSING - will download)")
    
    return status


if __name__ == "__main__":
    print("=" * 70)
    print("DeepCNL Market Data Downloader")
    print("Only downloads missing files - preserves existing data")
    print("=" * 70)
    
    # Check current status
    status = verify_data_folder()
    
    # Check if prices-split-adjusted.csv exists
    if not status['prices-split-adjusted.csv']['exists']:
        print("\n" + "=" * 70)
        print("⚠️  WARNING: prices-split-adjusted.csv not found!")
        print("=" * 70)
        print("This file should already be present in the Data/ folder.")
        print("Please ensure you have this file before running the analysis.")
        print("\nIf you need this file:")
        print("  1. Download from: https://www.kaggle.com/datasets/dgawlik/nyse")
        print("  2. Place it in: Data/prices-split-adjusted.csv")
        print("=" * 70)
    
    # Determine what needs downloading
    needs_download = []
    if not status['SPY20000101_20171111.csv']['exists']:
        needs_download.append('SPY')
    if not status['SP500^GSPC20000101_20171111.csv']['exists']:
        needs_download.append('S&P 500 Index')
    
    if not needs_download:
        print("\n✅ All market data files are present!")
        print("\nYou can now run: python stock_network_analysis.py")
        exit(0)
    
    # Ask for confirmation
    print(f"\n📋 Files to download: {', '.join(needs_download)}")
    response = input("\nProceed with download? (y/n): ")
    
    if response.lower() != 'y':
        print("Download cancelled.")
        exit(0)
    
    # Download missing files
    print("\n" + "=" * 70)
    print("Downloading Missing Market Data")
    print("=" * 70)
    
    results = {}
    
    # Download SPY if missing
    if not status['SPY20000101_20171111.csv']['exists']:
        results['SPY'] = download_spy_data()
    else:
        results['SPY'] = 'already_exists'
    
    # Download S&P 500 Index if missing
    if not status['SP500^GSPC20000101_20171111.csv']['exists']:
        results['SP500'] = download_sp500_index()
    else:
        results['SP500'] = 'already_exists'
    
    # Final summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    
    if results['SPY'] == 'already_exists':
        print("✅ SPY data: Already exists (skipped)")
    elif results['SPY']:
        print(f"✅ SPY data: Downloaded successfully")
    else:
        print("❌ SPY data: Failed to download")
    
    if results['SP500'] == 'already_exists':
        print("✅ S&P 500 data: Already exists (skipped)")
    elif results['SP500']:
        print(f"✅ S&P 500 data: Downloaded successfully")
    else:
        print("❌ S&P 500 data: Failed to download")
    
    # Check final status
    print("\n" + "=" * 70)
    print("Final Data Folder Status")
    print("=" * 70)
    verify_data_folder()
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Verify all data files are present in Data/ folder")
    print("2. Run: python stock_network_analysis.py")
    print("=" * 70)