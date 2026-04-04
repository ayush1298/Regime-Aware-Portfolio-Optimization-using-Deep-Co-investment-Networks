from sklearn import preprocessing
import pandas as pd
import numpy as np
import itertools

class Data_util:

    def __init__(self, ticker_num, window, feature_num, input_path, target_path):
        self.dict_dyadic = {}
        self.dict_ticker = {}
        self.ticker_num = ticker_num
        self.input_path = input_path
        self.target_path = target_path
        self.window = window
        self.feature_num = feature_num
        self.compare_data = None
        self.groups = None
        self.actual_ticker_num = ticker_num  # Track actual vs requested

    def check_dyadic(self, i):
        return self.dict_dyadic[i]

    def check_ticker(self, i):
        return self.dict_ticker[i]

    def read_data(self, fname):
        """Read the raw csv data as a pandas dataframe"""
        df = pd.read_csv(fname, index_col=0, parse_dates=True, date_format='ISO8601')
        df = df.sort_index()
        df.columns = df.columns.str.lower()

        if 'symbol' in df.columns:
            # Multi-ticker file: deduplicate by (date, symbol) pairs
            df = df[~df.reset_index().duplicated(subset=[df.index.name or 'date', 'symbol'], keep='first').values]
        else:
            # Single-ticker file: deduplicate by date only
            df = df[~df.index.duplicated(keep='first')]

        # Handle both 'close' and 'adj close' columns
        if 'adj close' not in df.columns:
            if 'close' in df.columns:
                df["adj close"] = df['close']
            else:
                raise ValueError(f"Required column 'close' or 'adj close' not found. Available columns: {list(df.columns)}")
        else:
            # If adj close exists but we still need close for the copy
            if 'close' not in df.columns:
                df['close'] = df['adj close']
        
        # CRITICAL FIX: Ensure all numeric columns are properly typed
        numeric_columns = ['high', 'low', 'volume', 'adj close', 'close']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN in critical columns
        if 'adj close' in df.columns:
            df = df.dropna(subset=['adj close'])
        
        # Drop columns if they exist
        if 'open' in df.columns:
            df.drop(['open'], axis=1, inplace=True)
        if 'close' in df.columns and 'adj close' in df.columns:
            df.drop(['close'], axis=1, inplace=True)

        return df

    def normalize_data(self, df):
        """Normalize the data with min-max scaler for each feature"""
        min_max_scaler = preprocessing.MinMaxScaler()
        if 'high' in df.columns:
            df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1, 1))
        if 'low' in df.columns:
            df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1, 1))
        if 'volume' in df.columns:
            df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1, 1))
        if 'adj close' in df.columns:
            df['adj close'] = min_max_scaler.fit_transform(df['adj close'].values.reshape(-1, 1))
        return df
    
    def get_full_index(self, groups):
        """Get the index with maximum data points"""
        maxindexnumber = groups.count().max()['volume']
        for g in groups:
            if g[1].count()['volume'] == maxindexnumber:
                return g[1].index
        return None

    def get_groups(self):
        if self.groups is not None:
            data = self.groups
            data = data[data.shape[0], data.shape[1], 0]
            outputs = []
            all_edge_combinations = [c for c in itertools.combinations(range(self.actual_ticker_num), 2)]
            for (i, j) in all_edge_combinations:
                outputs.append(np.concatenate(data[(i, j), :, :], axis=1))

    def group_select(self, df):
        """Group data by tickers and align time series - FIXED to handle missing data"""
        # Get available tickers sorted
        available_tickers = sorted(list(set(df.symbol)))
        
        print(f"📊 Available tickers in date range: {len(available_tickers)}")
        
        # Filter tickers by data completeness
        groups = df.groupby('symbol', sort=True)
        fullindex = self.get_full_index(groups)
        
        if fullindex is None:
            raise ValueError("Could not determine full time index")
        
        # Calculate required timesteps
        total_timesteps = len(fullindex)
        remainder = total_timesteps % self.window
        required_timesteps = total_timesteps - remainder if remainder != 0 else total_timesteps
        
        print(f"📅 Total trading days: {total_timesteps}")
        print(f"📅 Required timesteps (after window alignment): {required_timesteps}")
        
        # Filter tickers with sufficient data (at least 80% coverage)
        min_required_rows = int(required_timesteps * 0.8)
        valid_tickers = []
        
        for ticker in available_tickers:
            ticker_data = df[df.symbol == ticker]
            if len(ticker_data) >= min_required_rows:
                valid_tickers.append(ticker)
        
        print(f"✅ Tickers with ≥80% data coverage: {len(valid_tickers)}")
        
        if len(valid_tickers) == 0:
            raise ValueError(f"No tickers have sufficient data (need ≥{min_required_rows} rows)")
        
        # Use only tickers with complete data, up to requested amount
        tickers_to_use = valid_tickers[:self.ticker_num]
        self.actual_ticker_num = len(tickers_to_use)
        
        print(f"🎯 Using {self.actual_ticker_num} tickers (requested {self.ticker_num})")
        
        if self.actual_ticker_num < 3:
            raise ValueError(f"Insufficient tickers: need at least 3, found {self.actual_ticker_num}")
        
        # Filter dataframe to selected tickers
        df = df[df.symbol.isin(tickers_to_use)]
        
        # Build result array
        result = []
        groups = df.groupby('symbol', sort=True)
        
        ticker_idx = 0
        for ticker_name, group_df in groups:
            group_df = group_df.copy()
            group_df.sort_index()
            self.dict_ticker[ticker_idx] = ticker_name
            ticker_idx += 1

            # Align time series
            if len(group_df.index) < len(fullindex):
                group_df = group_df.reindex(fullindex)
                # Fix: Explicitly convert object columns before fillna
                for col in group_df.select_dtypes(include=['object']).columns:
                    group_df[col] = pd.to_numeric(group_df[col], errors='coerce')
                group_df = group_df.fillna(0)
            
            # Extract data
            length = 0
            for item in group_df.drop('symbol', axis=1).values:
                result.append(item)
                length += 1
                if length >= required_timesteps:
                    break

        result = np.array(result)
        
        # Validate shape
        expected_size = self.actual_ticker_num * required_timesteps * self.feature_num
        actual_size = result.size
        
        print(f"📦 Data shape validation:")
        print(f"   Expected: {self.actual_ticker_num} × {required_timesteps} × {self.feature_num} = {expected_size}")
        print(f"   Actual: {actual_size}")
        
        if expected_size != actual_size:
            raise ValueError(
                f"Data shape mismatch!\n"
                f"  Expected: {self.actual_ticker_num} tickers × {required_timesteps} steps × {self.feature_num} features = {expected_size}\n"
                f"  Actual: {actual_size}\n"
                f"  Missing: {expected_size - actual_size} elements"
            )
        
        return result.reshape(self.actual_ticker_num, required_timesteps, self.feature_num)

    def timeseries_enumerate(self, data):
        """Enumerate all combinations of time series from data"""
        outputs = []
        out_comparison = []
        all_edge_combinations = [c for c in itertools.combinations(range(self.actual_ticker_num), 2)]
        
        print(f"🔗 Generating {len(all_edge_combinations)} pairwise combinations from {self.actual_ticker_num} tickers")
        
        c = 0
        for (i, j) in all_edge_combinations:
            self.dict_dyadic[c] = (i, j)
            c += 1
            outputs.append(np.concatenate(data[(i, j), :, :], axis=1).transpose())
            out_comparison.append(data[(i, j), :, 3])  # close price
        
        outputs = np.array(outputs)
        out_comparison = np.array(out_comparison)
        self.compare_data = out_comparison.reshape(len(all_edge_combinations), 2, out_comparison.shape[2])
        
        return outputs

    def load_x(self, period):
        """Load input data for given period"""
        print("📥 Loading input data...")
        data = self.read_data(self.input_path)
        start, end = period
        
        if not isinstance(start, str):
            start = str(start)
        if not isinstance(end, str):
            end = str(end)
        
        print(f"📅 Period: {start} to {end}")
        
        data = data[(data.index >= start) & (data.index <= end)]
        
        if data.empty:
            raise ValueError(f"No data found for period {start} to {end}")
        
        print(f"📊 Total rows in period: {len(data)}")
        
        data = self.normalize_data(data)
        return self.timeseries_enumerate(self.group_select(data))

    def load_y(self, period):
        """Load target data for given period"""
        print("📥 Loading target data...")
        data = self.read_data(self.target_path)
        start, end = period
        
        if not isinstance(start, str):
            start = str(start)
        if not isinstance(end, str):
            end = str(end)
        
        data = data[(data.index >= start) & (data.index <= end)]
        
        if data.empty:
            raise ValueError(f"No data found for period {start} to {end}")
        
        print(f"📊 Total rows for target: {len(data)}")
        
        # CRITICAL FIX: Ensure adj close is numeric before any operations
        if 'adj close' in data.columns:
            data['adj close'] = pd.to_numeric(data['adj close'], errors='coerce')
            # Drop any rows where conversion failed
            data = data.dropna(subset=['adj close'])
        
        if 'symbol' in data.columns:
            # Group by date index and sum across all tickers
            daily_sum = data.groupby(data.index)['adj close'].sum()
        else:
            # Single series
            daily_sum = data['adj close']
        
        print(f"📊 Daily sum data type: {daily_sum.dtype}")
        print(f"📊 Daily sum shape: {len(daily_sum)}")
        
        # Calculate daily percentage change
        daily_pct_change = daily_sum.pct_change()
        ys = daily_pct_change.fillna(0).values
        
        # Convert to binary: 1 if positive return, 0 otherwise
        ys = np.array([1 if i > 0 else 0 for i in ys])
        
        print(f"📊 Total daily returns: {len(ys)}")
    
        # Align with window size
        remainder = int(len(ys)) % self.window
        if remainder != 0:
            ys = ys[0:int(int(len(ys) / self.window) * self.window)]
        
        # Skip first window (no prediction for initial period)
        ys = ys[self.window:]
        
        print(f"📊 Final target labels: {len(ys)} (after window alignment)")
        
        return ys