"""
Time Series Builder for Google Cluster Data - CORRECTED VERSION

This module converts discrete job events into continuous time series
of job arrival rates.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class TimeSeriesBuilder:
    """
    Builds time series from Google Cluster job events.
    
    Converts discrete job submission events into continuous time series
    of arrival rates, with optional temporal features.
    """
    
    def __init__(self, window: str = '5min'):
        """
        Initialize the builder.
        
        Args:
            window: Resampling window (e.g., '1min', '5min', '1h')
        """
        self.window = window
        logger.info(f"TimeSeriesBuilder initialized with window={window}")
    
    def build_arrival_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build time series of job arrival rates.
        
        Args:
            df: DataFrame with job arrivals (must have 'datetime' column)
                This should be the output from GoogleCluster2011Loader.load_all_arrivals()
            
        Returns:
            DataFrame with time series:
                - arrival_rate: Jobs per hour
                - job_count: Raw count per window
        """
        logger.info(f"Building arrival series from {len(df)} job arrivals...")
        
        if len(df) == 0:
            raise ValueError("No job arrivals found in data")
        
        if 'datetime' not in df.columns:
            raise ValueError("DataFrame must have 'datetime' column")
        
        # Ensure datetime is the index
        df_indexed = df.set_index('datetime')
        
        # Group by time window and count arrivals
        ts_counts = df_indexed.resample(self.window).size()
        ts_counts.name = 'job_count'
        
        # Convert to rate per hour
        window_hours = pd.Timedelta(self.window).total_seconds() / 3600
        ts_rate = ts_counts / window_hours
        
        # Create DataFrame with both metrics
        df_ts = pd.DataFrame({
            'arrival_rate': ts_rate,
            'job_count': ts_counts
        })
        
        # Fill missing values (windows with no events)
        df_ts = df_ts.fillna(0)
        
        logger.info(f"Created time series with {len(df_ts)} time steps")
        logger.info(f"Time range: {df_ts.index.min()} to {df_ts.index.max()}")
        
        # Sanity check: warn if date range seems wrong
        year_min = df_ts.index.min().year
        year_max = df_ts.index.max().year
        
        if year_min < 2000 or year_max > 2020:
            logger.warning(f"⚠️  Suspicious date range: {year_min} to {year_max}")
            logger.warning("    Expected dates around 2011. Check timestamp conversion!")
        else:
            logger.info(f"✓ Date range looks correct: {year_min} to {year_max}")
        
        return df_ts
    
    def add_temporal_features(self, df_ts: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features to time series.
        
        Args:
            df_ts: Time series DataFrame
            
        Returns:
            DataFrame with added columns:
                - hour: Hour of day (0-23)
                - day_of_week: Day of week (0=Monday, 6=Sunday)
                - is_weekend: Boolean indicator
                - is_business_hours: Boolean indicator (9am-5pm weekdays)
        """
        df_ts = df_ts.copy()
        
        # Extract temporal features
        df_ts['hour'] = df_ts.index.hour
        df_ts['day_of_week'] = df_ts.index.dayofweek
        df_ts['is_weekend'] = df_ts['day_of_week'] >= 5
        
        # Business hours: 9am-5pm on weekdays
        df_ts['is_business_hours'] = (
            (df_ts['hour'] >= 9) & 
            (df_ts['hour'] < 17) & 
            (~df_ts['is_weekend'])
        )
        
        logger.info("Added temporal features")
        return df_ts
    
    def add_statistical_features(self, df_ts: pd.DataFrame,
                                 windows: List[int] = [12, 24, 48]) -> pd.DataFrame:
        """
        Add rolling statistical features.
        
        Args:
            df_ts: Time series DataFrame
            windows: List of window sizes for rolling statistics
            
        Returns:
            DataFrame with added rolling statistics
        """
        df_ts = df_ts.copy()
        
        for window in windows:
            # Rolling mean
            df_ts[f'arrival_rate_mean_{window}'] = (
                df_ts['arrival_rate']
                .rolling(window=window, min_periods=1)
                .mean()
            )
            
            # Rolling std
            df_ts[f'arrival_rate_std_{window}'] = (
                df_ts['arrival_rate']
                .rolling(window=window, min_periods=1)
                .std()
                .fillna(0)
            )
            
            # Rolling max
            df_ts[f'arrival_rate_max_{window}'] = (
                df_ts['arrival_rate']
                .rolling(window=window, min_periods=1)
                .max()
            )
            
            # Rolling min
            df_ts[f'arrival_rate_min_{window}'] = (
                df_ts['arrival_rate']
                .rolling(window=window, min_periods=1)
                .min()
            )
        
        logger.info(f"Added rolling statistics for windows: {windows}")
        return df_ts
    
    def add_lag_features(self, df_ts: pd.DataFrame,
                        lags: List[int] = [1, 2, 6, 12]) -> pd.DataFrame:
        """
        Add lagged features.
        
        Args:
            df_ts: Time series DataFrame
            lags: List of lag values
            
        Returns:
            DataFrame with lagged features
        """
        df_ts = df_ts.copy()
        
        for lag in lags:
            df_ts[f'arrival_rate_lag_{lag}'] = (
                df_ts['arrival_rate'].shift(lag).fillna(0)
            )
        
        logger.info(f"Added lag features: {lags}")
        return df_ts
    
    def compute_statistics(self, df_ts: pd.DataFrame) -> Dict:
        """
        Compute statistics of the time series.
        
        Args:
            df_ts: Time series DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'length': len(df_ts),
            'mean_arrival_rate': df_ts['arrival_rate'].mean(),
            'std_arrival_rate': df_ts['arrival_rate'].std(),
            'min_arrival_rate': df_ts['arrival_rate'].min(),
            'max_arrival_rate': df_ts['arrival_rate'].max(),
            'median_arrival_rate': df_ts['arrival_rate'].median(),
            'total_jobs': df_ts['job_count'].sum(),
            'zero_arrivals_pct': (df_ts['arrival_rate'] == 0).mean() * 100
        }
        
        return stats
    
    def print_statistics(self, df_ts: pd.DataFrame):
        """Print time series statistics."""
        stats = self.compute_statistics(df_ts)
        
        print("\n" + "="*70)
        print("TIME SERIES STATISTICS")
        print("="*70)
        print(f"Length:              {stats['length']:,} time steps")
        print(f"Total Jobs:          {stats['total_jobs']:,.0f}")
        print(f"\nArrival Rate (jobs/hour):")
        print(f"  Mean:              {stats['mean_arrival_rate']:.2f}")
        print(f"  Std:               {stats['std_arrival_rate']:.2f}")
        print(f"  Min:               {stats['min_arrival_rate']:.2f}")
        print(f"  Max:               {stats['max_arrival_rate']:.2f}")
        print(f"  Median:            {stats['median_arrival_rate']:.2f}")
        print(f"\nZero Arrivals:       {stats['zero_arrivals_pct']:.1f}%")
        print("="*70 + "\n")
    
    def save_to_csv(self, df_ts: pd.DataFrame, filepath: str):
        """Save time series to CSV file."""
        df_ts.to_csv(filepath)
        logger.info(f"Time series saved to {filepath}")
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """Load time series from CSV file."""
        df_ts = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Time series loaded from {filepath}")
        return df_ts


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('.')
    from src.data.loader import GoogleCluster2011Loader
    
    print("Time Series Builder - Example Usage\n")
    
    try:
        import matplotlib.pyplot as plt

        # Load data
        data_dir = "data/raw/2011"
        loader = GoogleCluster2011Loader(data_dir)
        
        # Load arrivals from first file
        print("Loading arrival data...")
        df_arrivals = loader.load_all_arrivals(max_files=500)
        
        # Show data info
        print(f"\nLoaded {len(df_arrivals):,} arrivals")
        print(f"Date range: {df_arrivals['datetime'].min()} to {df_arrivals['datetime'].max()}")
        
        # Build time series
        builder = TimeSeriesBuilder(window='5min')
        df_ts = builder.build_arrival_series(df_arrivals)
        
        # Add features
        df_ts = builder.add_temporal_features(df_ts)
        df_ts = builder.add_statistical_features(df_ts, windows=[12, 24])
        
        # Save time series to CSV
        output_file = "data/processed/time_series_dt5min.csv"
        builder.save_to_csv(df_ts, output_file)
        
        # Show statistics
        builder.print_statistics(df_ts)
        
        # Show sample
        print("Time series sample (first 20 rows):")
        print(df_ts[['arrival_rate', 'job_count', 'hour', 'day_of_week']].head(20))
        
        # Visualize
        try:
    
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            
            # Plot arrival rate
            df_ts['arrival_rate'].plot(ax=axes[0], 
                                      title='Job Arrival Rate Over Time')
            axes[0].set_ylabel('Jobs/hour')
            axes[0].grid(True, alpha=0.3)
            
            # Plot distribution
            df_ts['arrival_rate'].hist(bins=50, ax=axes[1])
            axes[1].set_title('Arrival Rate Distribution')
            axes[1].set_xlabel('Jobs/hour')
            axes[1].set_ylabel('Frequency')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('time_series_corrected.png', dpi=150)
            print("\n✅ Plot saved to: time_series_corrected.png")
            
        except ImportError:
            print("\nMatplotlib not available for visualization")
            
    except FileNotFoundError:
        print("Data directory not found. Please check the path.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()