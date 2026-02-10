"""
Google Cluster 2011 Data Loader - CORRECTED VERSION

This loader properly handles the Google Cluster timestamp format.
The timestamps are in MICROSECONDS since epoch, not seconds.
"""

import pandas as pd
import gzip
from pathlib import Path
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleCluster2011Loader:
    """
    Loads Google Cluster 2011 trace data.
    
    IMPORTANT: Timestamps in the data are in MICROSECONDS, not seconds!
    """
    
    COLUMNS = [
        'timestamp',           # Microseconds since epoch
        'missing_info',
        'job_id',
        'event_type',
        'user',
        'scheduling_class',
        'job_name',
        'logical_job_name'
    ]
    
    EVENT_TYPES = {
        0: 'SUBMIT',
        1: 'SCHEDULE',
        2: 'EVICT',
        3: 'FAIL',
        4: 'FINISH',
        5: 'KILL',
        6: 'LOST',
        7: 'UPDATE_PENDING',
        8: 'UPDATE_RUNNING'
    }
    
    def __init__(self, data_dir: str):
        """
        Initialize the loader.
        
        Args:
            data_dir: Path to directory with part-*.csv.gz files
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        self.files = sorted(self.data_dir.glob('part-*.csv.gz'))
        logger.info(f"Found {len(self.files)} files in {data_dir}")
    
    def load_file(self, filepath: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Load a single file.
        
        Args:
            filepath: Path to .csv.gz file
            max_rows: Maximum rows to load (None = all)
            
        Returns:
            DataFrame with parsed data
        """
        try:
            with gzip.open(filepath, 'rt') as f:
                df = pd.read_csv(
                    f, 
                    names=self.COLUMNS, 
                    header=None,
                    nrows=max_rows
                )
            
            # Convert timestamp from microseconds to datetime
            # CRITICAL: unit='us' for microseconds!
            TRACE_START = pd.Timestamp('2011-05-01 00:00:00')
            time_delta = pd.to_timedelta(df['timestamp'], unit='us')
            df['datetime'] = TRACE_START + time_delta  # ✓ Dates correctes !
            
            # Add event name
            df['event_name'] = df['event_type'].map(self.EVENT_TYPES)
            
            # Remove rows with invalid timestamps
            df = df.dropna(subset=['datetime'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise
    
    def load_arrivals(self, filepath: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Load only SUBMIT events (job arrivals) from a file.
        
        Args:
            filepath: Path to file
            max_rows: Max rows to read from file
            
        Returns:
            DataFrame with only SUBMIT events
        """
        df = self.load_file(filepath, max_rows)
        arrivals = df[df['event_type'] == 0].copy()
        arrivals = arrivals.sort_values('datetime')
        
        return arrivals[['job_id', 'timestamp', 'datetime', 'user']]
    
    def load_all_arrivals(self, max_files: Optional[int] = None,
                         max_rows_per_file: Optional[int] = None) -> pd.DataFrame:
        """
        Load arrivals from multiple files.
        
        Args:
            max_files: Maximum number of files to load
            max_rows_per_file: Max rows per file
            
        Returns:
            Combined DataFrame with all arrivals
        """
        files_to_load = self.files[:max_files] if max_files else self.files
        
        logger.info(f"Processing {len(files_to_load)} files...")
        
        all_arrivals = []
        
        for i, filepath in enumerate(files_to_load, 1):
            logger.info(f"[{i}/{len(files_to_load)}] Processing {filepath.name}")
            
            arrivals = self.load_arrivals(filepath, max_rows_per_file)
            
            if len(arrivals) > 0:
                all_arrivals.append(arrivals)
                logger.info(f"  Found {len(arrivals)} job arrivals")
        
        if not all_arrivals:
            raise ValueError("No arrivals found in any file")
        
        logger.info("Consolidating all arrivals...")
        df_all = pd.concat(all_arrivals, ignore_index=True)
        df_all = df_all.sort_values('datetime').reset_index(drop=True)
        
        logger.info(f"Total arrivals: {len(df_all):,}")
        logger.info(f"Date range: {df_all['datetime'].min()} to {df_all['datetime'].max()}")
        
        return df_all
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """Get statistics about loaded data."""
        stats = {
            'total_arrivals': len(df),
            'unique_jobs': df['job_id'].nunique(),
            'unique_users': df['user'].nunique() if 'user' in df.columns else 0,
            'time_range': {
                'start': df['datetime'].min(),
                'end': df['datetime'].max(),
                'duration': df['datetime'].max() - df['datetime'].min()
            }
        }
        return stats
    
    def print_statistics(self, df: pd.DataFrame):
        """Print statistics."""
        stats = self.get_statistics(df)
        
        print("\n" + "="*70)
        print("GOOGLE CLUSTER DATA - ARRIVALS")
        print("="*70)
        print(f"Total Arrivals:    {stats['total_arrivals']:,}")
        print(f"Unique Jobs:       {stats['unique_jobs']:,}")
        print(f"Unique Users:      {stats['unique_users']:,}")
        print(f"\nTime Range:")
        print(f"  Start:           {stats['time_range']['start']}")
        print(f"  End:             {stats['time_range']['end']}")
        print(f"  Duration:        {stats['time_range']['duration']}")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Test the loader
    print("Google Cluster 2011 Loader - Test\n")
    
    data_dir = "data/raw/2011"
    
    try:
        loader = GoogleCluster2011Loader(data_dir)
        
        # Load arrivals from first file
        print("Loading arrivals from first file...\n")
        df = loader.load_all_arrivals(max_files=10)
        
        # Show statistics
        loader.print_statistics(df)
        
        # Show sample
        print("Sample arrivals:")
        print(df[['job_id', 'datetime']].head(10))
        
        # Check timestamp range
        print(f"\nTimestamp validation:")
        print(f"  Min datetime: {df['datetime'].min()}")
        print(f"  Max datetime: {df['datetime'].max()}")
        print(f"  Year range: {df['datetime'].dt.year.min()} to {df['datetime'].dt.year.max()}")
        
        # Expected: dates should be around 2011
        if df['datetime'].dt.year.min() < 2000 or df['datetime'].dt.year.max() > 2020:
            print("\n⚠️  WARNING: Timestamps seem incorrect!")
            print("    Expected dates around May 2011 (29 days)")
        else:
            print("\n✅ Timestamps look correct!")
        
    except FileNotFoundError:
        print(f"Data directory not found: {data_dir}")
        print("Please ensure Google Cluster data is in the correct location.")
    except Exception as e:
        print(f"Error: {e}")