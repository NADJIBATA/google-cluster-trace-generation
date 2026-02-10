"""
Quick Test Script - Verify Timestamp Correction

This script tests the corrected loader and time series builder.
"""

import sys
sys.path.append('.')

from src.data.loader import GoogleCluster2011Loader
from src.preprocessing.time_series_builder import TimeSeriesBuilder

def main():
    print("="*70)
    print("TESTING CORRECTED LOADER AND TIME SERIES BUILDER")
    print("="*70)
    
    data_dir = "data/raw/2011"
    
    try:
        # Step 1: Load data with corrected loader
        print("\n📂 Step 1: Loading data with corrected loader...")
        loader = GoogleCluster2011Loader(data_dir)
        
        # Load from first file only for quick test
        df_arrivals = loader.load_all_arrivals(max_files=1)
        
        # Show statistics
        loader.print_statistics(df_arrivals)
        
        # Step 2: Check timestamp validity
        print("\n🕐 Step 2: Validating timestamps...")
        year_min = df_arrivals['datetime'].dt.year.min()
        year_max = df_arrivals['datetime'].dt.year.max()
        
        print(f"Year range: {year_min} to {year_max}")
        
        if 2010 <= year_min <= 2012 and 2010 <= year_max <= 2012:
            print("✅ Timestamps look CORRECT! (Expected: May 2011)")
        else:
            print("⚠️  WARNING: Timestamps still look wrong!")
            print("    Expected years around 2011")
            return
        
        # Step 3: Build time series
        print("\n📈 Step 3: Building time series...")
        builder = TimeSeriesBuilder(window='5min')
        df_ts = builder.build_arrival_series(df_arrivals)
        
        # Add features
        df_ts = builder.add_temporal_features(df_ts)
        
        # Show statistics
        builder.print_statistics(df_ts)
        
        # Step 4: Show sample data
        print("📊 Step 4: Sample time series data:")
        print("\nFirst 10 time steps:")
        print(df_ts[['arrival_rate', 'job_count', 'hour']].head(10))
        
        print("\nLast 10 time steps:")
        print(df_ts[['arrival_rate', 'job_count', 'hour']].tail(10))
        
        # Step 5: Basic validation
        print("\n✓ Step 5: Validation checks...")
        
        total_arrivals = df_ts['job_count'].sum()
        expected_arrivals = len(df_arrivals)
        
        print(f"Total arrivals in original data: {expected_arrivals:,}")
        print(f"Total arrivals in time series:   {int(total_arrivals):,}")
        
        if total_arrivals == expected_arrivals:
            print("✅ Arrival counts match!")
        else:
            print(f"⚠️  Mismatch: {abs(total_arrivals - expected_arrivals)} arrivals")
        
        # Check for reasonable values
        max_rate = df_ts['arrival_rate'].max()
        mean_rate = df_ts['arrival_rate'].mean()
        
        print(f"\nArrival rate statistics:")
        print(f"  Mean:  {mean_rate:.2f} jobs/hour")
        print(f"  Max:   {max_rate:.2f} jobs/hour")
        
        if max_rate > 100000:
            print("⚠️  Very high arrival rates detected - this might indicate an issue")
        else:
            print("✅ Arrival rates look reasonable")
        
        # Final summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        if (2010 <= year_min <= 2012 and 
            total_arrivals == expected_arrivals and 
            max_rate < 100000):
            print("✅ ALL CHECKS PASSED!")
            print("\nThe corrected loader is working properly.")
            print("You can now use:")
            print("  - src.data.loader.GoogleCluster2011Loader")
            print("  - src.preprocessing.time_series_builder_corrected.TimeSeriesBuilder")
        else:
            print("⚠️  SOME CHECKS FAILED")
            print("\nPlease review the output above for details.")
        
        print("="*70)
        
    except FileNotFoundError:
        print(f"\n❌ Error: Data directory not found: {data_dir}")
        print("\nPlease ensure Google Cluster data files are in:")
        print(f"  {data_dir}/part-*.csv.gz")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()