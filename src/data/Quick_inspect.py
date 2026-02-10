import gzip
import pandas as pd

filepath = 'data/raw/2011/part-00000-of-00500.csv.gz'

print("First 5 raw lines from the file:")
print("="*70)
with gzip.open(filepath, 'rt') as f:
    for i in range(10000):
        line = f.readline().strip()
        parts = line.split(',')
        print(f"Line {i}:")
        print(f"  Timestamp (raw): {parts[0]}")
        print(f"  Event type: {parts[3]}")
        print()

# Load and check timestamp column
COLUMNS = ['timestamp', 'missing_info', 'job_id', 'event_type', 
           'user', 'scheduling_class', 'job_name', 'logical_job_name']

with gzip.open(filepath, 'rt') as f:
    df = pd.read_csv(f, names=COLUMNS, header=None, nrows=20)

print("\nTimestamp column (first 10 SUBMIT events):")
print("="*70)
df_submit = df[df['event_type'] == 0].head(10000)
print(df_submit[['timestamp', 'event_type']].to_string())

print(f"\nTimestamp range:")
print(f"  Min: {df_submit['timestamp'].min()}")
print(f"  Max: {df_submit['timestamp'].max()}")