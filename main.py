import pandas as pd

LIVE_FILE = "v5.1/live.parquet"
live = pd.read_parquet(LIVE_FILE)
print(live.columns.tolist())