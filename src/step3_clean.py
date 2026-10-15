import pandas as pd
import numpy as np

df = pd.read_csv("data/clean_top20.csv", parse_dates=["date"])
df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
df["dow"] = df["date"].dt.dayofweek          # 0 = Monday ... 6 = Sunday

xmas = (df["date"].dt.month == 12) & (df["date"].dt.day == 25)
df.loc[xmas, "sales"] = np.nan

first_sale = df[df["sales"] > 0].groupby("item_id")["date"].min()
df = df[df["date"] >= df["item_id"].map(first_sale)].copy()

MIN_RUN = 7   # 7+ consecutive zero days = stockout, not real demand

def flag_runs(s):
    z = s.eq(0)
    run_id = (~z).cumsum()
    run_len = z.groupby(run_id).transform("sum")
    return z & (run_len >= MIN_RUN)

df["stockout"] = df.groupby("item_id")["sales"].transform(flag_runs)

# Show each stockout period
runs = df[df["stockout"]].copy()
runs["gap"] = (runs.groupby("item_id")["date"].diff() != pd.Timedelta(days=1)).cumsum()
periods = runs.groupby(["item_id", "gap"])["date"].agg(start="min", end="max", days="count")
print("\nStockout periods:\n", periods.reset_index(level=1, drop=True))

df["sales_raw"] = df["sales"]                       # keep the original
df.loc[df["stockout"], "sales"] = np.nan

typical = df.groupby(["item_id", "dow"])["sales"].transform("median")
df["sales"] = df["sales"].fillna(typical)

summary = df.groupby("item_id").agg(
    start=("date", "min"),
    days=("date", "count"),
    stockout_days=("stockout", "sum"),
)
print("\nPer-item summary:\n", summary)

df.to_csv("data/model_ready.csv", index=False)
print("\nSaved data/model_ready.csv:", df.shape)