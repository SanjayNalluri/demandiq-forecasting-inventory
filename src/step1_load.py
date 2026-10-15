import pandas as pd

sales  = pd.read_csv("data/sales_train_evaluation.csv")
cal    = pd.read_csv("data/calendar.csv")
prices = pd.read_csv("data/sell_prices.csv")

# Scope: Store CA_1, FOODS_3 dept
sub = sales[(sales.store_id == "CA_1") & (sales.dept_id == "FOODS_3")].copy()
d_cols = [c for c in sub.columns if c.startswith("d_")]

# Top 20 items by total units sold
sub["total"] = sub[d_cols].sum(axis=1)
top = sub.nlargest(20, "total")

# Wide -> long (one row per item per day)
long = top.melt(id_vars=["item_id"], value_vars=d_cols,
                var_name="d", value_name="sales")

# Add dates, events, SNAP flag, prices
long = long.merge(cal[["d", "date", "wm_yr_wk", "event_name_1", "snap_CA"]], on="d")
long["date"] = pd.to_datetime(long["date"])
p = prices[prices.store_id == "CA_1"][["item_id", "wm_yr_wk", "sell_price"]]
long = long.merge(p, on=["item_id", "wm_yr_wk"], how="left")

long.to_csv("data/clean_top20.csv", index=False)
print(long.shape)
print(long.groupby("item_id")["sales"].mean().round(1).sort_values())