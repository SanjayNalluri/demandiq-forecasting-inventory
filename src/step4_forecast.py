import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

df = pd.read_csv("data/model_ready.csv", parse_dates=["date"])
df = df.sort_values(["item_id", "date"]).reset_index(drop=True)

H = 28                                           # forecast horizon (days)
TEST_START = df["date"].max() - pd.Timedelta(days=H - 1)
TRAIN_FROM = pd.Timestamp("2014-01-01")
test = df[df["date"] >= TEST_START].copy()
print("Test period:", TEST_START.date(), "to", df["date"].max().date())

def score(g, col):
    real = ~g["stockout"]                        # only days with real demand
    a, f = g.loc[real, "sales_raw"], g.loc[real, col]
    wape = (a - f).abs().sum() / a.sum() * 100
    rmse = np.sqrt(((a - f) ** 2).mean())
    bias = (f - a).sum() / a.sum() * 100
    return wape, rmse, bias

last_week = df[(df["date"] < TEST_START) & (df["date"] >= TEST_START - pd.Timedelta(days=7))]
naive = last_week.set_index(["item_id", "dow"])["sales"]
test["f_naive"] = naive.reindex(list(zip(test["item_id"], test["dow"]))).values

ets_out = []
for item, g in df[(df["date"] >= TRAIN_FROM) & (df["date"] < TEST_START)].groupby("item_id"):
    y = g.set_index("date")["sales"].asfreq("D")
    fit = ExponentialSmoothing(y, trend="add", damped_trend=True,
                               seasonal="add", seasonal_periods=7).fit()
    fc = fit.forecast(H)
    ets_out.append(pd.DataFrame({"item_id": item, "date": fc.index, "f_ets": fc.values}))

test = test.merge(pd.concat(ets_out), on=["item_id", "date"])
test["f_ets"] = test["f_ets"].clip(lower=0)

def add_features(d):
    d = d.copy()
    g = d.groupby("item_id")["sales"]
    for L in [28, 35, 42, 49, 364]:
        d[f"lag_{L}"] = g.shift(L)
    d["roll_mean_7"]  = g.transform(lambda s: s.shift(28).rolling(7).mean())
    d["roll_mean_28"] = g.transform(lambda s: s.shift(28).rolling(28).mean())
    d["roll_std_28"]  = g.transform(lambda s: s.shift(28).rolling(28).std())
    d["month"] = d["date"].dt.month
    d["dom"] = d["date"].dt.day
    d["is_event"] = d["event_name_1"].notna().astype(int)
    prev_price = d.groupby("item_id")["sell_price"].shift(1)
    d["price_change"] = (d["sell_price"] / prev_price - 1).fillna(0)
    d["item_code"] = d["item_id"].astype("category").cat.codes
    return d

FEATS = ["item_code", "dow", "month", "dom", "snap_CA", "is_event",
         "sell_price", "price_change", "lag_28", "lag_35", "lag_42",
         "lag_49", "lag_364", "roll_mean_7", "roll_mean_28", "roll_std_28"]

feat = add_features(df)
train = feat[(feat["date"] >= TRAIN_FROM) & (feat["date"] < TEST_START) & (~feat["stockout"])]

xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                   subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb.fit(train[FEATS], train["sales"])

tf = feat[feat["date"] >= TEST_START].copy()
tf["f_xgb"] = xgb.predict(tf[FEATS]).clip(0)
test = test.merge(tf[["item_id", "date", "f_xgb"]], on=["item_id", "date"])

models = {"Seasonal naive": "f_naive", "ETS (Holt-Winters)": "f_ets", "XGBoost": "f_xgb"}
res = pd.DataFrame([[name, *score(test, col)] for name, col in models.items()],
                   columns=["Model", "WAPE_%", "RMSE", "Bias_%"]).round(2)
print(f"\nScored on {(~test['stockout']).sum()} real-demand days of {len(test)}")
print("\n", res)

best = res.loc[res["WAPE_%"].idxmin(), "Model"]
gain = (1 - res["WAPE_%"].min() / res.loc[0, "WAPE_%"]) * 100
print(f"\nBest model: {best} | WAPE {gain:.1f}% lower than naive")

per_item = test.groupby("item_id").apply(lambda g: pd.Series(
    {c: score(g, c)[0] for c in models.values()})).round(1)
print("\nWAPE % per item:\n", per_item)

test.to_csv("outputs/forecasts.csv", index=False)

items = ["FOODS_3_586", "FOODS_3_090", "FOODS_3_252", "FOODS_3_318"]
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, item in zip(axes.flat, items):
    g = test[test["item_id"] == item]
    ax.plot(g["date"], g["sales_raw"], "k-o", ms=3, label="Actual")
    for name, col in models.items():
        ax.plot(g["date"], g[col], label=name)
    ax.set_title(item); ax.tick_params(axis="x", rotation=45)
axes[0, 0].legend()
plt.tight_layout(); plt.savefig("outputs/04_forecast_vs_actual.png"); plt.close()
print("\nSaved outputs/forecasts.csv and 04_forecast_vs_actual.png")

# Ensemble: simple average of ETS and XGBoost
test["f_ens"] = (test["f_ets"] + test["f_xgb"]) / 2
models["Ensemble (ETS+XGB)"] = "f_ens"

res = pd.DataFrame([[name, *score(test, col)] for name, col in models.items()],
                   columns=["Model", "WAPE_%", "RMSE", "Bias_%"]).round(2)
print("\nWith ensemble:\n", res)
test.to_csv("outputs/forecasts.csv", index=False)