import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings("ignore")

L          = 3       # lead time (days) from DC to store
S          = 5.0     # ordering cost ($ per order per SKU)
HOLD_RATE  = 0.30    # annual holding cost = 30% of unit cost
COST_RATIO = 0.70    # unit cost = 70% of selling price (30% margin)
Z          = 1.645   # 95% cycle service level
MAX_DAYS   = 14      # max order size = 14 days of demand (shelf life / space)
DROP = ["FOODS_3_541", "FOODS_3_635", "FOODS_3_808"]   # discontinued

hist = pd.read_csv("data/model_ready.csv", parse_dates=["date"]).sort_values(["item_id", "date"])
fc   = pd.read_csv("outputs/forecasts.csv", parse_dates=["date"]).sort_values(["item_id", "date"])
hist = hist[~hist["item_id"].isin(DROP)]
fc   = fc[~fc["item_id"].isin(DROP)]

TEST_START = fc["date"].min()
TRAIN_FROM = pd.Timestamp("2014-01-01")
VAL_START  = TEST_START - pd.Timedelta(days=28)

sigma = {}
for item, g in hist[hist["date"] < TEST_START].groupby("item_id"):
    tr = g[(g["date"] >= TRAIN_FROM) & (g["date"] < VAL_START)].set_index("date")["sales"].asfreq("D")
    va = g[g["date"] >= VAL_START]
    f = ExponentialSmoothing(tr, trend="add", damped_trend=True, seasonal="add",
                             seasonal_periods=7).fit().forecast(28).clip(lower=0).values
    real = ~va["stockout"].values
    err = va["sales_raw"].values[real] - f[real]
    sigma[item] = np.sqrt(np.mean(err ** 2)) if real.sum() >= 7 else tr.std()

params = {}
for item, g in fc.groupby("item_id"):
    price   = g["sell_price"].mean()
    cost    = COST_RATIO * price
    h_year  = HOLD_RATE * cost
    d_daily = g["f_ens"].mean()
    eoq     = math.sqrt(2 * d_daily * 365 * S / h_year)
    last7   = hist[(hist["item_id"] == item) & (hist["date"] < TEST_START)]["sales"].tail(7)
    params[item] = dict(
        price=price, margin=price - cost, h_daily=h_year / 365,
        sigma=sigma[item], EOQ=eoq,
        Q=math.ceil(min(eoq, MAX_DAYS * d_daily)),
        SS=math.ceil(Z * sigma[item] * math.sqrt(L)),
        start_inv=round(last7.sum()), last7=list(last7),
    )

ptab = pd.DataFrame(params).T[["price", "sigma", "EOQ", "Q", "SS"]].astype(float).round(1)
print("Policy parameters:\n", ptab)

def simulate(demand, fcast, p, policy):
    on_hand, pipeline = p["start_inv"], []           # pipeline = [(arrival_day, qty)]
    past_sales = list(p["last7"])
    r = dict(hold=0.0, order=0.0, lost_cost=0.0, lost=0.0, demand=0.0,
             orders=0, so_days=0, inv=[])
    for t in range(len(demand)):
        on_hand += sum(q for d, q in pipeline if d == t)          # receive deliveries
        pipeline = [(d, q) for d, q in pipeline if d != t]

        sold = min(on_hand, demand[t]); lost = demand[t] - sold   # serve customers
        on_hand -= sold; past_sales.append(sold)

        ip = on_hand + sum(q for _, q in pipeline)                # inventory position
        q = 0
        if policy == "baseline" and t % 7 == 0:                  # Monday: reorder last week's sales
            q = sum(past_sales[-7:])
        elif policy == "optimized":
            lt = fcast[t + 1:t + 1 + L]
            lt = np.append(lt, [fcast.mean()] * (L - len(lt)))
            if ip <= lt.sum() + p["SS"]:                          # hit reorder point R
                q = p["Q"]
        if q > 0:
            pipeline.append((t + L, q)); r["orders"] += 1; r["order"] += S

        r["hold"] += on_hand * p["h_daily"]
        r["lost"] += lost; r["lost_cost"] += lost * p["margin"]
        r["demand"] += demand[t]; r["so_days"] += lost > 0
        r["inv"].append(on_hand)
    r["total"] = r["hold"] + r["order"] + r["lost_cost"]
    r["fill_rate"] = 1 - r["lost"] / r["demand"]
    return r

rows, sims = [], {}
for item, g in fc.groupby("item_id"):
    demand, fcast, p = g["sales_raw"].values, g["f_ens"].values, params[item]
    for pol in ["baseline", "optimized"]:
        r = simulate(demand, fcast, p, pol)
        sims[(item, pol)] = r
        rows.append([item, pol, r["total"], r["hold"], r["order"], r["lost_cost"],
                     r["fill_rate"] * 100, r["so_days"], np.mean(r["inv"]), r["orders"]])

res = pd.DataFrame(rows, columns=["item_id", "policy", "total_cost", "holding", "ordering",
                                  "stockout_cost", "fill_rate_%", "stockout_days",
                                  "avg_inventory", "orders"])
tot = res.groupby("policy").agg(total_cost=("total_cost", "sum"), holding=("holding", "sum"),
                                ordering=("ordering", "sum"), stockout_cost=("stockout_cost", "sum"),
                                fill_rate=("fill_rate_%", "mean"), stockout_days=("stockout_days", "sum"),
                                avg_inventory=("avg_inventory", "sum"), orders=("orders", "sum")).round(2)
print("\nTotals across 17 SKUs (28 days):\n", tot)

b, o = tot.loc["baseline"], tot.loc["optimized"]
print(f"\nTotal cost:     {(1 - o.total_cost / b.total_cost) * 100:.1f}% lower")
print(f"Fill rate:      {b.fill_rate:.1f}% -> {o.fill_rate:.1f}%")
print(f"Stockout days:  {int(b.stockout_days)} -> {int(o.stockout_days)}")
print(f"Avg inventory:  {(o.avg_inventory / b.avg_inventory - 1) * 100:+.1f}%")

ptab.to_csv("outputs/inventory_policy.csv")
res.round(2).to_csv("outputs/inventory_results.csv", index=False)

item = "FOODS_3_090"
dates = fc[fc["item_id"] == item]["date"]
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(dates, fc[fc["item_id"] == item]["sales_raw"], color="lightgrey", label="Daily demand")
ax.plot(dates, sims[(item, "baseline")]["inv"], "r-o", ms=3, label="Baseline: on-hand")
ax.plot(dates, sims[(item, "optimized")]["inv"], "g-o", ms=3, label="Optimized (Q,R): on-hand")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title(f"{item}: inventory simulation, baseline vs optimized")
ax.legend(); plt.xticks(rotation=45); plt.tight_layout()
plt.savefig("outputs/05_inventory_sim.png"); plt.close()
print("\nSaved inventory_policy.csv, inventory_results.csv, 05_inventory_sim.png")

sigma_naive = {}
for item, g in hist[hist["date"] < TEST_START].groupby("item_id"):
    va = g[g["date"] >= VAL_START]
    lastwk = g[(g["date"] < VAL_START) & (g["date"] >= VAL_START - pd.Timedelta(days=7))]
    f = va["dow"].map(lastwk.set_index("dow")["sales"]).values
    real = ~va["stockout"].values
    err = va["sales_raw"].values[real] - f[real]
    sigma_naive[item] = np.sqrt(np.mean(err ** 2)) if real.sum() >= 7 else g["sales"].std()

ss_naive = sum(math.ceil(Z * sigma_naive[i] * math.sqrt(L)) for i in params)
ss_ens   = sum(math.ceil(Z * sigma[i] * math.sqrt(L)) for i in params)
print(f"\nSafety stock @95%: naive {ss_naive} units | model {ss_ens} units "
      f"| {(1 - ss_ens / ss_naive) * 100:.1f}% less")

avg28 = hist[hist["date"] < TEST_START].groupby("item_id")["sales"].apply(lambda s: s.tail(28).mean())
Z_LIST = [0.0, 0.5, 0.84, 1.28, 1.645, 2.05, 2.33]

rows = []
for name, col, sig in [("Naive", "f_naive", sigma_naive), ("Ensemble", "f_ens", sigma)]:
    for z in Z_LIST:
        lost = dem = inv = 0.0
        for item, g in fc.groupby("item_id"):
            p = dict(params[item])
            p["SS"] = math.ceil(z * sig[item] * math.sqrt(L))
            p["Q"] = math.ceil(7 * avg28[item])                 # same weekly Q for both
            p["start_inv"] = round(p["Q"] / 2 + p["SS"])         # start at steady state
            r = simulate(g["sales_raw"].values, g[col].values, p, "optimized")
            lost += r["lost"]; dem += r["demand"]; inv += np.mean(r["inv"])
        rows.append([name, z, (1 - lost / dem) * 100, inv])

fr = pd.DataFrame(rows, columns=["forecast", "z", "fill_rate_%", "avg_inventory"]).round(2)
print("\nService-level sweep:\n", fr)
fr.to_csv("outputs/service_frontier.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
for name, g in fr.groupby("forecast"):
    ax.plot(g["avg_inventory"], g["fill_rate_%"], "-o", label=f"{name} forecast")
ax.set_xlabel("Average inventory held (units, 17 SKUs)")
ax.set_ylabel("Fill rate (%)")
ax.set_title("Inventory vs service: naive vs ensemble forecast")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig("outputs/06_service_frontier.png"); plt.close()
print("Saved 06_service_frontier.png")