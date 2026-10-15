import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/clean_top20.csv", parse_dates=["date"])
df = df.sort_values(["item_id", "date"])

daily = df.groupby("date")["sales"].sum()

plt.figure(figsize=(12, 4))
daily.plot(alpha=0.3, label="Daily sales")
daily.rolling(28).mean().plot(label="28-day average", linewidth=2)
plt.title("Total daily sales: top 20 FOODS_3 items, Store CA_1")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/01_trend.png"); plt.close()

order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["dow"] = df["date"].dt.day_name()
dow = df.groupby("dow")["sales"].mean().reindex(order)

dow.plot(kind="bar", figsize=(8, 4), title="Average sales by day of week")
plt.ylabel("Units/day per item"); plt.tight_layout()
plt.savefig("outputs/02_day_of_week.png"); plt.close()
print("\nAvg sales by weekday:\n", dow.round(1))

stats = df.groupby("item_id")["sales"].agg(mean="mean", std="std")
stats["cv"] = stats["std"] / stats["mean"]
stats["zero_pct"] = df.groupby("item_id")["sales"].apply(lambda s: (s == 0).mean() * 100)

def longest_zero_run(s):
    z = (s == 0)
    return int(z.groupby((~z).cumsum()).sum().max())

stats["max_zero_streak"] = df.groupby("item_id")["sales"].apply(longest_zero_run)
stats = stats.round(2).sort_values("cv")
stats.to_csv("outputs/item_stats.csv")
print("\nItem stats:\n", stats)

df["is_event"] = df["event_name_1"].notna()

snap = df.groupby("snap_CA")["sales"].mean()
event = df.groupby("is_event")["sales"].mean()

print(f"\nSNAP days avg: {snap[1]:.1f} | Non-SNAP: {snap[0]:.1f} | Lift: {(snap[1]/snap[0]-1)*100:.1f}%")
print(f"Event days avg: {event[True]:.1f} | Normal: {event[False]:.1f} | Lift: {(event[True]/event[False]-1)*100:.1f}%")

fig, axes = plt.subplots(5, 4, figsize=(18, 14), sharex=True)
for ax, (item, g) in zip(axes.flat, df.groupby("item_id")):
    ax.plot(g["date"], g["sales"].rolling(7).mean(), linewidth=0.8)
    ax.set_title(item, fontsize=9)
plt.suptitle("7-day avg sales per item"); plt.tight_layout()
plt.savefig("outputs/03_items_grid.png"); plt.close()
print("\nCharts saved in outputs/")