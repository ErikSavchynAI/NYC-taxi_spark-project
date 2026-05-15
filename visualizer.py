import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

warnings.filterwarnings("ignore")

CONFIG = {
    "erik_q1": {"x": "hour", "y": "count", "title": "Завантажені години (Вихідні)", "type": "bar"},
    "erik_q2": {"x": "passenger_count", "y": "avg_total", "title": "Середня вартість від кількості пасажирів",
                "type": "bar"},
    "erik_q6": {"x": "vendor_id", "y": "total_revenue", "title": "Найприбутковіша година провайдера", "type": "bar"},
    "denys_q1": {"x": "payment_type", "y": "total_revenue", "title": "Дохід за типом оплати", "type": "bar"},
    "denys_q2": {"x": "hack_license", "y": "count", "title": "Топ водіїв без чайових", "type": "bar"},
    "denys_q3": {"x": "vendor_id", "y": "avg_cost_per_mile", "title": "Середня вартість милі (>5 миль)", "type": "bar"},
    "denys_q4": {"x": "pickup_datetime", "y": "running_total", "title": "Накопичувальний заробіток водія",
                 "type": "line"},
    "denys_q6": {"x": "date", "y": "avg_daily", "title": "Динаміка середнього чеку", "type": "line"},
    "mark_q1": {"x": "has_tolls", "y": "avg_tip", "title": "Чайові: платні vs безкоштовні дороги", "type": "bar"},
    "mark_q3": {"x": "dist_cat", "y": "count", "title": "Розподіл поїздок за дистанцією", "type": "bar"},
    "mark_q6": {"x": "date", "y": "crd_percent", "title": "% оплат карткою за днями", "type": "line"},
    "sanya_q1": {"x": "medallion", "y": "avg_daily_rev", "title": "Середній дохід авто (>30 поїздок)", "type": "bar"},
    "sanya_q4": {"x": "time_hr", "y": "moving_avg_3h", "title": "Ковзне середнє поїздок (3 год)", "type": "line"},
}


def plot_query(df, name):
    os.makedirs("data/visualizations", exist_ok=True)
    pdf = df.toPandas()

    if pdf.empty or name not in CONFIG:
        return

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    cfg = CONFIG[name]

    if cfg["type"] == "bar":
        if len(pdf) > 20:
            pdf = pdf.head(20)
        sns.barplot(data=pdf, x=cfg["x"], y=cfg["y"], hue=cfg["x"], palette="viridis", legend=False)
        plt.xticks(rotation=45)
    elif cfg["type"] == "line":
        if "date" in cfg["x"] or "time" in cfg["x"]:
            pdf[cfg["x"]] = pd.to_datetime(pdf[cfg["x"]])
            pdf = pdf.sort_values(cfg["x"])
        sns.lineplot(data=pdf, x=cfg["x"], y=cfg["y"], marker="o", color="b")
        plt.xticks(rotation=45)

    plt.title(cfg["title"])
    plt.tight_layout()
    plt.savefig(f"data/visualizations/{name}.png", dpi=300)
    plt.close()
