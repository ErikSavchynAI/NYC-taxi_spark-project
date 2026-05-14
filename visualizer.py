import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_query(df, name):
    os.makedirs("data/visualizations", exist_ok=True)
    pdf = df.toPandas()

    if pdf.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    if name == "q1_avg_tip":
        sns.barplot(data=pdf, x="passenger_count", y="avg_tip", palette="viridis")
        plt.title("Середній розмір чайових залежно від кількості пасажирів")
        plt.xlabel("Кількість пасажирів")
        plt.ylabel("Середні чайові ($)")

    elif name == "q3_revenue_vendor":
        sns.barplot(data=pdf, x="vendor_id", y="total_revenue", palette="Set2")
        plt.title("Сумарний дохід за постачальником послуг")
        plt.xlabel("Постачальник")
        plt.ylabel("Дохід ($)")

    elif name == "q4_daily_ma":
        pdf['date'] = pd.to_datetime(pdf['date'])
        pdf = pdf.sort_values('date')
        plt.plot(pdf['date'], pdf['daily_revenue'], label='Щоденний дохід', marker='o')
        plt.plot(pdf['date'], pdf['3_day_moving_avg'], label='Ковзне середнє (3 дні)', linestyle='--')
        plt.title("Динаміка доходу таксі")
        plt.xlabel("Дата")
        plt.ylabel("Дохід ($)")
        plt.xticks(rotation=45)
        plt.legend()

    else:
        plt.close()
        return

    plt.tight_layout()
    plt.savefig(f"data/visualizations/{name}.png", dpi=300)
    plt.close()
