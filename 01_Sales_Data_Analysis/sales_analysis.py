"""Sales Data Analysis and Visualization
Run: python sales_analysis.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA = Path(__file__).parent / "data" / "sales_data.csv"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["order_date"])
df["revenue"] = df["quantity"] * df["unit_price"]

print("\n--- Dataset preview ---")
print(df.head())
print("\n--- Dataset information ---")
print(df.info())
print("\n--- Summary statistics ---")
print(df[["quantity", "unit_price", "revenue"]].describe())

monthly = df.groupby(df["order_date"].dt.to_period("M"))["revenue"].sum()
monthly.index = monthly.index.astype(str)
monthly.to_csv(OUT / "monthly_revenue.csv", header=["revenue"])

plt.figure(figsize=(9, 5))
monthly.plot(marker="o")
plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUT / "monthly_revenue.png", dpi=150)
plt.close()

category = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=category.index, y=category.values)
plt.title("Revenue by Product Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(OUT / "revenue_by_category.png", dpi=150)
plt.close()

region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
print("\n--- Revenue by region ---")
print(region)
print("\nSaved charts and summary CSV in outputs/.")
