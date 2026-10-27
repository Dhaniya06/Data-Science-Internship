"""Student Performance Analysis
Run: python student_analysis.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data" / "student_performance.csv")

print("--- Preview ---")
print(df.head())
print("\n--- Summary ---")
print(df.describe(numeric_only=True))
print("\nCorrelation matrix:")
print(df.select_dtypes("number").corr().round(2))

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="study_hours_per_week", y="exam_score", hue="attendance_pct")
plt.title("Study Hours and Exam Score")
plt.tight_layout()
plt.savefig(OUT / "study_hours_vs_score.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="exam_score", bins=10, kde=True)
plt.title("Distribution of Exam Scores")
plt.tight_layout()
plt.savefig(OUT / "exam_score_distribution.png", dpi=150)
plt.close()

df.groupby("class_level")["exam_score"].mean().sort_values(ascending=False).to_csv(
    OUT / "average_score_by_class.csv", header=["average_exam_score"])
print("\nSaved charts and average-score CSV in outputs/.")
