"""House Price Prediction using Regression
Run: python house_price_prediction.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data" / "house_prices.csv")
X = df[["area_sqft", "bedrooms", "age_years", "location"]]
y = df["price_lakhs"]

numeric = ["area_sqft", "bedrooms", "age_years"]
categorical = ["location"]
preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])
model = make_pipeline(preprocess, LinearRegression())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)
print(f"MAE (lakhs): {mae:.2f}")
print(f"RMSE (lakhs): {rmse:.2f}")
print(f"R-squared: {r2:.3f}")

plt.figure(figsize=(7, 5))
plt.scatter(y_test, pred, alpha=0.8)
lo = min(y_test.min(), pred.min())
hi = max(y_test.max(), pred.max())
plt.plot([lo, hi], [lo, hi], linestyle="--")
plt.xlabel("Actual price (lakhs)")
plt.ylabel("Predicted price (lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=150)
plt.close()

pd.DataFrame({"actual_price_lakhs": y_test, "predicted_price_lakhs": pred}).to_csv(
    OUT / "predictions.csv", index=False)
print("Saved predictions and chart in outputs/.")
