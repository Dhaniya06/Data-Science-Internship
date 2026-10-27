"""Iris Flower Classification
Run: python iris_classification.py
Uses the Iris dataset bundled with scikit-learn.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
df = X.copy()
df["species"] = y.map(dict(enumerate(iris.target_names)))
df.to_csv(OUT / "iris_dataset_export.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"Test accuracy: {accuracy_score(y_test, pred):.3f}")
print("\nClassification report:")
print(classification_report(y_test, pred, target_names=iris.target_names))

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=iris.target_names)
plt.title("Iris Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=150)
plt.close()

print("\nExample prediction for first test row:")
print("Predicted:", iris.target_names[model.predict(X_test.iloc[[0]])[0]])
print("Actual:", iris.target_names[y_test.iloc[0]])
print("Saved dataset export and confusion matrix in outputs/.")
