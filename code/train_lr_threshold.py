"""
Logistic Regression with threshold tuning
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# Dataset
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=5,
    n_redundant=2,
    weights=[0.95, 0.05],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
model.fit(X_train, y_train)

# Probabilities instead of predictions
y_prob = model.predict_proba(X_test)[:, 1]

# Tuned threshold
threshold = 0.3
y_pred = (y_prob >= threshold).astype(int)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Logistic Regression (Threshold Tuned)")
print(f"Threshold: {threshold}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")
