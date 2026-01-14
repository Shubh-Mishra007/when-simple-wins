"""
Credit Card Fraud Detection dataset experiment
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Load data
df = pd.read_csv("data/creditcard.csv")

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500),
    "XGBoost": XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        random_state=42, eval_metric="logloss"
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = (precision, recall, f1)

print("Credit Card Fraud Detection Results")
for k, v in results.items():
    print(f"{k}: Precision={v[0]:.3f}, Recall={v[1]:.3f}, F1={v[2]:.3f}")
