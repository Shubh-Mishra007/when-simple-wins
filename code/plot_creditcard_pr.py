import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve

df = pd.read_csv("data/creditcard.csv")
X = df.drop("Class", axis=1); y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

models = {
    "LR": LogisticRegression(class_weight="balanced", max_iter=1000),
    "MLP": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500),
    "XGB": XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        random_state=42, eval_metric="logloss"
    )
}

plt.figure(figsize=(8,6))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    plt.plot(recall, precision, label=name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves on Credit Card Fraud Detection")
plt.legend()
plt.savefig("figs/pr_creditcard.png")
plt.show()
