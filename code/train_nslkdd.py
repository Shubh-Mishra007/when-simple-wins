"""
NSL-KDD Intrusion Detection experiment
Binary classification: normal vs attack
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Column names (fixed for NSL-KDD)
cols = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label","difficulty"
]

# Load data
train = pd.read_csv("data/nsl_kdd/KDDTrain+.txt", names=cols)
test = pd.read_csv("data/nsl_kdd/KDDTest+.txt", names=cols)

# Binary labels: normal vs attack
train["label"] = train["label"].apply(lambda x: 0 if x == "normal" else 1)
test["label"] = test["label"].apply(lambda x: 0 if x == "normal" else 1)

X_train = train.drop(["label", "difficulty"], axis=1)
y_train = train["label"]
X_test = test.drop(["label", "difficulty"], axis=1)
y_test = test["label"]

# Encode categorical features
for col in X_train.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        random_state=42,
        eval_metric="logloss"
    )
}

print("NSL-KDD Intrusion Detection Results")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred)
    f = f1_score(y_test, y_pred)
    print(f"{name}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")
