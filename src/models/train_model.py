import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, average_precision_score
from xgboost import XGBClassifier
import joblib

FEATURES = [
    "log_amount",
    "velocity_1h",
    "velocity_24h",
    "geo_mismatch",
    "email_domain_risk",
    "synthetic_identity_flag",
    "user_degree",
    "device_degree",
    "ip_degree",
]


def train(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    pr_auc = average_precision_score(y_test, probs)

    joblib.dump(model, "artifacts/models/fraud_model.pkl")

    return {
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
    }
