import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib
import os

def main():
    os.makedirs("ai_model", exist_ok=True)
    RANDOM_STATE = 42


    data = fetch_openml("adult", version=2, as_frame=True)
    X, y = data.data, data.target
    y = (y == ">50K").astype(int)

    cat_features = X.select_dtypes(include=["category", "object"]).columns
    num_features = X.select_dtypes(exclude=["category", "object"]).columns

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                num_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="first", sparse_output=False)),
                    ]
                ),
                cat_features,
            ),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="logloss", n_jobs=-1)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (macro)": prec,
            "Recall (macro)": rec,
            "F1-score (macro)": f1,
        })

    joblib.dump(preprocessor, "ai_model/preprocessor.pkl")

    for name, model in models.items():
        joblib.dump(model, f"ai_model/{name.replace(' ', '_').lower()}_model.pkl")