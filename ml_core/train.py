import pandas as pd
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


def train_models():
    """Основная функция обучения моделей"""
    RANDOM_STATE = 42

    os.makedirs("/opt/airflow/ml_core/trained_models", exist_ok=True)
    models_path = "/opt/airflow/ml_core/trained_models"

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    df = pd.read_csv(url, header=None, names=column_names, na_values=' ?', skipinitialspace=True)

    X = df.drop('income', axis=1)
    y = df['income'].str.strip()
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
        "XGBoost": XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, use_label_encoder=False,
                                 eval_metric="logloss", n_jobs=-1)
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
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
        })

    joblib.dump(preprocessor, f"{models_path}/preprocessor.pkl")
    for name, model in models.items():
        joblib.dump(model, f"{models_path}/{name.replace(' ', '_').lower()}_model.pkl")

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(f"{models_path}/training_metrics.csv", index=False)

    print("Тренировка моделей окончена! Метрики:")
    print(metrics_df)

    return results