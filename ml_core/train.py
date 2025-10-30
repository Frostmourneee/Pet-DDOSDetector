from typing import Any, Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from storage import get_storage_client


def get_df() -> pd.DataFrame:
    """Загрузка датасета из HDFS или скачивание из источника."""
    storage = get_storage_client()
    hdfs_path = "/user/airflow/datasets/adult_income/adult.data"

    if storage.file_exists(hdfs_path):
        return storage.read_csv(hdfs_path)
    else:
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        )
        column_names = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        ]
        df = pd.read_csv(
            url, header=None, names=column_names, na_values=" ?", skipinitialspace=True
        )

        storage.write_csv(df, hdfs_path)

        return df


def train_models() -> List[Dict[str, Any]]:
    """Основная функция обучения моделей.

    Returns:
        Список словарей с метриками для каждой обученной модели
    """
    RANDOM_STATE = 42

    storage = get_storage_client()
    models_hdfs_path = "/user/airflow/models"
    if storage.file_exists(f"{models_hdfs_path}/preprocessor.pkl"):
        print("Модели уже существуют в HDFS. Пропускаем обучение.")

        metrics_df = storage.read_csv(f"{models_hdfs_path}/training_metrics.csv")
        print("Метрики существующих моделей:")
        print(metrics_df)

        return metrics_df.to_dict("records")

    print("Модели не найдены в HDFS. Начинаем обучение...")

    df = get_df()

    X = df.drop("income", axis=1)
    y = df["income"].str.strip()
    y = (y == ">50K").astype(int)

    cat_features = X.select_dtypes(include=["category", "object"]).columns
    num_features = X.select_dtypes(exclude=["category", "object"]).columns

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
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
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-score": f1,
            }
        )

    metrics_df = pd.DataFrame(results)

    storage = get_storage_client()
    models_hdfs_path = "/user/airflow/models"
    storage.write_csv(metrics_df, f"{models_hdfs_path}/training_metrics.csv")

    storage.write_joblib(preprocessor, f"{models_hdfs_path}/preprocessor.pkl")

    for name, model in models.items():
        filename = name.replace(" ", "_").lower() + "_model.pkl"
        storage.write_joblib(model, f"{models_hdfs_path}/{filename}")

    print("Тренировка моделей окончена! Метрики:")
    print(metrics_df)

    return results
