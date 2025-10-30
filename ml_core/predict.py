from typing import Any, Dict, Optional

import pandas as pd

from storage import get_storage_client

MODELS: Dict[str, Any] = {}
PREPROCESSOR: Optional[Any] = None
MODELS_LOADED: bool = False


def are_models_loaded() -> bool:
    """Проверка загружены ли модели в память."""
    return MODELS_LOADED


def load_models() -> bool:
    """Загрузка моделей и препроцессора из хранилища.

    Returns:
        True если модели успешно загружены, False в случае ошибки
    """
    global PREPROCESSOR, MODELS, MODELS_LOADED

    if MODELS_LOADED:
        return True

    try:
        storage = get_storage_client()
        models_hdfs_path = "/user/airflow/models"
        PREPROCESSOR = storage.read_joblib(f"{models_hdfs_path}/preprocessor.pkl")

        model_names = ["Logistic Regression", "Random Forest", "XGBoost"]
        for name in model_names:
            filename = name.replace(" ", "_").lower() + "_model.pkl"
            MODELS[name] = storage.read_joblib(f"{models_hdfs_path}/{filename}")

        MODELS_LOADED = True
        print("Модели успешно загружены")
        return True
    except Exception as e:
        print(f"Ошибка при загрузке моделей: {e}")
        return False


def predict(
    input_data: Dict[str, Any], model_name: str = "Random Forest"
) -> Dict[str, Any]:
    """Предсказание дохода на основе входных данных.

    Args:
        input_data: Входные данные в виде словаря
        model_name: Название модели для предсказания

    Returns:
        Словарь с результатами предсказания

    Raises:
        ValueError: Если указанная модель не найдена
    """
    if model_name not in MODELS:
        raise ValueError(
            f"Модель '{model_name}' не найдена. Доступные: {list(MODELS.keys())}"
        )

    df = pd.DataFrame([input_data])
    X_transformed = PREPROCESSOR.transform(df)
    model = MODELS[model_name]

    result_proba = model.predict_proba(X_transformed)[0][1]
    result = model.predict(X_transformed)[0]

    return {
        "model_used": model_name,
        "predict": int(result),
        "proba": float(result_proba),
    }
