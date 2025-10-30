import joblib
import pandas as pd

MODELS = {}
PREPROCESSOR = None
MODELS_LOADED = False

def are_models_loaded():
    return MODELS_LOADED

def load_models():
    global PREPROCESSOR, MODELS, MODELS_LOADED

    if MODELS_LOADED:
        return True

    try:
        models_path = "/ml_core/trained_models"
        PREPROCESSOR = joblib.load(f"{models_path}/preprocessor.pkl")

        model_names = ["Logistic Regression", "Random Forest", "XGBoost"]
        for name in model_names:
            filename = name.replace(' ', '_').lower() + "_model.pkl"
            MODELS[name] = joblib.load(f"{models_path}/{filename}")

        MODELS_LOADED = True
        print("Модели успешно загружены")
        return True
    except Exception as e:
        print(f"Ошибка при загрузке моделей: {e}")
        return False


def predict(input_data: dict, model_name: str = "Random Forest"):
    """Предсказание дохода"""
    if model_name not in MODELS:
        raise ValueError(f"Модель '{model_name}' не найдена. Доступные: {list(MODELS.keys())}")

    df = pd.DataFrame([input_data])
    X_transformed = PREPROCESSOR.transform(df)
    model = MODELS[model_name]

    result_proba = model.predict_proba(X_transformed)[0][1]
    result = model.predict(X_transformed)[0]

    return {
        "model_used": model_name,
        "predict": int(result),
        "proba": float(result_proba)
    }