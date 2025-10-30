from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd

from fastapi_app.server.schemas.predict import PredictRequest, PredictResponse
from sklearn.exceptions import NotFittedError

api_router = APIRouter(tags=["Prediction"])

MODELS = {}
PREPROCESSOR = None


def load_models():
    global PREPROCESSOR
    PREPROCESSOR = joblib.load("fastapi_app/ai_model/preprocessor.pkl")

    model_names = ["Logistic Regression", "Random Forest", "XGBoost"]
    for name in model_names:
        filename = name.replace(' ', '_').lower() + "_model.pkl"
        MODELS[name] = joblib.load(f"fastapi_app/ai_model/{filename}")


load_models()


@api_router.post("/predict", response_model=PredictResponse)
async def predict_income(input_data: PredictRequest, model_name: str = "Random Forest"):
    if model_name not in MODELS:
        raise HTTPException(status_code=400,
                            detail=f"Модель '{model_name}' не найдена. Доступные: {list(MODELS.keys())}")

    try:
        input_object = input_data.dict(by_alias=True)
        df = pd.DataFrame([input_object])

        X_transformed = PREPROCESSOR.transform(df)
        model = MODELS[model_name]

        predict_proba = model.predict_proba(X_transformed)[0][1]
        predict = model.predict(X_transformed)[0]

        return PredictResponse(
            model_used=model_name,
            predict=int(predict),
            proba=float(predict_proba)
        )

    except (ValueError, KeyError) as e:
        raise HTTPException(
            status_code=422,
            detail=f"Некорректные входные данные: {str(e)}"
        )
    except NotFittedError:
        raise HTTPException(
            status_code=500,
            detail="Модель или препроцессор не были обучены (NotFittedError)"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка сервера при предсказании"
        )
