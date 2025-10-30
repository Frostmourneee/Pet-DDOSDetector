from fastapi import APIRouter, HTTPException

from fastapi_app.server.schemas.predict import PredictRequest, PredictResponse
from ml_core.predict import are_models_loaded, load_models, predict

api_router = APIRouter(tags=["Prediction"])

PREDICT_RESPONSES = {
    503: {"description": "Модели еще не загружены. Пожалуйста, подождите пока завершится обучение моделей."},
    400: {"description": "Неверные входные данные"},
    500: {"description": "Внутренняя ошибка сервера"}
}

@api_router.post("/predict", response_model=PredictResponse, responses=PREDICT_RESPONSES)
async def predict_income(input_data: PredictRequest, model_name: str = "Random Forest"):
    if not are_models_loaded():
        if not load_models():
            raise HTTPException(
                status_code=503,
                detail="Модели еще не загружены. Пожалуйста, подождите пока завершится обучение моделей."
            )

    try:
        input_dict = input_data.dict(by_alias=True)
        result = predict(input_dict, model_name)
        return PredictResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")