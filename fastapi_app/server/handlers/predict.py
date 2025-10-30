from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from fastapi_app.server.schemas.predict import PredictRequest, PredictResponse
from ml_core.predict import are_models_loaded, load_models, predict

api_router: APIRouter = APIRouter(tags=["Prediction"])

PREDICT_RESPONSES: Dict[int, Dict[str, str]] = {
    503: {
        "description": "Модели еще не загружены. Пожалуйста, подождите пока завершится обучение моделей."
    },
    400: {"description": "Неверные входные данные"},
    500: {"description": "Внутренняя ошибка сервера"},
}


@api_router.post(
    "/predict", response_model=PredictResponse, responses=PREDICT_RESPONSES
)
async def predict_income(
    input_data: PredictRequest, model_name: str = "Random Forest"
) -> PredictResponse:
    """
    Предсказание дохода на основе входных данных.

    Args:
        input_data: Входные данные для предсказания
        model_name: Название модели для использования.
                   По умолчанию "Random Forest"

    Returns:
        Результат предсказания с вероятностями

    Raises:
        HTTPException: 503 - если модели не загружены
        HTTPException: 400 - при неверных входных данных
        HTTPException: 500 - при внутренней ошибке сервера
    """
    if not are_models_loaded():
        if not load_models():
            raise HTTPException(
                status_code=503,
                detail="Модели еще не загружены. Пожалуйста, подождите пока завершится обучение моделей.",
            )

    try:
        input_dict = input_data.dict(by_alias=True)
        result = predict(input_dict, model_name)
        return PredictResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")
