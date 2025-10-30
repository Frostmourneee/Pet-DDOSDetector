from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    age: int = Field(..., description="Возраст человека",)
    workclass: str = Field(..., description="Категория рабочего класса")
    fnlwgt: int = Field(..., description="Итоговый вес (final weight)")
    education: str = Field(..., description="Уровень образования")
    education_num: int = Field(..., description="Числовой уровень образования", alias="education-num")
    marital_status: str = Field(..., description="Семейное положение", alias="marital-status")
    occupation: str = Field(..., description="Тип занятости")
    relationship: str = Field(..., description="Семейная роль / статус отношений")
    race: str = Field(..., description="Раса")
    sex: str = Field(..., description="Пол")
    capital_gain: int = Field(..., description="Прирост капитала", alias="capital-gain")
    capital_loss: int = Field(..., description="Потери капитала", alias="capital-loss")
    hours_per_week: int = Field(..., description="Часов работы в неделю", alias="hours-per-week")
    native_country: str = Field(..., description="Страна происхождения", alias="native-country")

    class Config:
        validate_by_name = False
        json_schema_extra = {
            "example": {
                "age": 28,
                "workclass": "Local-gov",
                "fnlwgt": 336951,
                "education": "Assoc-acdm",
                "education-num": 12,
                "marital-status": "Married-civ-spouse",
                "occupation": "Protective-serv",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }

class PredictResponse(BaseModel):
    model_used: str = Field(..., description="Название модели, использованной для предсказания")
    predict: int = Field(..., description="Предсказанный класс: 0 (≤50K) или 1 (>50K)")
    proba: float = Field(..., ge=0.0, le=1.0, description="Вероятность принадлежности к классу 1")

    class Config:
        json_schema_extra = {
            "example": {
                "model_used": "Random Forest",
                "predict": 1,
                "proba": 0.68
            }
        }