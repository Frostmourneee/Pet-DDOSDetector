from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run

from fastapi_app.config.default import DefaultSettings
from fastapi_app.config.utils import get_settings
from fastapi_app.server.handlers import list_of_routes


def bind_routes(application: FastAPI, setting: DefaultSettings) -> None:
    """
    Биндит все апи-руты
    """
    for route in list_of_routes:
        application.include_router(route, prefix=setting.PATH_PREFIX)


def get_app() -> FastAPI:
    """
    Инициализация приложения
    """
    description = "MLOps-система с Airflow + FastAPI + HDFS поверх AdultIncome датасета"

    tags_metadata = []

    application = FastAPI(
        title="MLOps",
        description=description,
        openapi_url="/openapi",
        version="1.0.0",
        openapi_tags=tags_metadata,
    )
    settings = get_settings()
    bind_routes(application, settings)
    application.state.settings = settings
    return application


app = get_app()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    settings_for_application = get_settings()
    run(
        "fastapi_app.__main__:app",
        host=urlparse(settings_for_application.APP_HOST).netloc,
        port=settings_for_application.APP_PORT,
        reload=True,
        log_level="debug",
    )
