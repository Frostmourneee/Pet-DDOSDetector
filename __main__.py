from fastapi import FastAPI
from uvicorn import run
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlparse

from config.default import DefaultSettings
from config.utils import get_settings
from server.handlers import list_of_routes


def bind_routes(application: FastAPI, setting: DefaultSettings) -> None:
    """
    Bind all routes to application.
    """
    for route in list_of_routes:
        application.include_router(route, prefix=setting.PATH_PREFIX)

def get_app() -> FastAPI:
    """
    Creates application and all dependable objects.
    """
    description = "DDOSDetector — MLOps-система для обнаружения ботов с Airflow + FastAPI + HDFS"

    tags_metadata = []

    application = FastAPI(
        title="DDOSDetector",
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
        "__main__:app",
        host=urlparse(settings_for_application.APP_HOST).netloc,
        port=settings_for_application.APP_PORT,
        reload=True,
        log_level="debug",
    )