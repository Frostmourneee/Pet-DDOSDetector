from os import environ

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Optional

load_dotenv()

class DefaultSettings(BaseSettings):
    """
    Default configs for application.

    Usually, we there are three environments: for development, testing and production.
    But in this situation, we only have standard settings for local development.
    """

    ENV: str = environ.get("ENV", "local")

    PATH_PREFIX: str = environ.get("PATH_PREFIX", "/api/v1")

    APP_PORT: str = environ.get("APP_PORT", 8080)
    AIRFLOW_PORT: str = environ.get("AIRFLOW_PORT", 8081)
