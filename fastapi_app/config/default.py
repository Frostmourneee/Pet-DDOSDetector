from os import environ

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

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
    HDFS_WEB_PORT: str = environ.get("HDFS_WEB_PORT", 9870)
    HDFS_PORT: str = environ.get("HDFS_PORT", 9000)
    HDFS_NODE_MANAGER_PORT: str = environ.get("HDFS_NODE_MANAGER_PORT", 8042)
