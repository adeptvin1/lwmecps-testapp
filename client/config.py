import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Client settings
    SERVER_URL: str = os.getenv("SERVER_URL", "http://localhost:8000")
    CLIENT_PORT: int = int(os.getenv("CLIENT_PORT", 8001))

    # Check settings defaults
    DEFAULT_TIMEOUT: float = float(os.getenv("DEFAULT_TIMEOUT", 1.0))
    DEFAULT_COUNT: int = int(os.getenv("DEFAULT_COUNT", 10))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "network_checks")

    class Config:
        env_file = ".env"


settings = Settings()
