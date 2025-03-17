import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Server settings
    host: str = os.getenv("host", "0.0.0.0")
    port: int = int(os.getenv("port", 8000))
    debug: bool = os.getenv("debug", "True").lower() == "true"
    # prometheus_port: str = os.getenv("prometheus_port", 9000)
    cpu_limit: int = os.getenv("cpu_limit", "1000")  # mCPUs
    ram_limit: int = os.getenv("ram_limit", "1024")  # в MB
    # MongoDB settings
    # MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    # DATABASE_NAME: str = os.getenv("DATABASE_NAME", "network_checks")

    # # Security settings
    # SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key-here")

    # class Config:
    #     env_file = ".env"

    # TODO: Добавить переменные окружения чтобы прокидывать лимиты пода.


settings = Settings()
