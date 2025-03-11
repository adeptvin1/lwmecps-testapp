import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # MongoDB settings
    # MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    # DATABASE_NAME: str = os.getenv("DATABASE_NAME", "network_checks")
    
    # # Security settings
    # SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key-here")
    # ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
    
    # class Config:
    #     env_file = ".env"

settings = Settings()