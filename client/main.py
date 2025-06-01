from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from fastapi.responses import JSONResponse
import uvicorn
from motor.errors import ServerSelectionTimeoutError
import logging

from client.api.endpoints import router as api_router
from client.config import settings
from client.database import init_db

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LWMECPS Client Test App API",
    description="API для мониторинга сетевых соединений",
    version="0.0.1"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Регистрация роутеров
app.include_router(api_router, prefix="/api")


# События приложения
@app.on_event("startup")
async def startup_db_client():
    try:
        await init_db()
        logger.info("Database connection initialized successfully")
    except ServerSelectionTimeoutError:
        logger.error("Could not connect to MongoDB. Please check if MongoDB is running.")
        raise
    except Exception as e:
        logger.error(f"Error initializing database connection: {str(e)}")
        raise


# Обработка ошибок
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    logger.warning(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Корневой эндпоинт
@app.get("/")
async def root():
    return {
        "message": "Welcome to LWMECPS CLIENT TESTAPP API",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run(
        "client.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
