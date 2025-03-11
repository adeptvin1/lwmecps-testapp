from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import uvicorn
from typing import List

from api.endpoints import router as api_router
from config import settings

app = FastAPI(
    title="LWMECPS Server Test App API",
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
# @app.on_event("startup")
# async def startup_db_client():
#     await init_db()

# @app.on_event("shutdown")
# async def shutdown_db_client():
#     await close_db_connection()

# Обработка ошибок
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )

# Корневой эндпоинт
@app.get("/")
async def root():
    return {
        "message": "Welcome to LWMECPS SERVER TESTAPP API",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )