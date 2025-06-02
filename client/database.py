from motor.motor_asyncio import AsyncIOMotorClient
from client.config import settings
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)

# Глобальные переменные для соединения
client: Optional[AsyncIOMotorClient] = None
db = None
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


async def init_db():
    """Инициализация подключения к базе данных с пулом соединений и таймаутами"""
    global client, db
    try:
        client = AsyncIOMotorClient(
            settings.MONGO_URI,
            maxPoolSize=50,  # Максимальный размер пула соединений
            minPoolSize=10,  # Минимальный размер пула
            maxIdleTimeMS=30000,  # Максимальное время простоя соединения
            connectTimeoutMS=5000,  # Таймаут подключения
            serverSelectionTimeoutMS=5000,  # Таймаут выбора сервера
            waitQueueTimeoutMS=10000,  # Таймаут ожидания в очереди
            retryWrites=True,  # Повторные попытки записи
            retryReads=True,  # Повторные попытки чтения
        )
        db = client[settings.DATABASE_NAME]

        # Проверяем соединение
        await db.command('ping')
        logger.info("Successfully connected to MongoDB")

        # Создание индексов для оптимизации запросов
        await db.experiments.create_index("_id")
        await db.experiment_results.create_index([
            ("experiment_id", 1),
            ("timestamp", -1)
        ])
        await db.experiment_results.create_index([
            ("experiment_id", 1),
            ("profile_index", 1)
        ])
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise


async def get_database():
    """Получает или создает подключение к базе данных с пулом соединений."""
    global client, db
    if client is None:
        await init_db()
    return db


async def close_db():
    """Закрывает соединение с базой данных."""
    global client
    if client:
        client.close()
        client = None
        db = None
        logger.info("Database connection closed")
