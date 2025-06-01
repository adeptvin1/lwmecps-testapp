from motor.motor_asyncio import AsyncIOMotorClient
from client.config import settings
import logging

logger = logging.getLogger(__name__)

# Глобальные переменные для соединения
client = None
db = None


async def init_db():
    """Инициализация подключения к базе данных"""
    global client, db
    try:
        client = AsyncIOMotorClient(settings.MONGO_URI)
        db = client[settings.DATABASE_NAME]

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
    """Получение экземпляра базы данных"""
    global db
    if db is None:
        await init_db()
    return db


async def close_db_connection():
    """Закрытие соединения с базой данных"""
    global client, db
    if client:
        client.close()
        client = None
        db = None
        logger.info("Database connection closed")
