from motor.motor_asyncio import AsyncIOMotorClient
from client.config import settings
import logging
import asyncio
from typing import Optional, Any, Callable, TypeVar, Awaitable

logger = logging.getLogger(__name__)

# Глобальные переменные для соединения
client: Optional[AsyncIOMotorClient] = None
db = None
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

T = TypeVar('T')

async def with_retry(operation: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
    """Выполняет операцию с базой данных с повторными попытками.
    
    Args:
        operation: Асинхронная функция для выполнения
        *args: Позиционные аргументы для операции
        **kwargs: Именованные аргументы для операции
        
    Returns:
        Результат операции
        
    Raises:
        Exception: Если все попытки выполнения завершились ошибкой
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                break
            logger.warning(f"Database operation failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            await asyncio.sleep(RETRY_DELAY)
    
    logger.error(f"Database operation failed after {MAX_RETRIES} attempts: {str(last_error)}")
    raise last_error

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
