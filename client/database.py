from motor.motor_asyncio import AsyncIOMotorClient
from client.config import settings

# Глобальные переменные для соединения
client = None
db = None


async def init_db():
    """Инициализация подключения к базе данных"""
    global client, db
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


async def get_database():
    """Получение экземпляра базы данных"""
    global db
    if db is None:
        await init_db()
    return db
