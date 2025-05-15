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
    await db.checks.create_index("id", unique=True)
    await db.results.create_index([("check_id", 1), ("timestamp", -1)])


async def get_database():
    """Получение экземпляра базы данных"""
    global db
    if db is None:
        await init_db()
    return db
