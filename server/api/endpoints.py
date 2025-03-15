from fastapi import APIRouter
from prometheus_client import Gauge
import time
from config import settings
import random
import asyncio
router = APIRouter()

# TODO: Добавить на каждый запрос "синтетическое" потребление ресурсов,
# чтобы при утилизации ресурсов
# он увеличивал задержку эмулируя тротлинг
# параметры о емкости пода можно брать из переменных окружения.


REQUEST_RATE = Gauge("latency_requests_per_second",
                     "Requests per second to /latency")

# Метрики для эмулируемой загрузки CPU и RAM
CPU_USAGE = Gauge("simulated_cpu_usage", "Simulated CPU usage in percentage")
RAM_USAGE = Gauge("simulated_ram_usage", "Simulated RAM usage in MB")

last_request_time = time.time()
request_count = 0


async def simulate_resource_usage():
    """Эмулирует увеличение задержки при высокой загрузке ресурсов."""
    # Фейковая загрузка CPU
    cpu_usage = random.uniform(0, settings.cpu_limit)
    # Фейковая загрузка RAM
    memory_usage = random.uniform(0, settings.ram_limit)

    load_factor = max(cpu_usage / settings.cpu_limit,
                      memory_usage / settings.ram_limit)

    if load_factor > 0.8:  # Если загрузка выше 80%, увеличиваем задержку
        delay = random.uniform(0.1, 0.5) * load_factor
        await asyncio.sleep(delay)


@router.get("/latency")
async def check_latency():
    global last_request_time, request_count
    await simulate_resource_usage()

    request_count += 1
    now = time.time()
    elapsed = now - last_request_time
    if elapsed > 1:  # обновляем RPS раз в секунду
        REQUEST_RATE.set(request_count / elapsed)
        request_count = 0
        last_request_time = now
    return {"message": "pong"}
