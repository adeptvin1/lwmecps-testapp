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
CPU_USAGE = Gauge("simulated_cpu_usage", "Simulated CPU usage in mCPUs")
RAM_USAGE = Gauge("simulated_ram_usage", "Simulated RAM usage in MB")

last_request_time = time.time()
request_count = 0
current_cpu_usage = 0
current_memory_usage = 0
last_reset_time = time.time()


async def simulate_resource_usage():
    """Эмулирует увеличение задержки при высокой загрузке ресурсов."""
    global current_cpu_usage, current_memory_usage, last_reset_time
    # Сбрасываем CPU usage раз в секунду
    now = time.time()
    if now - last_reset_time >= 1:
        current_cpu_usage = max(0, current_cpu_usage - 50)  # Постепенное снижение нагрузки
        current_memory_usage = max(0, current_memory_usage - 50)  # Постепенное снижение нагрузки RAM
        last_reset_time = now

    current_cpu_usage += random.randint(5, 20)  # mcpus
    current_memory_usage += random.randint(5, 20)  # ram mb

    current_cpu_usage = min(current_cpu_usage, settings.cpu_limit)
    current_memory_usage = min(current_memory_usage, settings.ram_limit)

    # Обновляем метрику CPU
    CPU_USAGE.set(current_cpu_usage)
    RAM_USAGE.set(current_memory_usage)

    # Вычисляем фактор загрузки как среднее между CPU и RAM
    cpu_load = current_cpu_usage / settings.cpu_limit
    ram_load = current_memory_usage / settings.ram_limit
    load_factor = (cpu_load + ram_load) / 2

    # Нелинейное увеличение задержки при росте нагрузки
    if load_factor > 0.5:  # Начинаем увеличивать задержку при загрузке выше 50%
        # Экспоненциальное увеличение задержки
        delay = settings.max_latency * (load_factor ** 2)
        await asyncio.sleep(delay)


@router.get("/latency")
async def check_latency():
    global last_request_time, request_count
    await simulate_resource_usage()

    request_count += 1
    now = time.time()
    elapsed = now - last_request_time
    if elapsed >= 1:  # обновляем RPS раз в секунду
        REQUEST_RATE.set(request_count / elapsed)
        request_count = 0
        last_request_time = now
    return {"message": "pong"}
