from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timedelta
import httpx
import time
import socket
from client.models import (
    ExperimentSettings, ExperimentResult, Experiment, 
    LoadProfile, ExperimentState, ExperimentStats,
    ExperimentGroup, GroupStats, Host
)
import asyncio
from client.database import get_database, with_retry
from bson import ObjectId
from typing import List, Dict, Set, Optional
import json
import logging

logger = logging.getLogger(__name__)

# Константы для оптимизации
CHUNK_SIZE = 1000  # Размер чанка для сохранения результатов
MAX_CONCURRENT_TASKS = 100  # Максимальное количество одновременных задач
HTTP_TIMEOUT = 30.0  # Таймаут для HTTP запросов
HTTP_LIMITS = httpx.Limits(max_keepalive_connections=50, max_connections=100)

# Создаем кастомный JSON encoder для ObjectId
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Глобальный HTTP клиент с настройками
http_client = httpx.AsyncClient(
    timeout=HTTP_TIMEOUT,
    limits=HTTP_LIMITS,
    follow_redirects=True
)

# Сет для отслеживания активных экспериментов
active_experiments: Set[str] = set()
# Блокировка для предотвращения race conditions при работе с active_experiments
active_experiments_lock = asyncio.Lock()


# Функция для преобразования документа MongoDB в JSON-сериализуемый формат
def parse_mongo_doc(doc):
    if doc is None:
        return None

    if isinstance(doc, list):
        return [parse_mongo_doc(item) for item in doc]

    if isinstance(doc, dict):
        return {k: parse_mongo_doc(v) for k, v in doc.items()}

    if isinstance(doc, ObjectId):
        return str(doc)

    if isinstance(doc, datetime):
        return doc.isoformat()

    return doc


router = APIRouter()


async def get_experiment_by_id(experiment_id: str, db=Depends(get_database)):
    """Получает эксперимент по ID."""
    try:
        if not ObjectId.is_valid(experiment_id):
            raise HTTPException(status_code=400, detail="Invalid experiment ID format")

        experiment = await db.experiments.find_one(
            {"_id": ObjectId(experiment_id)}
        )
        if experiment is None:
            raise HTTPException(status_code=404, detail="Experiment not found")

        # Преобразуем документ MongoDB в JSON-сериализуемый формат
        experiment = parse_mongo_doc(experiment)
        return experiment

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


async def check_dns(host: str) -> tuple[bool, str]:
    """Проверяет резолвинг DNS для хоста.
    
    Returns:
        tuple[bool, str]: (успешность резолва, сообщение об ошибке)
    """
    try:
        socket.gethostbyname(host)
        return True, ""
    except socket.gaierror as e:
        return False, f"DNS resolution failed: {str(e)}"


async def try_host(host: str, port: int, timeout: float) -> ExperimentResult:
    """Пробует отправить запрос к одному хосту с таймаутом."""
    # Сначала проверяем DNS
    dns_ok, dns_error = await check_dns(host)
    if not dns_ok:
        return ExperimentResult(
            status_code=0,
            latency_ms=-1,
            timestamp=datetime.now(),
            error=dns_error,
            host_info=Host(host=host, port=port)
        )

    try:
        start_time = time.monotonic()
        response = await http_client.get(f"http://{host}:{port}/api/latency")
        end_time = time.monotonic()
        latency = (end_time - start_time) * 1000
        return ExperimentResult(
            status_code=response.status_code,
            latency_ms=latency,
            timestamp=datetime.now(),
            host_info=Host(host=host, port=port)
        )
    except httpx.ConnectError as e:
        return ExperimentResult(
            status_code=0,
            latency_ms=-1,
            timestamp=datetime.now(),
            error=f"Connection error: {str(e)}",
            host_info=Host(host=host, port=port)
        )
    except httpx.ConnectTimeout as e:
        return ExperimentResult(
            status_code=0,
            latency_ms=-1,
            timestamp=datetime.now(),
            error=f"Connection timeout: {str(e)}",
            host_info=Host(host=host, port=port)
        )
    except httpx.ReadTimeout as e:
        return ExperimentResult(
            status_code=0,
            latency_ms=-1,
            timestamp=datetime.now(),
            error=f"Read timeout: {str(e)}",
            host_info=Host(host=host, port=port)
        )
    except httpx.HTTPError as e:
        return ExperimentResult(
            status_code=0,
            latency_ms=-1,
            timestamp=datetime.now(),
            error=f"HTTP error: {str(e)}",
            host_info=Host(host=host, port=port)
        )
    except Exception as e:
        return ExperimentResult(
            status_code=0,
            latency_ms=-1,
            timestamp=datetime.now(),
            error=f"Unexpected error: {str(e)}",
            host_info=Host(host=host, port=port)
        )


async def measure_latency(hosts: List[Host], timeout: float) -> ExperimentResult:
    """Измеряет латентность, пробуя хосты по очереди."""
    results = []
    for host_info in hosts:
        # Преобразуем словарь в объект Host, если это необходимо
        if isinstance(host_info, dict):
            host_info = Host(**host_info)
        result = await try_host(host_info.host, host_info.port, timeout)
        results.append(result)
        if result.status_code == 200 and result.latency_ms > 0:
            return result
    
    # Если все хосты не ответили, возвращаем последний результат с ошибкой
    return results[-1] if results else None


@router.get("/test-latency")
async def test_latency(settings: ExperimentSettings):
    """Тестирует латентность с использованием списка хостов с приоритетами."""
    result = await measure_latency(settings.hosts, settings.timeout)
    return result


@router.get("/result_experiment")
async def result_experiment(experiment_id: str, db=Depends(get_database)):
    experiment = await get_experiment_by_id(experiment_id, db)
    return experiment


@router.get("/list_experiments")
async def list_experiments(db=Depends(get_database)):
    """Получает список всех экспериментов."""
    experiments = []
    async for experiment in db.experiments.find():
        # Преобразуем документ MongoDB в JSON-сериализуемый формат
        experiment = parse_mongo_doc(experiment)
        
        # Удаляем полные результаты, оставляем только их количество
        if "results" in experiment:
            experiment["results_count"] = len(experiment["results"])
            del experiment["results"]
        
        experiments.append(experiment)
    return experiments


@router.post("/create_experiment")
async def create_experiment(
    experiment: Experiment,
    db=Depends(get_database)
):
    """Создает новый эксперимент с динамической нагрузкой."""
    # Преобразуем в словарь для MongoDB
    exp_dict = experiment.dict()

    # Вставляем в базу данных
    result = await db.experiments.insert_one(exp_dict)

    return {"_id": str(result.inserted_id)}


@router.post("/manage_experiment")
async def manage_experiment(state: str,
                            experiment_id: str,
                            db=Depends(get_database)
                            ):
    if state not in ["start", "pause", "stop"]:
        raise HTTPException(status_code=400,
                            detail="Invalid state. "
                            "Use 'start', 'pause' or 'stop'"
                            )

    new_state = ""

    if state == "start":
        new_state = ExperimentState.RUNNING
        # Запускаем задачу в фоновом режиме
        asyncio.create_task(run_experiment(experiment_id))
    elif state == "pause":
        new_state = ExperimentState.PAUSED
    elif state == "stop":
        new_state = ExperimentState.COMPLETED

    # Обновляем состояние в базе данных
    await db.experiments.update_one(
        {"_id": ObjectId(experiment_id)},
        {"$set": {"state": new_state}}
    )

    return {"_id": experiment_id, "state": new_state}


@router.post("/delete_experiments")
async def delete_experiments():
    """Эндпоинт для удаления всех экспериментов"""
    db = await get_database()
    db.experiments.delete_many({})
    return 200


async def save_results_chunk(db, experiment_id: str, profile_index: int, results: List[dict]):
    """Сохраняет чанк результатов в базу данных с повторными попытками."""
    if not results:
        return
        
    results_docs = [{
        "experiment_id": experiment_id,
        "profile_index": profile_index,
        "timestamp": datetime.now(),
        "result": result
    } for result in results]
    
    try:
        await with_retry(
            db.experiment_results.insert_many,
            results_docs,
            ordered=False  # Продолжать вставку даже при ошибках
        )
    except Exception as e:
        logger.error(f"Error saving results chunk: {str(e)}")
        # Продолжаем выполнение, не прерывая эксперимент


async def get_experiment_stats_optimized(experiment_id: str, db) -> Dict:
    """Получает статистику эксперимента с использованием агрегации MongoDB."""
    pipeline = [
        {"$match": {"experiment_id": experiment_id}},
        {"$group": {
            "_id": None,
            "total_requests": {"$sum": 1},
            "success_count": {
                "$sum": {"$cond": [{"$eq": ["$result.status_code", 200]}, 1, 0]}
            },
            "total_latency": {"$sum": "$result.latency_ms"}
        }}
    ]
    
    stats = await db.experiment_results.aggregate(pipeline).to_list(1)
    if not stats:
        return None
        
    stats = stats[0]
    return {
        "total_requests": stats["total_requests"],
        "success_rate": stats["success_count"] / stats["total_requests"] if stats["total_requests"] > 0 else 0,
        "average_latency": stats["total_latency"] / stats["total_requests"] if stats["total_requests"] > 0 else 0
    }


async def check_and_update_group_status(experiment_id: str, db):
    """Проверяет и обновляет статус группы экспериментов после завершения эксперимента.
    
    Находит все группы, содержащие завершившийся эксперимент, и проверяет,
    завершились ли все эксперименты в группе. Если да, обновляет статус группы на COMPLETED.
    """
    try:
        # Находим все группы, содержащие этот эксперимент
        # В MongoDB запрос {"experiment_ids": value} находит документы, где value содержится в массиве
        groups = await db.groups.find({
            "experiment_ids": experiment_id,
            "state": ExperimentState.RUNNING  # Проверяем только запущенные группы
        }).to_list(None)
        
        if not groups:
            logger.debug(f"No running groups found for experiment {experiment_id}")
            return
        
        logger.info(f"Found {len(groups)} running group(s) containing experiment {experiment_id}")
        
        for group in groups:
            group_id = str(group["_id"])
            experiment_ids = group.get("experiment_ids", [])
            
            if not experiment_ids:
                logger.warning(f"Group {group_id} has no experiment_ids")
                continue
            
            logger.debug(f"Checking group {group_id} with {len(experiment_ids)} experiments")
            
            # Проверяем статусы всех экспериментов в группе
            all_finished = True
            
            for exp_id in experiment_ids:
                try:
                    if not ObjectId.is_valid(exp_id):
                        logger.warning(f"Invalid experiment ID format in group {group_id}: {exp_id}")
                        all_finished = False
                        break
                    
                    experiment = await db.experiments.find_one({"_id": ObjectId(exp_id)})
                    if not experiment:
                        logger.warning(f"Experiment {exp_id} not found in database for group {group_id}")
                        all_finished = False
                        break
                    
                    exp_state = experiment.get("state")
                    
                    # Если хотя бы один эксперимент еще выполняется или приостановлен, группа не завершена
                    if exp_state in [ExperimentState.RUNNING, ExperimentState.PENDING, ExperimentState.PAUSED]:
                        all_finished = False
                        logger.debug(f"Group {group_id} not finished: experiment {exp_id} is {exp_state}")
                        break
                    
                except Exception as exp_error:
                    logger.error(f"Error checking experiment {exp_id} in group {group_id}: {str(exp_error)}")
                    all_finished = False
                    break
            
            # Если все эксперименты завершены (completed или failed), обновляем статус группы
            if all_finished:
                try:
                    await with_retry(
                        db.groups.update_one,
                        {"_id": ObjectId(group_id)},
                        {"$set": {"state": ExperimentState.COMPLETED, "updated_at": datetime.now()}}
                    )
                    logger.info(f"Group {group_id} status updated to COMPLETED - all experiments finished")
                except Exception as update_error:
                    logger.error(f"Failed to update group {group_id} status: {str(update_error)}")
            else:
                logger.debug(f"Group {group_id} not yet completed - some experiments still running")
                
    except Exception as e:
        logger.error(f"Error checking group status for experiment {experiment_id}: {str(e)}", exc_info=True)
        # Не прерываем выполнение, если проверка группы не удалась


async def run_experiment(experiment_id: str):
    """Фоновая задача для выполнения эксперимента с динамической нагрузкой."""
    # Используем блокировку для предотвращения race conditions
    async with active_experiments_lock:
        if experiment_id in active_experiments:
            logger.warning(f"Experiment {experiment_id} is already running")
            return
        
        # Проверяем статус эксперимента в БД перед запуском
        try:
            db = await get_database()
            experiment = await db.experiments.find_one({"_id": ObjectId(experiment_id)})
            if not experiment:
                logger.error(f"Experiment {experiment_id} not found")
                return
            
            # Если эксперимент уже завершен или провалился, не запускаем его снова
            exp_state = experiment.get("state")
            if exp_state in [ExperimentState.COMPLETED, ExperimentState.FAILED]:
                logger.info(f"Experiment {experiment_id} is already {exp_state}, skipping")
                return
            
            # Если эксперимент уже выполняется (по статусу в БД), не запускаем его снова
            if exp_state == ExperimentState.RUNNING:
                logger.warning(f"Experiment {experiment_id} is already running (status in DB)")
                return
        
        except Exception as e:
            logger.error(f"Error checking experiment {experiment_id} status: {str(e)}")
            return
        
        # Добавляем в активные эксперименты только после всех проверок
        active_experiments.add(experiment_id)
    
    # Инициализация переменных вне блока блокировки
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    save_semaphore = asyncio.Semaphore(1)  # Семафор для сохранения результатов
    
    try:
        # Получаем эксперимент еще раз для работы
        db = await get_database()
        experiment = await with_retry(_get_experiment_from_db, experiment_id, db)
        
        if not experiment:
            logger.error(f"Experiment {experiment_id} not found")
            # Удаляем из активных, если эксперимент не найден
            async with active_experiments_lock:
                active_experiments.discard(experiment_id)
            return

        settings = experiment["settings"]
        load_profiles = settings["load_profiles"]
        current_profile_index = experiment["current_profile_index"]
        
        while current_profile_index < len(load_profiles):
            try:
                profile = load_profiles[current_profile_index]
                concurrent_users = profile["concurrent_users"]
                request_interval = profile["request_interval"]
                profile_duration = profile["profile_duration"]
                
                start_time = datetime.now()
                end_time = start_time + timedelta(seconds=profile_duration)
                
                results_buffer = []
                last_save_time = time.monotonic()
                next_request_time = time.monotonic()  # Время следующего запроса
                
                # Метрики для мониторинга
                request_count = 0
                success_count = 0
                total_latency = 0
                last_metrics_time = time.monotonic()
                last_request_count = 0
                
                logger.info(f"Starting profile {current_profile_index + 1}/{len(load_profiles)} "
                          f"with {concurrent_users} users, interval {request_interval}s, "
                          f"duration {profile_duration}s")
                
                while datetime.now() < end_time:
                    try:
                        current_time = time.monotonic()
                        
                        # Если пришло время для следующего запроса
                        if current_time >= next_request_time:
                            # Создаем задачи с таймаутом
                            tasks = []
                            for _ in range(concurrent_users):
                                async with semaphore:
                                    task = asyncio.create_task(
                                        measure_latency(settings["hosts"], settings["timeout"])
                                    )
                                    tasks.append(task)
                            
                            # Ждем завершения всех задач с общим таймаутом
                            done, pending = await asyncio.wait(
                                tasks,
                                timeout=settings["timeout"] * 2,  # Двойной таймаут для всех задач
                                return_when=asyncio.ALL_COMPLETED
                            )
                            
                            # Отменяем оставшиеся задачи
                            for task in pending:
                                task.cancel()
                                try:
                                    await task
                                except asyncio.CancelledError:
                                    pass
                            
                            # Собираем результаты
                            valid_results = []
                            for task in done:
                                try:
                                    result = await task
                                    if result and result.status_code == 200:
                                        valid_results.append(result.dict())
                                        success_count += 1
                                        total_latency += result.latency_ms
                                except Exception as e:
                                    logger.error(f"Error in task: {str(e)}")
                            
                            request_count += concurrent_users
                            results_buffer.extend(valid_results)
                            
                            # Планируем следующий запрос
                            next_request_time = current_time + request_interval
                            
                            # Логируем метрики каждые 5 секунд
                            if current_time - last_metrics_time >= 5.0:
                                current_rps = (request_count - last_request_count) / (current_time - last_metrics_time)
                                avg_latency = total_latency / success_count if success_count > 0 else 0
                                success_rate = (success_count / request_count * 100) if request_count > 0 else 0
                                
                                logger.info(
                                    f"Profile {current_profile_index + 1} metrics: "
                                    f"RPS={current_rps:.2f}, "
                                    f"Success={success_rate:.1f}%, "
                                    f"Latency={avg_latency:.1f}ms, "
                                    f"Total requests={request_count}"
                                )
                                
                                last_metrics_time = current_time
                                last_request_count = request_count
                            
                            # Сохраняем результаты если:
                            # 1. Буфер заполнен
                            # 2. Прошло достаточно времени с последнего сохранения
                            if (len(results_buffer) >= CHUNK_SIZE or 
                                current_time - last_save_time >= 5.0):  # Сохраняем каждые 5 секунд
                                async with save_semaphore:
                                    await save_results_chunk(db, experiment_id, current_profile_index, results_buffer)
                                    results_buffer = []
                                    last_save_time = current_time
                        
                        # Спим до следующего запроса или 100мс, что меньше
                        sleep_time = min(next_request_time - time.monotonic(), 0.1)
                        if sleep_time > 0:
                            await asyncio.sleep(sleep_time)
                        
                    except asyncio.CancelledError:
                        logger.info(f"Experiment {experiment_id} was cancelled")
                        raise
                    except Exception as e:
                        logger.error(f"Error during profile execution: {str(e)}")
                        continue
                
                # Финальные метрики профиля
                total_time = time.monotonic() - (start_time.timestamp())
                avg_rps = request_count / total_time if total_time > 0 else 0
                avg_latency = total_latency / success_count if success_count > 0 else 0
                success_rate = (success_count / request_count * 100) if request_count > 0 else 0
                
                logger.info(
                    f"Profile {current_profile_index + 1} completed: "
                    f"Avg RPS={avg_rps:.2f}, "
                    f"Success={success_rate:.1f}%, "
                    f"Latency={avg_latency:.1f}ms, "
                    f"Total requests={request_count}"
                )
                
                # Сохраняем оставшиеся результаты
                if results_buffer:
                    async with save_semaphore:
                        await save_results_chunk(db, experiment_id, current_profile_index, results_buffer)
                
                current_profile_index += 1
                await with_retry(
                    db.experiments.update_one,
                    {"_id": ObjectId(experiment_id)},
                    {"$set": {"current_profile_index": current_profile_index}}
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error processing profile {current_profile_index}: {str(e)}")
                current_profile_index += 1
                continue
        
        # Помечаем эксперимент как завершенный
        await with_retry(
            db.experiments.update_one,
            {"_id": ObjectId(experiment_id)},
            {"$set": {"state": ExperimentState.COMPLETED}}
        )
    except asyncio.CancelledError:
        logger.info(f"Experiment {experiment_id} was cancelled")
        try:
            await with_retry(
                db.experiments.update_one,
                {"_id": ObjectId(experiment_id)},
                {"$set": {"state": ExperimentState.FAILED}}
            )
        except Exception as update_error:
            logger.error(f"Failed to update experiment state: {str(update_error)}")
        raise
    except Exception as e:
        logger.error(f"Fatal error in experiment {experiment_id}: {str(e)}")
        try:
            await with_retry(
                db.experiments.update_one,
                {"_id": ObjectId(experiment_id)},
                {"$set": {"state": ExperimentState.FAILED}}
            )
        except Exception as update_error:
            logger.error(f"Failed to update experiment state: {str(update_error)}")
    finally:
        # Проверяем и обновляем статус группы экспериментов после завершения эксперимента
        # Это выполнится независимо от того, как завершился эксперимент (успешно, с ошибкой или отменен)
        if db is not None:
            try:
                await check_and_update_group_status(experiment_id, db)
            except Exception as group_check_error:
                # Логируем ошибку, но не прерываем выполнение
                logger.error(f"Failed to check group status in finally block: {str(group_check_error)}")
        
        # Удаляем эксперимент из активных с блокировкой
        async with active_experiments_lock:
            active_experiments.discard(experiment_id)  # Используем discard вместо remove для безопасности


@router.get("/experiment_stats")
async def get_experiment_stats(experiment_id: str, db=Depends(get_database)):
    """Получает статистику эксперимента."""
    try:
        stats = await get_experiment_stats_optimized(experiment_id, db)
        if not stats:
            raise HTTPException(status_code=404, detail="No results found for this experiment")
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create_experiment_group")
async def create_experiment_group(
    group: ExperimentGroup,
    db=Depends(get_database)
):
    """Создает новую группу экспериментов."""
    group_dict = group.dict()
    result = await db.groups.insert_one(group_dict)
    return {"_id": str(result.inserted_id)}


@router.post("/add_experiments_to_group")
async def add_experiments_to_group(
    group_id: str,
    experiment_ids: List[str],
    db=Depends(get_database)
):
    """Добавляет эксперименты в группу."""
    # Проверяем существование группы
    group = await db.groups.find_one({"_id": ObjectId(group_id)})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Проверяем существование экспериментов
    for exp_id in experiment_ids:
        experiment = await db.experiments.find_one({"_id": ObjectId(exp_id)})
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
    
    # Добавляем эксперименты в группу
    await db.groups.update_one(
        {"_id": ObjectId(group_id)},
        {"$addToSet": {"experiment_ids": {"$each": experiment_ids}}}
    )
    
    return {"_id": group_id, "added_experiments": experiment_ids}


@router.get("/group_stats")
async def get_group_stats(group_id: str, db=Depends(get_database)):
    """Получает статистику группы экспериментов."""
    try:
        # Проверяем валидность ID
        if not group_id or not ObjectId.is_valid(group_id):
            raise HTTPException(status_code=400, detail="Invalid group ID format")

        # Получаем группу
        group = await db.groups.find_one({"_id": ObjectId(group_id)})
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")

        # Получаем список ID экспериментов в группе
        experiment_ids = group.get("experiment_ids", [])
        if not experiment_ids:
            return GroupStats(
                state=group.get("state", ExperimentState.PENDING),
                experiments_stats={},
                total_requests=0,
                average_latency=0,
                success_rate=0
            )

        # Получаем текущее время для расчета временных окон
        current_time = datetime.now()
        time_1m_ago = current_time - timedelta(minutes=1)
        time_5m_ago = current_time - timedelta(minutes=5)
        time_15m_ago = current_time - timedelta(minutes=15)

        # Получаем статистику по всем экспериментам в группе
        pipeline = [
            {
                "$match": {
                    "experiment_id": {"$in": experiment_ids}
                }
            },
            {
                "$group": {
                    "_id": "$experiment_id",
                    "total_requests": {"$sum": 1},
                    "success_count": {
                        "$sum": {"$cond": [{"$eq": ["$result.status_code", 200]}, 1, 0]}
                    },
                    "total_latency": {"$sum": "$result.latency_ms"}
                }
            }
        ]

        # Получаем статистику за последние 1, 5 и 15 минут
        pipeline_1m = [
            {
                "$match": {
                    "experiment_id": {"$in": experiment_ids},
                    "timestamp": {"$gte": time_1m_ago}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_latency": {"$sum": "$result.latency_ms"},
                    "total_requests": {"$sum": 1}
                }
            }
        ]

        pipeline_5m = [
            {
                "$match": {
                    "experiment_id": {"$in": experiment_ids},
                    "timestamp": {"$gte": time_5m_ago}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_latency": {"$sum": "$result.latency_ms"},
                    "total_requests": {"$sum": 1}
                }
            }
        ]

        pipeline_15m = [
            {
                "$match": {
                    "experiment_id": {"$in": experiment_ids},
                    "timestamp": {"$gte": time_15m_ago}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_latency": {"$sum": "$result.latency_ms"},
                    "total_requests": {"$sum": 1}
                }
            }
        ]

        # Получаем статистику по каждому эксперименту
        experiments_stats = {}
        total_requests = 0
        total_latency = 0
        total_success = 0

        async for stats in db.experiment_results.aggregate(pipeline):
            exp_id = stats["_id"]
            experiment = await get_experiment_by_id(exp_id, db)
            if not experiment:
                continue

            # Получаем текущий профиль
            current_profile = experiment.get("settings", {}).get("load_profiles", [])
            if not current_profile:
                continue

            current_profile_index = experiment.get("current_profile_index", 0)
            if current_profile_index >= len(current_profile):
                continue

            current_profile = current_profile[current_profile_index]

            # Обновляем общую статистику
            total_requests += stats["total_requests"]
            total_latency += stats["total_latency"]
            total_success += stats["success_count"]

            # Сохраняем статистику по эксперименту
            experiments_stats[exp_id] = ExperimentStats(
                current_users=current_profile.get("concurrent_users", 0),
                current_interval=current_profile.get("request_interval", 0),
                current_profile_index=current_profile_index,
                total_profiles=len(current_profile),
                average_latency=stats["total_latency"] / stats["total_requests"] if stats["total_requests"] > 0 else 0,
                average_latency_per_interval=0,  # Это значение можно вычислить при необходимости
                requests_count=stats["total_requests"],
                success_rate=stats["success_count"] / stats["total_requests"] if stats["total_requests"] > 0 else 0
            )

        # Получаем средние задержки за разные периоды
        stats_1m = await db.experiment_results.aggregate(pipeline_1m).to_list(1)
        stats_5m = await db.experiment_results.aggregate(pipeline_5m).to_list(1)
        stats_15m = await db.experiment_results.aggregate(pipeline_15m).to_list(1)

        avg_latency_1m = stats_1m[0]["total_latency"] / stats_1m[0]["total_requests"] if stats_1m and stats_1m[0]["total_requests"] > 0 else 0
        avg_latency_5m = stats_5m[0]["total_latency"] / stats_5m[0]["total_requests"] if stats_5m and stats_5m[0]["total_requests"] > 0 else 0
        avg_latency_15m = stats_15m[0]["total_latency"] / stats_15m[0]["total_requests"] if stats_15m and stats_15m[0]["total_requests"] > 0 else 0

        return GroupStats(
            state=group.get("state", ExperimentState.PENDING),
            experiments_stats=experiments_stats,
            total_requests=total_requests,
            average_latency=total_latency / total_requests if total_requests > 0 else 0,
            success_rate=total_success / total_requests if total_requests > 0 else 0,
            avg_latency_1m=avg_latency_1m,
            avg_latency_5m=avg_latency_5m,
            avg_latency_15m=avg_latency_15m
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting group stats: {str(e)}"
        )


@router.get("/list_groups")
async def list_groups(db=Depends(get_database)):
    """Получает список всех групп экспериментов."""
    groups = []
    async for group in db.groups.find():
        group = parse_mongo_doc(group)
        groups.append(group)
    return groups


@router.post("/delete_groups")
async def delete_groups(db=Depends(get_database)):
    """Удаляет все группы экспериментов."""
    await db.groups.delete_many({})
    return 200


@router.post("/manage_group")
async def manage_group(
    group_id: str,
    state: str,
    db=Depends(get_database)
):
    """Управляет состоянием группы экспериментов."""
    if state not in ["start", "pause", "stop"]:
        raise HTTPException(status_code=400,
                            detail="Invalid state. "
                            "Use 'start', 'pause' or 'stop'"
                            )
    
    group = await db.groups.find_one({"_id": ObjectId(group_id)})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    current_group_state = group.get("state", ExperimentState.PENDING)
    
    new_state = ""
    if state == "start":
        # Проверяем, не завершена ли уже группа
        if current_group_state == ExperimentState.COMPLETED:
            logger.warning(f"Group {group_id} is already completed, cannot start")
            raise HTTPException(
                status_code=400,
                detail="Group is already completed and cannot be started again"
            )
        
        # Проверяем, не запущена ли уже группа
        if current_group_state == ExperimentState.RUNNING:
            logger.warning(f"Group {group_id} is already running")
            # Возвращаем текущий статус без повторного запуска
            return {"_id": group_id, "state": current_group_state, "message": "Group is already running"}
        
        new_state = ExperimentState.RUNNING
        
        # Запускаем только те эксперименты, которые еще не завершены и не запущены
        experiment_ids = group.get("experiment_ids", [])
        started_count = 0
        skipped_count = 0
        
        for exp_id in experiment_ids:
            try:
                # Проверяем статус эксперимента перед запуском
                experiment = await db.experiments.find_one({"_id": ObjectId(exp_id)})
                if not experiment:
                    logger.warning(f"Experiment {exp_id} not found, skipping")
                    skipped_count += 1
                    continue
                
                exp_state = experiment.get("state")
                
                # Пропускаем уже завершенные или провалившиеся эксперименты
                if exp_state in [ExperimentState.COMPLETED, ExperimentState.FAILED]:
                    logger.debug(f"Experiment {exp_id} is already {exp_state}, skipping")
                    skipped_count += 1
                    continue
                
                # Проверяем, не запущен ли уже эксперимент
                async with active_experiments_lock:
                    if exp_id in active_experiments:
                        logger.debug(f"Experiment {exp_id} is already running, skipping")
                        skipped_count += 1
                        continue
                
                # Запускаем эксперимент
                asyncio.create_task(run_experiment(exp_id))
                started_count += 1
                
            except Exception as e:
                logger.error(f"Error starting experiment {exp_id}: {str(e)}")
                skipped_count += 1
        
        logger.info(f"Group {group_id}: started {started_count} experiments, skipped {skipped_count}")
        
    elif state == "pause":
        new_state = ExperimentState.PAUSED
    elif state == "stop":
        new_state = ExperimentState.COMPLETED
    
    await db.groups.update_one(
        {"_id": ObjectId(group_id)},
        {"$set": {"state": new_state, "updated_at": datetime.now()}}
    )
    
    return {"_id": group_id, "state": new_state}
