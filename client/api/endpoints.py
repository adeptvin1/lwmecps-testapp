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
from client.database import get_database
from bson import ObjectId
from typing import List, Dict
import json


# Создаем кастомный JSON encoder для ObjectId
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


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
    try:
        experiment = await db.experiments.find_one(
                            {"_id": ObjectId(experiment_id)}
                            )
        if experiment is None:
            raise HTTPException(status_code=404, detail="Experiment not found")

        # Преобразуем документ MongoDB в JSON-сериализуемый формат
        experiment = parse_mongo_doc(experiment)
        return experiment

    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Invalid experiment ID: {str(e)}"
                            )


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
        async with httpx.AsyncClient(timeout=timeout) as client:
            start_time = time.monotonic()
            response = await client.get(f"http://{host}:{port}/api/latency")
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


@router.get("/experiment_stats")
async def get_experiment_stats(experiment_id: str, db=Depends(get_database)):
    """Получает текущую статистику эксперимента."""
    experiment = await get_experiment_by_id(experiment_id, db)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Проверяем, что current_profile_index не выходит за пределы списка профилей
    current_profile_index = min(experiment["current_profile_index"], len(experiment["settings"]["load_profiles"]) - 1)
    current_profile = experiment["settings"]["load_profiles"][current_profile_index]
    results = experiment.get("results", [])
    
    if not results:
        return ExperimentStats(
            current_users=current_profile["concurrent_users"],
            current_interval=current_profile["request_interval"],
            current_profile_index=experiment["current_profile_index"],
            total_profiles=len(experiment["settings"]["load_profiles"]),
            average_latency=0,
            average_latency_per_interval=0,
            requests_count=0,
            success_rate=0
        )

    # Общая статистика
    latencies = [r.get("latency_ms", 0) for r in results]
    success_count = sum(1 for r in results if r.get("status_code", 0) == 200)
    
    # Статистика за интервал
    request_interval = max(1, int(current_profile["request_interval"]))
    timestamps = []
    for r in results:
        ts = r.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        timestamps.append(ts or datetime.now())
    
    # Группируем результаты по интервалам
    interval_results = {}
    for result, timestamp in zip(results, timestamps):
        # Округляем timestamp до ближайшего интервала
        second = int((timestamp.second // request_interval) * request_interval)
        second = min(max(second, 0), 59)
        interval_start = timestamp.replace(
            microsecond=0,
            second=second
        )
        if interval_start not in interval_results:
            interval_results[interval_start] = []
        interval_results[interval_start].append(result.get("latency_ms", 0))
    
    # Считаем среднюю задержку за интервал
    interval_latencies = []
    for interval_results_list in interval_results.values():
        if interval_results_list:  # Проверяем, что список не пустой
            interval_latencies.append(sum(interval_results_list) / len(interval_results_list))
    
    avg_latency_per_interval = sum(interval_latencies) / len(interval_latencies) if interval_latencies else 0
    
    return ExperimentStats(
        current_users=current_profile["concurrent_users"],
        current_interval=current_profile["request_interval"],
        current_profile_index=experiment["current_profile_index"],
        total_profiles=len(experiment["settings"]["load_profiles"]),
        average_latency=sum(latencies) / len(results),
        average_latency_per_interval=avg_latency_per_interval,
        requests_count=len(results),
        success_rate=success_count / len(results)
    )


async def run_experiment(experiment_id: str):
    """Фоновая задача для выполнения эксперимента с динамической нагрузкой."""
    db = await get_database()
    experiment = await get_experiment_by_id(experiment_id, db)
    
    if not experiment:
        return

    settings = experiment["settings"]
    load_profiles = settings["load_profiles"]
    current_profile_index = experiment["current_profile_index"]
    
    while current_profile_index < len(load_profiles):
        profile = load_profiles[current_profile_index]
        concurrent_users = profile["concurrent_users"]
        request_interval = profile["request_interval"]
        profile_duration = profile["profile_duration"]
        
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=profile_duration)
        
        while datetime.now() < end_time:
            tasks = []
            for _ in range(concurrent_users):
                task = measure_latency(settings["hosts"], settings["timeout"])
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Обновляем результаты в базе данных
            await db.experiments.update_one(
                {"_id": ObjectId(experiment_id)},
                {"$push": {"results": {"$each": [r.dict() for r in results]}}}
            )
            
            await asyncio.sleep(request_interval)
        
        current_profile_index += 1
        await db.experiments.update_one(
            {"_id": ObjectId(experiment_id)},
            {"$set": {"current_profile_index": current_profile_index}}
        )
    
    # Помечаем эксперимент как завершенный
    await db.experiments.update_one(
        {"_id": ObjectId(experiment_id)},
        {"$set": {"state": ExperimentState.COMPLETED}}
    )


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

        experiments_stats = {}
        total_requests = 0
        total_latency = 0
        success_count = 0
        
        # Обрабатываем каждый эксперимент в группе
        for exp_id in group.get("experiment_ids", []):
            try:
                # Проверяем валидность ID эксперимента
                if not ObjectId.is_valid(exp_id):
                    continue

                experiment = await get_experiment_by_id(exp_id, db)
                if not experiment:
                    continue
                    
                results = experiment.get("results", [])
                if not results:
                    continue
                    
                # Собираем статистику по результатам
                latencies = [r.get("latency_ms", 0) for r in results if isinstance(r, dict)]
                success_count += sum(1 for r in results if isinstance(r, dict) and r.get("status_code", 0) == 200)
                total_requests += len(results)
                total_latency += sum(latencies)
                
                # Получаем текущий профиль
                current_profile = experiment.get("settings", {}).get("load_profiles", [])
                if not current_profile:
                    continue
                    
                current_profile_index = experiment.get("current_profile_index", 0)
                if current_profile_index >= len(current_profile):
                    continue
                    
                current_profile = current_profile[current_profile_index]
                
                experiments_stats[exp_id] = ExperimentStats(
                    current_users=current_profile.get("concurrent_users", 0),
                    current_interval=current_profile.get("request_interval", 0),
                    current_profile_index=current_profile_index,
                    total_profiles=len(current_profile),
                    average_latency=sum(latencies) / len(results) if results else 0,
                    average_latency_per_interval=0,  # Это значение нужно будет вычислить
                    requests_count=len(results),
                    success_rate=sum(1 for r in results if isinstance(r, dict) and r.get("status_code", 0) == 200) / len(results) if results else 0
                )
            except Exception as e:
                # Пропускаем эксперимент при ошибке, но продолжаем обработку остальных
                continue
        
        return GroupStats(
            state=group.get("state", ExperimentState.PENDING),
            experiments_stats=experiments_stats,
            total_requests=total_requests,
            average_latency=total_latency / total_requests if total_requests > 0 else 0,
            success_rate=success_count / total_requests if total_requests > 0 else 0
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
    
    new_state = ""
    if state == "start":
        new_state = ExperimentState.RUNNING
        # Запускаем все эксперименты в группе
        for exp_id in group["experiment_ids"]:
            asyncio.create_task(run_experiment(exp_id))
    elif state == "pause":
        new_state = ExperimentState.PAUSED
    elif state == "stop":
        new_state = ExperimentState.COMPLETED
    
    await db.groups.update_one(
        {"_id": ObjectId(group_id)},
        {"$set": {"state": new_state}}
    )
    
    return {"_id": group_id, "state": new_state}
