from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import httpx
import time
from models import (
    ExperimentSettings, ExperimentResult, Experiment, 
    LoadProfile, ExperimentState, ExperimentStats,
    ExperimentGroup, GroupStats
)
import asyncio
from database import get_database
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

        # Преобразуем ObjectId в строку для вывода
        experiment["id"] = str(experiment["_id"])
        del experiment["_id"]  # Удаляем _id, так как уже есть id

        return experiment

    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Invalid experiment ID: {str(e)}"
                            )


async def measure_latency(host, port):
    try:
        async with httpx.AsyncClient() as client:
            start_time = time.monotonic()
            response = await client.get(f"http://{host}:{port}/api/latency")
            end_time = time.monotonic()
            latency = (end_time - start_time) * 1000
            return ExperimentResult(
                status_code=response.status_code,
                latency_ms=latency,
                timestamp=datetime.now()
            )
    except httpx.RequestError as e:
        return ExperimentResult(
            status_code=0,
            latency_ms=-1,
            timestamp=datetime.now(),
            error=str(e)
        )


@router.get("/test-latency")
async def test_latency(host: str, port: int):
    result = await measure_latency(host, port)
    return result


@router.get("/result_experiment")
async def result_experiment(experiment_id: str, db=Depends(get_database)):
    experiment = await get_experiment_by_id(experiment_id, db)
    return experiment


@router.get("/list_experiments")
async def list_experiments(db=Depends(get_database)):
    experiments = []
    async for exp in db.experiments.find():
        # Преобразуем документ MongoDB в JSON-сериализуемый формат
        exp = parse_mongo_doc(exp)
        exp["id"] = str(exp["_id"])
        del exp["_id"]
        experiments.append(exp)
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

    return {"experiment_id": experiment.id}


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
        new_state = "running"
        # Запускаем задачу в фоновом режиме
        asyncio.create_task(run_experiment(experiment_id))
    elif state == "pause":
        new_state = "paused"
    elif state == "stop":
        new_state = "completed"

    # Обновляем состояние в базе данных
    await db.experiments.update_one(
        {"_id": ObjectId(experiment_id)},
        {"$set": {"state": new_state}}
    )

    return {"experiment_id": experiment_id, "state": new_state}


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

    current_profile = experiment["settings"]["load_profiles"][experiment["current_profile_index"]]
    results = experiment.get("results", [])
    
    if not results:
        return ExperimentStats(
            experiment_id=experiment_id,
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
    request_interval = current_profile["request_interval"]
    timestamps = [r.get("timestamp", datetime.now()) for r in results]
    
    # Группируем результаты по интервалам
    interval_results = {}
    for result, timestamp in zip(results, timestamps):
        # Округляем timestamp до ближайшего интервала
        interval_start = timestamp.replace(
            microsecond=0,
            second=(timestamp.second // request_interval) * request_interval
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
        experiment_id=experiment_id,
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
    """Фоновая задача для выполнения эксперимента с динамической нагрузкой.
    
    Args:
        experiment_id (str): ID эксперимента для запуска
        
    Процесс выполнения:
    1. Получает настройки эксперимента из базы данных
    2. Для каждого профиля нагрузки:
       - concurrent_users: количество одновременных пользователей
       - request_interval: интервал между запросами в секундах
       - profile_duration: длительность профиля в секундах
    3. В течение profile_duration:
       - concurrent_users пользователей делают запросы
       - Ждет request_interval секунд
       - Повторяет, пока не истечет profile_duration
    4. Переходит к следующему профилю или завершает эксперимент
    """
    db = await get_database()
    experiment = await db.experiments.find_one({"_id": ObjectId(experiment_id)})
    
    if not experiment:
        print(f"Experiment {experiment_id} not found")
        return

    host = experiment["settings"]["host"]
    port = experiment["settings"]["port"]
    load_profiles = experiment["settings"]["load_profiles"]
    
    print(f"Starting experiment {experiment_id} with {len(load_profiles)} profiles")
    print(f"Current profile: {load_profiles[experiment['current_profile_index']]}")

    while True:
        # Проверяем текущее состояние эксперимента
        current_experiment = await db.experiments.find_one({"_id": ObjectId(experiment_id)})
        if not current_experiment or current_experiment["state"] != ExperimentState.RUNNING:
            print(f"Experiment {experiment_id} stopped or not found")
            break

        current_profile = load_profiles[experiment["current_profile_index"]]
        print(f"Running profile {experiment['current_profile_index']}: {current_profile}")
        
        concurrent_users = current_profile["concurrent_users"]
        request_interval = current_profile["request_interval"]
        profile_duration = current_profile["profile_duration"]
        
        print(f"Starting profile with {concurrent_users} concurrent users, "
              f"request interval {request_interval}s, "
              f"profile duration {profile_duration}s")
        
        profile_start_time = time.monotonic()
        while time.monotonic() - profile_start_time < profile_duration:
            # Создаем задачи для каждого пользователя
            request_tasks = []
            for _ in range(concurrent_users):
                task = asyncio.create_task(measure_latency(host, port))
                request_tasks.append(task)
            
            # Ждем завершения всех запросов
            request_results = await asyncio.gather(*request_tasks)
            print(f"Completed {len(request_results)} requests")
            
            # Сохраняем результаты
            for result in request_results:
                result_dict = result.dict() if hasattr(result, 'dict') else {
                    "status_code": result.status_code,
                    "latency_ms": result.latency_ms,
                    "timestamp": result.timestamp,
                    "error": getattr(result, 'error', None)
                }
                await db.experiments.update_one(
                    {"_id": ObjectId(experiment_id)},
                    {"$push": {"results": result_dict}}
                )
            
            # Ждем указанный интервал
            print(f"Waiting {request_interval} seconds before next iteration")
            await asyncio.sleep(request_interval)
        
        print(f"Profile duration completed")
        
        # Проверяем, нужно ли перейти к следующему профилю
        if experiment["current_profile_index"] < len(load_profiles) - 1:
            await db.experiments.update_one(
                {"_id": ObjectId(experiment_id)},
                {"$inc": {"current_profile_index": 1}}
            )
            experiment["current_profile_index"] += 1
            print(f"Moving to profile {experiment['current_profile_index']}")
        else:
            await db.experiments.update_one(
                {"_id": ObjectId(experiment_id)},
                {"$set": {"state": ExperimentState.COMPLETED}}
            )
            print(f"Experiment {experiment_id} completed")
            break


@router.post("/run_experiments_queue")
async def run_experiments_queue(
            experiment_ids: List[str],
            db=Depends(get_database)):
    """
    Запускает эксперименты в порядке, указанном в experiment_ids.
    """
    if not experiment_ids:
        raise HTTPException(
            status_code=400,
            detail="No experiment IDs provided")

    # Проверяем, существуют ли все переданные эксперименты
    existing_experiments = await db.experiments.find(
        {"_id": {"$in": [ObjectId(eid) for eid in experiment_ids]}}
    ).to_list(length=None)

    existing_ids = {str(exp["_id"]) for exp in existing_experiments}
    not_found_ids = [eid for eid in experiment_ids if eid not in existing_ids]

    if not_found_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Experiments not found: {not_found_ids}")

    # Запускаем выполнение в фоне
    asyncio.create_task(run_experiments_sequentially(experiment_ids, db))

    return {
        "message": "Experiments queue started",
        "experiment_ids": experiment_ids
        }


async def run_experiments_sequentially(experiment_ids: List[str], db):
    """Выполняет эксперименты последовательно, в заданном порядке."""
    for experiment_id in experiment_ids:
        # Обновляем статус перед запуском
        await db.experiments.update_one(
            {"_id": ObjectId(experiment_id)},
            {"$set": {"state": "running"}}
        )

        # Запускаем эксперимент
        await run_experiment(experiment_id)

        # Обновляем статус после завершения
        await db.experiments.update_one(
            {"_id": ObjectId(experiment_id)},
            {"$set": {"state": "completed"}}
        )


@router.post("/create_experiment_group")
async def create_experiment_group(
    group: ExperimentGroup,
    db=Depends(get_database)
):
    """Создает новую группу экспериментов."""
    # Преобразуем в словарь для MongoDB
    group_dict = group.dict(exclude={"id"})

    # Вставляем в базу данных
    result = await db.experiment_groups.insert_one(group_dict)
    group_id = str(result.inserted_id)

    return {"group_id": group_id}


@router.post("/add_experiments_to_group")
async def add_experiments_to_group(
    group_id: str,
    experiment_ids: List[str],
    db=Depends(get_database)
):
    """Добавляет один или несколько экспериментов в существующую группу по их ID."""
    group = await db.experiment_groups.find_one({"_id": ObjectId(group_id)})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    # Проверяем существование экспериментов
    existing_experiments = await db.experiments.find(
        {"_id": {"$in": [ObjectId(eid) for eid in experiment_ids]}}
    ).to_list(length=None)
    
    existing_ids = {str(exp["_id"]) for exp in existing_experiments}
    not_found_ids = [eid for eid in experiment_ids if eid not in existing_ids]
    
    if not_found_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Experiments not found: {not_found_ids}"
        )

    # Проверяем на дубликаты
    duplicate_ids = [
        eid for eid in experiment_ids 
        if eid in group["experiment_ids"]
    ]
    if duplicate_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Experiments with IDs {duplicate_ids} already exist in group"
        )

    # Добавляем ID экспериментов в группу
    await db.experiment_groups.update_one(
        {"_id": ObjectId(group_id)},
        {"$push": {"experiment_ids": {"$each": experiment_ids}}}
    )

    return {
        "group_id": group_id,
        "added_experiments": experiment_ids
    }


@router.get("/group_stats")
async def get_group_stats(group_id: str, db=Depends(get_database)):
    """Получает статистику по группе экспериментов."""
    group = await db.experiment_groups.find_one({"_id": ObjectId(group_id)})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    # Получаем статистику по каждому эксперименту
    experiments_stats = {}
    total_requests = 0
    total_latency = 0
    total_success = 0

    for experiment_id in group["experiment_ids"]:
        stats = await get_experiment_stats(experiment_id, db)
        experiments_stats[experiment_id] = stats
        
        # Обновляем общую статистику
        total_requests += stats.requests_count
        total_latency += stats.average_latency * stats.requests_count
        total_success += stats.success_rate * stats.requests_count

    # Рассчитываем средние значения
    avg_latency = total_latency / total_requests if total_requests > 0 else 0
    success_rate = total_success / total_requests if total_requests > 0 else 0

    return GroupStats(
        group_id=group_id,
        state=group["state"],
        experiments_stats=experiments_stats,
        total_requests=total_requests,
        average_latency=avg_latency,
        success_rate=success_rate
    )


@router.get("/list_groups")
async def list_groups(db=Depends(get_database)):
    """Получает список всех групп экспериментов."""
    groups = []
    async for group in db.experiment_groups.find():
        # Преобразуем документ MongoDB в JSON-сериализуемый формат
        group = parse_mongo_doc(group)
        group["id"] = str(group["_id"])
        del group["_id"]
        groups.append(group)
    return groups


@router.post("/delete_groups")
async def delete_groups(db=Depends(get_database)):
    """Удаляет все группы экспериментов."""
    await db.experiment_groups.delete_many({})
    return {"message": "All groups deleted successfully"}


@router.post("/manage_group")
async def manage_group(
    group_id: str,
    state: str,
    db=Depends(get_database)
):
    """Управляет состоянием группы экспериментов (запуск, пауза, остановка)."""
    if state not in ["start", "pause", "stop"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid state. Use 'start', 'pause' or 'stop'"
        )

    group = await db.experiment_groups.find_one({"_id": ObjectId(group_id)})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    new_state = ""
    if state == "start":
        new_state = ExperimentState.RUNNING
        # Запускаем все эксперименты в группе
        for experiment_id in group["experiment_ids"]:
            await db.experiments.update_one(
                {"_id": ObjectId(experiment_id)},
                {"$set": {"state": ExperimentState.RUNNING}}
            )
            asyncio.create_task(run_experiment(experiment_id))
    elif state == "pause":
        new_state = ExperimentState.PAUSED
        # Приостанавливаем все эксперименты в группе
        for experiment_id in group["experiment_ids"]:
            await db.experiments.update_one(
                {"_id": ObjectId(experiment_id)},
                {"$set": {"state": ExperimentState.PAUSED}}
            )
    elif state == "stop":
        new_state = ExperimentState.COMPLETED
        # Останавливаем все эксперименты в группе
        for experiment_id in group["experiment_ids"]:
            await db.experiments.update_one(
                {"_id": ObjectId(experiment_id)},
                {"$set": {"state": ExperimentState.COMPLETED}}
            )

    # Обновляем состояние группы
    await db.experiment_groups.update_one(
        {"_id": ObjectId(group_id)},
        {"$set": {"state": new_state}}
    )

    return {"group_id": group_id, "state": new_state}
