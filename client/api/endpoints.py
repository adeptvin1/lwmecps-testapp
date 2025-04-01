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
    except httpx.RequestError:
        return ExperimentResult(
                status_code=0,
                latency_ms=-1,
                timestamp=datetime.now()
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
    name: str,
    host: str,
    port: int,
    load_profiles: List[LoadProfile],
    db=Depends(get_database)
):
    """Создает новый эксперимент с динамической нагрузкой."""
    settings = ExperimentSettings(
        host=host,
        port=port,
        load_profiles=load_profiles
    )
    
    experiment = Experiment(
        name=name,
        settings=settings
    )

    # Преобразуем в словарь для MongoDB
    exp_dict = experiment.dict(exclude={"id"})

    # Вставляем в базу данных
    result = await db.experiments.insert_one(exp_dict)
    experiment_id = str(result.inserted_id)

    return {"experiment_id": experiment_id}


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
        {"$set": {"state": new_state, "updated_at": datetime.now()}}
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
            current_users=current_profile.users,
            current_interval=current_profile.interval_seconds,
            current_profile_index=experiment["current_profile_index"],
            total_profiles=len(experiment["settings"]["load_profiles"]),
            average_latency=0,
            requests_count=0,
            success_rate=0
        )

    latencies = [r.get("latency_ms", 0) for r in results]
    success_count = sum(1 for r in results if r.get("status_code", 0) == 200)
    
    return ExperimentStats(
        experiment_id=experiment_id,
        current_users=current_profile.users,
        current_interval=current_profile.interval_seconds,
        current_profile_index=experiment["current_profile_index"],
        total_profiles=len(experiment["settings"]["load_profiles"]),
        average_latency=sum(latencies) / len(latencies),
        requests_count=len(results),
        success_rate=success_count / len(results)
    )


async def run_experiment(experiment_id: str):
    """Фоновая задача для выполнения эксперимента с динамической нагрузкой."""
    db = await get_database()
    experiment = await db.experiments.find_one({"_id": ObjectId(experiment_id)})
    
    if not experiment:
        return

    host = experiment["settings"]["host"]
    port = experiment["settings"]["port"]
    load_profiles = experiment["settings"]["load_profiles"]

    while experiment["state"] == ExperimentState.RUNNING:
        current_profile = load_profiles[experiment["current_profile_index"]]
        
        # Создаем задачи для каждого пользователя
        tasks = []
        for _ in range(current_profile.users):
            task = asyncio.create_task(measure_latency(host, port))
            tasks.append(task)
        
        # Ждем завершения всех запросов
        results = await asyncio.gather(*tasks)
        
        # Сохраняем результаты
        for result in results:
            result_dict = result.dict() if hasattr(result, 'dict') else {
                "status_code": result.status_code,
                "latency_ms": result.latency_ms,
                "timestamp": result.timestamp,
                "error": getattr(result, 'error', None)
            }
            await db.experiments.update_one(
                {"_id": ObjectId(experiment_id)},
                {
                    "$push": {"results": result_dict},
                    "$set": {"updated_at": datetime.now()}
                }
            )
        
        # Ждем указанный интервал
        await asyncio.sleep(current_profile.interval_seconds)
        
        # Проверяем, нужно ли перейти к следующему профилю
        if experiment["current_profile_index"] < len(load_profiles) - 1:
            await db.experiments.update_one(
                {"_id": ObjectId(experiment_id)},
                {
                    "$inc": {"current_profile_index": 1},
                    "$set": {"updated_at": datetime.now()}
                }
            )
            experiment["current_profile_index"] += 1
        else:
            await db.experiments.update_one(
                {"_id": ObjectId(experiment_id)},
                {
                    "$set": {
                        "state": ExperimentState.COMPLETED,
                        "updated_at": datetime.now()
                    }
                }
            )
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
            {"$set": {"state": "running", "updated_at": datetime.now()}}
        )

        # Запускаем эксперимент
        await run_experiment(experiment_id)

        # Обновляем статус после завершения
        await db.experiments.update_one(
            {"_id": ObjectId(experiment_id)},
            {"$set": {"state": "completed", "updated_at": datetime.now()}}
        )


@router.post("/create_experiment_group")
async def create_experiment_group(
    name: str,
    experiments: List[Experiment],
    db=Depends(get_database)
):
    """Создает новую группу экспериментов."""
    group = ExperimentGroup(name=name)
    
    # Добавляем эксперименты в группу
    for experiment in experiments:
        if experiment.id in group.experiments:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate experiment ID: {experiment.id}"
            )
        group.experiments[experiment.id] = experiment

    # Преобразуем в словарь для MongoDB
    group_dict = group.dict(exclude={"id"})

    # Вставляем в базу данных
    result = await db.experiment_groups.insert_one(group_dict)
    group_id = str(result.inserted_id)

    return {"group_id": group_id}


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

    for experiment_id, experiment in group["experiments"].items():
        stats = await get_experiment_stats(experiment["id"], db)
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


@router.post("/manage_group")
async def manage_group(
    state: str,
    group_id: str,
    db=Depends(get_database)
):
    """Управляет состоянием группы экспериментов."""
    if state not in ["start", "pause", "stop"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid state. Use 'start', 'pause' or 'stop'"
        )

    group = await db.experiment_groups.find_one({"_id": ObjectId(group_id)})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    new_state = ExperimentState.RUNNING if state == "start" else \
                ExperimentState.PAUSED if state == "pause" else \
                ExperimentState.COMPLETED

    # Обновляем состояние группы
    await db.experiment_groups.update_one(
        {"_id": ObjectId(group_id)},
        {
            "$set": {
                "state": new_state,
                "updated_at": datetime.now()
            }
        }
    )

    # Если запускаем группу, запускаем все эксперименты
    if state == "start":
        for experiment in group["experiments"].values():
            await manage_experiment("start", experiment["id"], db)
    # Если останавливаем группу, останавливаем все эксперименты
    elif state == "stop":
        for experiment in group["experiments"].values():
            await manage_experiment("stop", experiment["id"], db)

    return {"group_id": group_id, "state": new_state}


@router.post("/add_experiment_to_group")
async def add_experiment_to_group(
    group_id: str,
    experiment: Experiment,
    db=Depends(get_database)
):
    """Добавляет новый эксперимент в существующую группу."""
    group = await db.experiment_groups.find_one({"_id": ObjectId(group_id)})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    if experiment.id in group["experiments"]:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment with ID {experiment.id} already exists in group"
        )

    # Добавляем эксперимент в группу
    await db.experiment_groups.update_one(
        {"_id": ObjectId(group_id)},
        {
            "$set": {
                f"experiments.{experiment.id}": experiment.dict(exclude={"id"}),
                "updated_at": datetime.now()
            }
        }
    )

    return {"group_id": group_id, "experiment_id": experiment.id}


@router.delete("/remove_experiment_from_group")
async def remove_experiment_from_group(
    group_id: str,
    experiment_id: str,
    db=Depends(get_database)
):
    """Удаляет эксперимент из группы."""
    group = await db.experiment_groups.find_one({"_id": ObjectId(group_id)})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    if experiment_id not in group["experiments"]:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment with ID {experiment_id} not found in group"
        )

    # Удаляем эксперимент из группы
    await db.experiment_groups.update_one(
        {"_id": ObjectId(group_id)},
        {
            "$unset": {f"experiments.{experiment_id}": ""},
            "$set": {"updated_at": datetime.now()}
        }
    )

    return {"group_id": group_id, "experiment_id": experiment_id}
