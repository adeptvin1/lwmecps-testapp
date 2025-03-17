from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import httpx
import time
from models import ExperimentSettings, ExperimentResult, Experiment
import asyncio
from database import get_database
from bson import ObjectId
from typing import List
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
                host: str,
                port: int,
                interval: int,
                count: int,
                clients: int,
                db=Depends(get_database)):

    settings = ExperimentSettings(
                host=host,
                port=port,
                interval=interval,
                count=count,
                clients=clients
                )
    experiment = Experiment(settings=settings)

    # Преобразуем в словарь для MongoDB
    exp_dict = experiment.dict(exclude={"id"})

    # Вставляем в базу данных
    result = await db.experiments.insert_one(exp_dict)
    experiment_id = str(result.inserted_id)

    # Возвращаем ID созданного эксперимента
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


async def run_experiment(experiment_id: str):
    """Фоновая задача для выполнения эксперимента"""
    # Получаем базу данных
    db = await get_database()

    # Получаем эксперимент
    experiment = await db.experiments.find_one(
                            {"_id": ObjectId(experiment_id)}
                            )
    if not experiment:
        return

    host = experiment["settings"]["host"]
    port = experiment["settings"]["port"]
    interval = experiment["settings"]["interval"]
    count = experiment["settings"]["count"]
    clients = experiment["settings"]["clients"]

    async def client_task():
        completed = 0
        while completed < count:
            current_exp = await db.experiments.find_one(
                                    {"_id": ObjectId(experiment_id)})
            if not current_exp or current_exp["state"] != "running":
                break
            result = await measure_latency(host, port)
            result_dict = result.dict() if hasattr(result, 'dict') else {
                "status_code": result.status_code,
                "latency_ms": result.latency_ms,
                "timestamp": result.timestamp
            }
            await db.experiments.update_one(
                {"_id": ObjectId(experiment_id)},
                {
                    "$push": {"results": result_dict},
                    "$set": {"updated_at": datetime.now()}
                }
            )
            completed += 1
            if completed >= count:
                await db.experiments.update_one(
                    {"_id": ObjectId(experiment_id)},
                    {
                        "$set":
                            {
                                "state": "completed",
                                "updated_at": datetime.now()
                            }
                    }
                )
                break
            await asyncio.sleep(interval)

    tasks = [asyncio.create_task(client_task()) for _ in range(clients)]
    await asyncio.gather(*tasks)


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
