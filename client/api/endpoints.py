from fastapi import APIRouter
# from typing import List, Optional
# from datetime import datetime
import httpx
import time
# from ..models import CheckSettings, Check, CheckResult, ChecksList
# from ..database import get_database
# from .dependencies import get_check_by_id

router = APIRouter()

# Клиент для измерения задержки


async def measure_latency(host, port):
    async with httpx.AsyncClient() as client:
        start_time = time.monotonic()
        response = await client.get(f"http://{host}:{port}/api/latency")
        end_time = time.monotonic()
        latency = (end_time - start_time) * 1000
        return {"status_code": response.status_code, "latency_ms": latency}


@router.get("/test-latency")
async def test_latency(host: str, port: int):
    result = await measure_latency(host, port)
    return result


@router.get("/result_experiment")
async def result_experiment(experiment_id: str):
    return None


@router.get("/list_experiments")
async def list_experiments():
    return None


@router.post("/create_experiment")
async def create_experiment(host: str, port: int, interval: int, count: int):
    return None


@router.post("/manage_experiment")
async def manage_experiment(state: str, experiment_id: str):
    return None
