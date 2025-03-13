from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class ExperimentSettings(BaseModel):
    host: str
    port: int
    interval: int  # интервал в секундах между проверками
    count: int  # общее количество проверок


class ExperimentResult(BaseModel):
    status_code: int
    latency_ms: float
    timestamp: datetime = datetime.now()

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Experiment(BaseModel):
    id: Optional[str] = None
    settings: ExperimentSettings
    state: str = "pending"  # pending, running, paused, completed
    results: List[ExperimentResult] = []
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
