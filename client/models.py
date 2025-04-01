from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class LoadProfile(BaseModel):
    users: int
    duration_seconds: int
    interval_seconds: float


class ExperimentState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


class ExperimentSettings(BaseModel):
    host: str
    port: int
    load_profiles: List[LoadProfile]


class ExperimentResult(BaseModel):
    status_code: int
    latency_ms: float
    timestamp: datetime
    error: Optional[str] = None


class Experiment(BaseModel):
    id: Optional[str] = None
    name: str
    settings: ExperimentSettings
    current_profile_index: int = 0
    state: ExperimentState = ExperimentState.CREATED
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    results: List[ExperimentResult] = []


class ExperimentStats(BaseModel):
    experiment_id: str
    current_users: int
    current_interval: float
    current_profile_index: int
    total_profiles: int
    average_latency: float
    requests_count: int
    success_rate: float
    timestamp: datetime = datetime.now()


class ExperimentGroup(BaseModel):
    id: Optional[str] = None
    name: str
    experiments: Dict[str, Experiment] = {}  # experiment_id -> Experiment
    state: ExperimentState = ExperimentState.CREATED
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()


class GroupStats(BaseModel):
    group_id: str
    state: ExperimentState
    experiments_stats: Dict[str, ExperimentStats]  # experiment_id -> ExperimentStats
    total_requests: int
    average_latency: float
    success_rate: float
    timestamp: datetime = datetime.now()
