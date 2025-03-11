from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
from datetime import datetime
import uuid

class CheckSettings(BaseModel):
    name: str
    target_host: str
    packet_count: int = Field(default=10, ge=1, le=100)
    interval_seconds: int = Field(default=60, ge=10)
    check_type: Literal["ping", "http", "tcp"] = "ping"
    timeout_seconds: float = Field(default=1.0, ge=0.1, le=10.0)
    
    @validator("target_host")
    def validate_host(cls, v):
        if not v or len(v) < 2:
            raise ValueError("Хост должен быть указан")
        return v

class CheckResult(BaseModel):
    check_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    latency_ms: float
    packet_loss_percent: float = Field(ge=0, le=100)
    duration_ms: float
    success: bool
    
    class Config:
        schema_extra = {
            "example": {
                "check_id": "0f8fad5b-d9cb-469f-a165-70867728950e",
                "timestamp": "2023-10-10T15:30:00.123456",
                "latency_ms": 45.3,
                "packet_loss_percent": 0.0,
                "duration_ms": 1020.5,
                "success": True
            }
        }

class Check(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    settings: CheckSettings
    created_at: datetime = Field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    is_active: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "id": "0f8fad5b-d9cb-469f-a165-70867728950e",
                "settings": {
                    "name": "Google DNS Check",
                    "target_host": "8.8.8.8",
                    "packet_count": 10,
                    "interval_seconds": 60,
                    "check_type": "ping",
                    "timeout_seconds": 1.0
                },
                "created_at": "2023-10-10T15:00:00",
                "last_run": "2023-10-10T15:30:00",
                "is_active": True
            }
        }

class ChecksList(BaseModel):
    checks: List[Check]
    
class ResultsList(BaseModel):
    results: List[CheckResult]