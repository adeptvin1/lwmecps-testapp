from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum
import uuid


class LoadProfile(BaseModel):
    """Профиль нагрузки для эксперимента.
    
    Параметры:
    - concurrent_users: количество одновременных пользователей
    - request_interval: интервал между запросами в секундах
    - profile_duration: длительность профиля в секундах
    """
    concurrent_users: int = Field(..., description="Количество одновременных пользователей")
    request_interval: float = Field(..., description="Интервал между запросами в секундах")
    profile_duration: float = Field(..., description="Длительность профиля в секундах")


class ExperimentState(str, Enum):
    """Состояния эксперимента."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class Host(BaseModel):
    """Хост для эксперимента.
    
    Параметры:
    - host: адрес сервера
    - port: порт сервера
    """
    host: str = Field(..., description="Адрес сервера")
    port: int = Field(..., description="Порт сервера")


class ExperimentSettings(BaseModel):
    """Настройки эксперимента.
    
    Параметры:
    - hosts: список хостов
    - load_profiles: список профилей нагрузки
    - timeout: максимальное время ожидания ответа в секундах
    """
    hosts: List[Host] = Field(..., description="Список хостов")
    load_profiles: List[LoadProfile] = Field(..., description="Список профилей нагрузки")
    timeout: float = Field(0.4, description="Максимальное время ожидания ответа в секундах")


class ExperimentResult(BaseModel):
    """Результат одного запроса в эксперименте.
    
    Параметры:
    - status_code: HTTP код ответа
    - latency_ms: время ответа в миллисекундах
    - timestamp: время выполнения запроса
    - error: описание ошибки (если есть)
    - host_info: информация о хосте, который ответил
    """
    status_code: int = Field(..., description="HTTP код ответа")
    latency_ms: float = Field(..., description="Время ответа в миллисекундах")
    timestamp: datetime = Field(..., description="Время выполнения запроса")
    error: Optional[str] = Field(None, description="Описание ошибки")
    host_info: Optional[Host] = Field(None, description="Информация о хосте, который ответил")


class Experiment(BaseModel):
    """Модель эксперимента.
    
    Параметры:
    - name: название эксперимента
    - settings: настройки эксперимента
    - state: текущее состояние
    - current_profile_index: индекс текущего профиля
    - created_at: время создания
    - updated_at: время последнего обновления
    """
    name: str = Field(..., description="Название эксперимента")
    settings: ExperimentSettings = Field(..., description="Настройки эксперимента")
    state: ExperimentState = Field(ExperimentState.PENDING, description="Текущее состояние")
    current_profile_index: int = Field(0, description="Индекс текущего профиля")
    created_at: datetime = Field(default_factory=datetime.now, description="Время создания")
    updated_at: datetime = Field(default_factory=datetime.now, description="Время последнего обновления")


class ExperimentStats(BaseModel):
    """Статистика эксперимента.
    
    Параметры:
    - current_users: текущее количество пользователей
    - current_interval: текущий интервал между запросами
    - current_profile_index: индекс текущего профиля
    - total_profiles: общее количество профилей
    - average_latency: среднее время ответа
    - average_latency_per_interval: среднее время ответа на интервал
    - requests_count: количество запросов
    - success_rate: процент успешных запросов
    """
    current_users: int = Field(..., description="Текущее количество пользователей")
    current_interval: float = Field(..., description="Текущий интервал между запросами")
    current_profile_index: int = Field(..., description="Индекс текущего профиля")
    total_profiles: int = Field(..., description="Общее количество профилей")
    average_latency: float = Field(..., description="Среднее время ответа")
    average_latency_per_interval: float = Field(..., description="Среднее время ответа на интервал")
    requests_count: int = Field(..., description="Количество запросов")
    success_rate: float = Field(..., description="Процент успешных запросов")


class ExperimentGroup(BaseModel):
    """Группа экспериментов.
    
    Параметры:
    - name: название группы
    - experiment_ids: список ID экспериментов
    - state: текущее состояние
    - created_at: время создания
    - updated_at: время последнего обновления
    """
    name: str = Field(..., description="Название группы")
    experiment_ids: List[str] = Field(default_factory=list, description="Список ID экспериментов")
    state: ExperimentState = Field(ExperimentState.PENDING, description="Текущее состояние")
    created_at: datetime = Field(default_factory=datetime.now, description="Время создания")
    updated_at: datetime = Field(default_factory=datetime.now, description="Время последнего обновления")


class GroupStats(BaseModel):
    """Статистика группы экспериментов.
    
    Параметры:
    - state: текущее состояние
    - experiments_stats: статистика по каждому эксперименту
    - total_requests: общее количество запросов
    - average_latency: среднее время ответа
    - success_rate: процент успешных запросов
    - avg_latency_1m: средняя задержка за последнюю минуту
    - avg_latency_5m: средняя задержка за последние 5 минут
    - avg_latency_15m: средняя задержка за последние 15 минут
    """
    state: ExperimentState = Field(..., description="Текущее состояние")
    experiments_stats: Dict[str, ExperimentStats] = Field(..., description="Статистика по каждому эксперименту")
    total_requests: int = Field(..., description="Общее количество запросов")
    average_latency: float = Field(..., description="Среднее время ответа")
    success_rate: float = Field(..., description="Процент успешных запросов")
    avg_latency_1m: float = Field(0.0, description="Средняя задержка за последнюю минуту")
    avg_latency_5m: float = Field(0.0, description="Средняя задержка за последние 5 минут")
    avg_latency_15m: float = Field(0.0, description="Средняя задержка за последние 15 минут")
