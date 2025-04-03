# lwmecps-app

## Быстрый старт

Для запуска приложения выполните:

```bash
docker-compose up -d --build
```

Это создаст и запустит все необходимые контейнеры (API, MongoDB) в фоновом режиме.

После запуска сервисы будут доступны по следующим адресам:
- API: http://localhost:8001
- Swagger UI: http://localhost:8001/docs
- MongoDB: mongodb://localhost:27017

## Описание

Данный сервис представляет собой API на основе FastAPI, предназначенный для проведения экспериментов по измерению задержки (latency) удаленных серверов. Сервис позволяет создавать, управлять и анализировать эксперименты по тестированию производительности API.

## Возможности

- Создание экспериментов с заданными параметрами (хост, порт, профили нагрузки)
- Создание групп экспериментов для параллельного тестирования нескольких серверов
- Управление состоянием экспериментов и групп (запуск, пауза, остановка)
- Измерение задержки удаленных API-эндпоинтов
- Сохранение результатов в MongoDB
- Получение списка всех экспериментов и групп
- Получение детальных результатов конкретного эксперимента или группы
- Динамическое добавление и удаление экспериментов из групп

## Требования

- Docker и Docker Compose
- Python 3.6+ (для локальной разработки)
- FastAPI
- httpx
- MongoDB
- asyncio

## Структура API

### Эндпоинты

#### GET `/test-latency`

Выполняет одиночный тест задержки для указанного хоста и порта.

**Параметры:**

- `host` - хост для тестирования
- `port` - порт для тестирования

**Возвращает:**

- Объект `ExperimentResult` с информацией о статус-коде, задержке в миллисекундах и временной метке

#### GET `/result_experiment`

Получает результаты эксперимента по его ID.

**Параметры:**

- `experiment_id` - идентификатор эксперимента

**Возвращает:**

- Объект эксперимента со всеми настройками и результатами

#### GET `/list_experiments`

Получает список всех экспериментов.

**Возвращает:**

- Массив объектов экспериментов (без полных результатов, только основная информация)

#### POST `/create_experiment`

Создает новый эксперимент с указанными параметрами.

**Параметры:**

- `name` - имя эксперимента
- `settings` - настройки эксперимента:
  - `host` - хост для тестирования
  - `port` - порт для тестирования
  - `load_profiles` - список профилей нагрузки

**Возвращает:**

- Идентификатор созданного эксперимента

#### POST `/manage_experiment`

Управляет состоянием эксперимента.

**Параметры:**

- `state` - состояние эксперимента ("start", "pause", "stop")
- `experiment_id` - идентификатор эксперимента

**Возвращает:**

- Идентификатор эксперимента и его новое состояние

#### POST `/create_experiment_group`

Создает новую группу экспериментов.

**Параметры:**

- `name` - имя группы
- `experiment_ids` - список ID экспериментов

**Возвращает:**

- Идентификатор созданной группы

#### POST `/add_experiments_to_group`

Добавляет один или несколько экспериментов в существующую группу.

**Параметры:**

- `group_id` - идентификатор группы
- `experiment_ids` - список ID экспериментов для добавления

**Возвращает:**

- Идентификатор группы и список добавленных экспериментов

#### GET `/group_stats`

Получает статистику по группе экспериментов.

**Параметры:**

- `group_id` - идентификатор группы

**Возвращает:**

- Объект `GroupStats` с информацией о группе и статистикой по каждому эксперименту

#### GET `/list_groups`

Получает список всех групп экспериментов.

**Возвращает:**

- Массив объектов групп

#### POST `/delete_groups`

Удаляет все группы экспериментов.

**Возвращает:**

- Сообщение об успешном удалении

#### POST `/manage_group`

Управляет состоянием группы экспериментов.

**Параметры:**

- `group_id` - идентификатор группы
- `state` - состояние группы ("start", "pause", "stop")

**Возвращает:**

- Идентификатор группы и ее новое состояние

## Примеры использования

### Создание эксперимента

```python
import requests

# Создание эксперимента
experiment_data = {
    "name": "Тестовый эксперимент",
    "settings": {
        "host": "example.com",
        "port": 80,
        "load_profiles": [
            {
                "concurrent_users": 10,
                "request_interval": 1.0,
                "profile_duration": 60.0
            },
            {
                "concurrent_users": 20,
                "request_interval": 0.5,
                "profile_duration": 30.0
            }
        ]
    }
}

response = requests.post("http://localhost:8001/api/create_experiment", json=experiment_data)
experiment_id = response.json()["experiment_id"]
```

### Создание группы экспериментов

```python
# Создание группы
group_data = {
    "name": "Тестовая группа",
    "experiment_ids": [experiment_id]
}

response = requests.post("http://localhost:8001/api/create_experiment_group", json=group_data)
group_id = response.json()["group_id"]

# Запуск группы
response = requests.post(
    "http://localhost:8001/api/manage_group",
    params={"group_id": group_id, "state": "start"}
)

# Получение статистики
response = requests.get(f"http://localhost:8001/api/group_stats?group_id={group_id}")
stats = response.json()
```

## Модели данных

### Experiment

```python
class Experiment:
    id: Optional[str]
    name: str
    settings: ExperimentSettings
    state: ExperimentState
    current_profile_index: int
    results: List[ExperimentResult]
    created_at: datetime
    updated_at: datetime
```

### ExperimentGroup

```python
class ExperimentGroup:
    id: Optional[str]
    name: str
    experiment_ids: List[str]
    state: ExperimentState
    created_at: datetime
    updated_at: datetime
```

### LoadProfile

```python
class LoadProfile:
    concurrent_users: int
    request_interval: float
    profile_duration: float
```

## Состояния экспериментов и групп

- `pending` - ожидает запуска
- `running` - выполняется
- `paused` - приостановлен
- `completed` - завершен
- `failed` - завершился с ошибкой
