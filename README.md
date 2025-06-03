# LWMECPS Test App

Система для тестирования сетевых соединений и эмуляции нагрузки.

## Компоненты

- **Server** (`/server`) - Серверный компонент для эмуляции нагрузки
- **Client** (`/client`) - Клиентский компонент для управления тестами
- **MongoDB** - База данных для хранения результатов

## Быстрый старт

```bash
# Запуск всех компонентов
docker-compose up -d

# Проверка статуса
docker-compose ps

# Просмотр логов
docker-compose logs -f
```

## Порты

- Client API: `8001`
- Server API: `8000`
- MongoDB: `27017`

## Структура проекта

```
.
├── client/           # Клиентский компонент
│   ├── api/         # API endpoints
│   ├── models/      # Pydantic модели
│   └── database.py  # MongoDB подключение
├── server/          # Серверный компонент
│   ├── api/         # API endpoints
│   └── config.py    # Конфигурация
└── docker-compose.yml
```

## Настройка

### Переменные окружения

Создайте файл `.env` в корневой директории:

```env
# Client
CLIENT_HOST=0.0.0.0
CLIENT_PORT=8001
CLIENT_DEBUG=True
MONGODB_URI=mongodb://mongodb:27017
DATABASE_NAME=network_checks

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_DEBUG=True
CPU_LIMIT=200
RAM_LIMIT=256
MAX_LATENCY=0.5
```

## Использование

1. Запустите систему:
   ```bash
   docker-compose up -d
   ```

2. Создайте эксперимент:
   ```bash
   curl -X POST "http://localhost:8001/api/create_experiment" \
     -H "Content-Type: application/json" \
     -d '{
       "settings": {
         "hosts": [{"host": "server", "port": 8000}],
         "timeout": 1.0,
         "load_profiles": [{
           "concurrent_users": 1,
           "request_interval": 1.0,
           "profile_duration": 60
         }]
       }
     }'
   ```

3. Запустите эксперимент:
   ```bash
   curl -X POST "http://localhost:8001/api/manage_experiment?state=start&experiment_id=<id>"
   ```

4. Проверьте статистику:
   ```bash
   curl -X GET "http://localhost:8001/api/experiment_stats?experiment_id=<id>"
   ```

## Мониторинг

- Метрики Prometheus доступны на:
  - Client: `http://localhost:8001/metrics`
  - Server: `http://localhost:8000/metrics`

## Разработка

### Требования

- Python 3.10+
- Docker и Docker Compose
- MongoDB

### Установка зависимостей

```bash
# Client
cd client
pip install -r requirements.txt

# Server
cd server
pip install -r requirements.txt
```

### Запуск в режиме разработки

```bash
# Client
cd client
uvicorn main:app --reload --host 0.0.0.0 --port 8001

# Server
cd server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Лицензия

MIT

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




