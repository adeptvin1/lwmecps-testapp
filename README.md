# lwmecps-app

## Описание

Данный сервис представляет собой API на основе FastAPI, предназначенный для проведения экспериментов по измерению задержки (latency) удаленных серверов. Сервис позволяет создавать, управлять и анализировать эксперименты по тестированию производительности API.

## Возможности

- Создание экспериментов с заданными параметрами (хост, порт, интервал, количество проверок)
- Создание групп экспериментов для параллельного тестирования нескольких серверов
- Управление состоянием экспериментов и групп (запуск, пауза, остановка)
- Измерение задержки удаленных API-эндпоинтов
- Сохранение результатов в MongoDB
- Получение списка всех экспериментов и групп
- Получение детальных результатов конкретного эксперимента или группы
- Динамическое добавление и удаление экспериментов из групп

## Требования

- Python 3.6+
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

- Массив объектов экспериментов

#### POST `/create_experiment`

Создает новый эксперимент с указанными параметрами.

**Параметры:**

- `name` - имя эксперимента
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
- `experiments` - список экспериментов

**Возвращает:**

- Идентификатор созданной группы

#### GET `/group_stats`

Получает статистику по группе экспериментов.

**Параметры:**

- `group_id` - идентификатор группы

**Возвращает:**

- Объект `GroupStats` со статистикой по всем экспериментам в группе

#### POST `/manage_group`

Управляет состоянием группы экспериментов.

**Параметры:**

- `state` - состояние группы ("start", "pause", "stop")
- `group_id` - идентификатор группы

**Возвращает:**

- Идентификатор группы и её новое состояние

#### POST `/add_experiment_to_group`

Добавляет эксперимент в существующую группу.

**Параметры:**

- `group_id` - идентификатор группы
- `experiment` - объект эксперимента

**Возвращает:**

- Идентификатор группы и добавленного эксперимента

#### DELETE `/remove_experiment_from_group`

Удаляет эксперимент из группы.

**Параметры:**

- `group_id` - идентификатор группы
- `experiment_id` - идентификатор эксперимента

**Возвращает:**

- Идентификатор группы и удаленного эксперимента

## Модели данных

### LoadProfile

- `users` - количество пользователей
- `duration_seconds` - длительность в секундах
- `interval_seconds` - интервал между запросами в секундах

### ExperimentSettings

- `host` - хост для тестирования
- `port` - порт для тестирования
- `load_profiles` - список профилей нагрузки

### ExperimentResult

- `status_code` - HTTP-статус код ответа
- `latency_ms` - время задержки в миллисекундах
- `timestamp` - временная метка
- `error` - опциональное сообщение об ошибке

### Experiment

- `id` - уникальный идентификатор эксперимента
- `name` - имя эксперимента
- `settings` - настройки эксперимента
- `current_profile_index` - индекс текущего профиля нагрузки
- `state` - состояние эксперимента
- `created_at` - дата и время создания
- `updated_at` - дата и время последнего обновления
- `results` - массив результатов

### ExperimentGroup

- `id` - уникальный идентификатор группы
- `name` - имя группы
- `experiments` - словарь экспериментов (ID -> Experiment)
- `state` - состояние группы
- `created_at` - дата и время создания
- `updated_at` - дата и время последнего обновления

### GroupStats

- `group_id` - идентификатор группы
- `state` - состояние группы
- `experiments_stats` - словарь статистики по экспериментам
- `total_requests` - общее количество запросов
- `average_latency` - средняя задержка
- `success_rate` - процент успешных запросов
- `timestamp` - временная метка

## Примеры использования

### Создание простого эксперимента

```python
# Создание профиля нагрузки
load_profile = {
    "users": 100,
    "duration_seconds": 60,
    "interval_seconds": 1.0
}

# Создание эксперимента
experiment = {
    "name": "test_experiment",
    "host": "example.com",
    "port": 8000,
    "load_profiles": [load_profile]
}

# Отправка запроса
response = await client.post("/create_experiment", json=experiment)
experiment_id = response.json()["experiment_id"]
```

### Создание группы экспериментов

```python
# Создание профилей нагрузки
load_profiles = [
    {
        "users": 100,
        "duration_seconds": 60,
        "interval_seconds": 1.0
    },
    {
        "users": 80,
        "duration_seconds": 60,
        "interval_seconds": 1.0
    },
    {
        "users": 30,
        "duration_seconds": 60,
        "interval_seconds": 1.0
    }
]

# Создание экспериментов для разных серверов
experiments = [
    {
        "name": "server1_test",
        "host": "server1.example.com",
        "port": 8000,
        "load_profiles": load_profiles
    },
    {
        "name": "server2_test",
        "host": "server2.example.com",
        "port": 8000,
        "load_profiles": load_profiles
    }
]

# Создание группы
group = {
    "name": "test_group",
    "experiments": experiments
}

# Отправка запроса
response = await client.post("/create_experiment_group", json=group)
group_id = response.json()["group_id"]
```

### Управление группой экспериментов

```python
# Запуск всех экспериментов в группе
await client.post(f"/manage_group?state=start&group_id={group_id}")

# Получение статистики
stats = await client.get(f"/group_stats?group_id={group_id}")
print(f"Average latency: {stats.average_latency}ms")
print(f"Success rate: {stats.success_rate}%")

# Пауза группы
await client.post(f"/manage_group?state=pause&group_id={group_id}")

# Остановка группы
await client.post(f"/manage_group?state=stop&group_id={group_id}")
```

### Добавление нового эксперимента в группу

```python
# Создание нового эксперимента
new_experiment = {
    "name": "server3_test",
    "host": "server3.example.com",
    "port": 8000,
    "load_profiles": load_profiles
}

# Добавление в группу
await client.post(f"/add_experiment_to_group?group_id={group_id}", json=new_experiment)
```

## Особенности реализации

- Асинхронное выполнение запросов через httpx
- Фоновое выполнение экспериментов с использованием asyncio.create_task
- Сериализация ObjectId и datetime для корректного представления в JSON
- Обработка исключений при невалидных запросах
- Поддержка динамической нагрузки через профили
- Возможность параллельного тестирования нескольких серверов

## Работа с базой данных

Сервис использует MongoDB для хранения экспериментов, групп и их результатов. Для работы с ObjectId используется кастомный JSON encoder.
