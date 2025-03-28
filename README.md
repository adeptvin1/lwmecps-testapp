# lwmecps-app

## Описание

Данный сервис представляет собой API на основе FastAPI, предназначенный для проведения экспериментов по измерению задержки (latency) удаленных серверов. Сервис позволяет создавать, управлять и анализировать эксперименты по тестированию производительности API.

## Возможности

- Создание экспериментов с заданными параметрами (хост, порт, интервал, количество проверок)
- Управление состоянием экспериментов (запуск, пауза, остановка)
- Измерение задержки удаленных API-эндпоинтов
- Сохранение результатов в MongoDB
- Получение списка всех экспериментов
- Получение детальных результатов конкретного эксперимента

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

- `host` - хост для тестирования
- `port` - порт для тестирования
- `interval` - интервал между запросами в секундах
- `count` - количество запросов для выполнения

**Возвращает:**

- Идентификатор созданного эксперимента

#### POST `/manage_experiment`

Управляет состоянием эксперимента.

**Параметры:**

- `state` - состояние эксперимента ("start", "pause", "stop")
- `experiment_id` - идентификатор эксперимента

**Возвращает:**

- Идентификатор эксперимента и его новое состояние

## Модели данных

### ExperimentSettings

- `host` - хост для тестирования
- `port` - порт для тестирования
- `interval` - интервал между запросами в секундах
- `count` - количество запросов для выполнения

### ExperimentResult

- `status_code` - HTTP-статус код ответа
- `latency_ms` - время задержки в миллисекундах
- `timestamp` - временная метка

### Experiment

- `id` - уникальный идентификатор эксперимента
- `settings` - настройки эксперимента (объект ExperimentSettings)
- `state` - состояние эксперимента ("created", "running", "paused", "completed")
- `results` - массив результатов (объекты ExperimentResult)
- `created_at` - дата и время создания
- `updated_at` - дата и время последнего обновления

## Примеры использования

### Создание нового эксперимента

```bash
POST http://localhost:8000/create_experiment?host=example.com&port=80&interval=5&count=10
```

### Запуск эксперимента

```bash
POST http://localhost:8000/manage_experiment?state=start&experiment_id=60a7b8c9d0e1f2g3h4i5j6k7
```

### Получение результатов эксперимента

```bash
GET http://localhost:8000/result_experiment?experiment_id=60a7b8c9d0e1f2g3h4i5j6k7
```

## Особенности реализации

- Асинхронное выполнение запросов через httpx
- Фоновое выполнение экспериментов с использованием asyncio.create_task
- Сериализация ObjectId и datetime для корректного представления в JSON
- Обработка исключений при невалидных запросах

## Работа с базой данных

Сервис использует MongoDB для хранения экспериментов и их результатов. Для работы с ObjectId используется кастомный JSON encoder.
