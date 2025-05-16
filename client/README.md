# LWMECPS Client Test App

Клиентский компонент для управления нагрузочным тестированием и сбора метрик.

## Особенности

- Управление экспериментами с динамической нагрузкой
- Группировка экспериментов
- Сбор и анализ метрик
- REST API для управления тестами

## Настройка

### Переменные окружения

| Переменная | Описание | Значение по умолчанию |
|------------|----------|----------------------|
| `host` | Хост для запуска клиента | `0.0.0.0` |
| `port` | Порт для запуска клиента | `8001` |
| `debug` | Режим отладки | `True` |
| `MONGODB_URI` | URI для подключения к MongoDB | `mongodb://mongodb:27017` |
| `DATABASE_NAME` | Имя базы данных | `network_checks` |

### API Endpoints

#### Управление экспериментами

- `POST /api/create_experiment` - Создание нового эксперимента
  ```json
  {
    "settings": {
      "hosts": [
        {"host": "server", "port": 8000}
      ],
      "timeout": 1.0,
      "load_profiles": [
        {
          "concurrent_users": 1,
          "request_interval": 1.0,
          "profile_duration": 60
        }
      ]
    }
  }
  ```

- `GET /api/experiment_stats?experiment_id=<id>` - Получение статистики эксперимента
- `POST /api/manage_experiment?state=<state>&experiment_id=<id>` - Управление экспериментом
  - `state`: `start`, `pause`, `stop`

#### Управление группами

- `POST /api/create_experiment_group` - Создание группы экспериментов
- `POST /api/add_experiments_to_group` - Добавление экспериментов в группу
- `GET /api/group_stats?group_id=<id>` - Получение статистики группы
- `POST /api/manage_group?state=<state>&group_id=<id>` - Управление группой

#### Дополнительные эндпоинты

- `GET /api/list_experiments` - Список всех экспериментов
- `GET /api/list_groups` - Список всех групп
- `POST /api/delete_experiments` - Удаление всех экспериментов
- `POST /api/delete_groups` - Удаление всех групп

## Запуск

```bash
# Запуск через Docker
docker-compose up client

# Запуск напрямую
python main.py
```

## Оптимизация

Для оптимизации производительности можно:
1. Настроить таймауты для запросов к серверам
2. Оптимизировать интервалы между запросами
3. Настроить количество одновременных пользователей
4. Оптимизировать длительность профилей нагрузки

## Мониторинг

Клиент собирает следующие метрики:
- Средняя латентность
- Количество успешных запросов
- Количество запросов в секунду
- Статистика по группам экспериментов 

## Примеры использования

### Создание эксперимента из файла

1. Создайте файл `experiment.json`:
```json
{
  "settings": {
    "hosts": [
      {"host": "server1", "port": 8000},
      {"host": "server2", "port": 8000}
    ],
    "timeout": 1.0,
    "load_profiles": [
      {
        "concurrent_users": 1,
        "request_interval": 1.0,
        "profile_duration": 60
      },
      {
        "concurrent_users": 5,
        "request_interval": 0.5,
        "profile_duration": 30
      }
    ]
  }
}
```

2. Создайте эксперимент:
```bash
curl -X POST "http://localhost:8001/api/create_experiment" \
  -H "Content-Type: application/json" \
  -d @experiment.json
```

### Работа с группами экспериментов

1. Создайте файл `group.json`:
```json
{
  "name": "Load Test Group",
  "experiment_ids": []
}
```

2. Создайте группу:
```bash
curl -X POST "http://localhost:8001/api/create_experiment_group" \
  -H "Content-Type: application/json" \
  -d @group.json
```

3. Добавьте эксперименты в группу:
```bash
curl -X POST "http://localhost:8001/api/add_experiments_to_group" \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "<group_id>",
    "experiment_ids": ["<experiment_id1>", "<experiment_id2>"]
  }'
```

4. Запустите группу экспериментов:
```bash
curl -X POST "http://localhost:8001/api/manage_group?state=start&group_id=<group_id>"
```

5. Проверьте статистику группы:
```bash
curl -X GET "http://localhost:8001/api/group_stats?group_id=<group_id>"
```

### Создание нескольких экспериментов из конфигурации

1. Создайте файл `experiments_config.json`:
```json
{
  "experiments": [
    {
      "settings": {
        "hosts": [{"host": "server1", "port": 8000}],
        "timeout": 1.0,
        "load_profiles": [
          {
            "concurrent_users": 1,
            "request_interval": 1.0,
            "profile_duration": 60
          }
        ]
      }
    },
    {
      "settings": {
        "hosts": [{"host": "server2", "port": 8000}],
        "timeout": 1.0,
        "load_profiles": [
          {
            "concurrent_users": 2,
            "request_interval": 0.5,
            "profile_duration": 30
          }
        ]
      }
    }
  ]
}
```

2. Создайте скрипт `create_experiments.sh`:
```bash
#!/bin/bash

# Создаем эксперименты
for exp in $(jq -c '.experiments[]' experiments_config.json); do
  exp_id=$(curl -s -X POST "http://localhost:8001/api/create_experiment" \
    -H "Content-Type: application/json" \
    -d "$exp" | jq -r '._id')
  echo "Created experiment: $exp_id"
  exp_ids+=("$exp_id")
done

# Создаем группу
group_id=$(curl -s -X POST "http://localhost:8001/api/create_experiment_group" \
  -H "Content-Type: application/json" \
  -d '{"name": "Auto Group"}' | jq -r '._id')

# Добавляем эксперименты в группу
curl -X POST "http://localhost:8001/api/add_experiments_to_group" \
  -H "Content-Type: application/json" \
  -d "{
    \"group_id\": \"$group_id\",
    \"experiment_ids\": [$(IFS=,; echo "${exp_ids[*]}")]
  }"

echo "Created group: $group_id"
```

3. Запустите скрипт:
```bash
chmod +x create_experiments.sh
./create_experiments.sh 