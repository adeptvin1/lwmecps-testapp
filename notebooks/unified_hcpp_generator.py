#!/usr/bin/env python3
"""
Унифицированный генератор HCPP данных с OpenCellID калибровкой

Генератор сети базовых станций с использованием Hard-Core Point Process (HCPP) 
и распределение пользователей на основе профилей нагрузки.

Основные возможности:
- Генерация базовых станций с помощью HCPP в области 10x10 км
- Интеграция с OpenCellID для реалистичной калибровки параметров
- Распределение пользователей по профилю нагрузки
- Визуализация с диаграммами Вороного
- Анализ нагрузки по времени
- Сравнение стандартного и оптимизированного HCPP
- Сравнение распределений нагрузки между разными конфигурациями
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import json
from collections import defaultdict
import os
import requests
from typing import Tuple, List, Dict, Optional


class Config:
    """Центральная конфигурация для всех параметров генерации"""
    
    # === ОСНОВНЫЕ ПАРАМЕТРЫ СЕТИ ===
    AREA_SIZE = 10  # размер области в км
    NUM_BASE_STATIONS = 4  # количество базовых станций
    NUM_USERS = 100  # количество пользователей по умолчанию
    
    # === ПАРАМЕТРЫ HCPP ===
    # Стандартные параметры
    STANDARD_MIN_DISTANCE = 1.0  # минимальное расстояние между станциями (км)
    STANDARD_INTENSITY_LAMBDA = 1.0  # интенсивность процесса
    STANDARD_BETA = 0.1  # параметр взаимодействия
    
    # Параметры генерации
    MAX_GENERATION_ATTEMPTS = 10000  # максимальное количество попыток генерации
    RELAXATION_THRESHOLD = 5000  # порог для ослабления ограничений
    RELAXATION_FACTOR = 0.9  # коэффициент ослабления min_distance
    
    # === ПАРАМЕТРЫ OPENCELLID ===
    OPENCELLID_API_KEY = ""
    OPENCELLID_BASE_URL = "https://opencellid.org"
    OPENCELLID_LIMIT = 1000  # максимальное количество станций для загрузки
    OPENCELLID_TIMEOUT = 10  # таймаут запроса в секундах
    
    # Управление синтетическими данными
    USE_SYNTHETIC_DATA_FALLBACK = False  # отключить fallback к синтетическим данным
    
    # Регион по умолчанию для анализа паттернов
    DEFAULT_REGION = 'spb'  # основной регион для анализа паттернов
    
    # Предустановленные регионы для OpenCellID (min_lat, max_lat, min_lon, max_lon)
    # Увеличены области для получения данных из API
    OPENCELLID_REGIONS = {
        'moscow': (55.6, 55.9, 37.5, 37.8),      # Москва (центр)
        'spb': (59.8, 60.1, 30.2, 30.5),         # Санкт-Петербург (центр)
        'ekb': (56.7, 57.0, 60.4, 60.7),         # Екатеринбург (центр)
        'kazan': (55.7, 56.0, 49.0, 49.3),       # Казань (центр)
        'novosibirsk': (54.9, 55.2, 82.8, 83.1), # Новосибирск (центр)
    }
    
    # === ПАРАМЕТРЫ СИНТЕТИЧЕСКИХ ДАННЫХ ===
    SYNTHETIC_CLUSTERS = 8  # количество кластеров для синтетических данных
    SYNTHETIC_CLUSTER_SIZE_MIN = 15  # минимальный размер кластера
    SYNTHETIC_CLUSTER_SIZE_MAX = 35  # максимальный размер кластера
    SYNTHETIC_RADIUS_SCALE = 3  # масштаб радиуса для экспоненциального распределения
    
    # === ПАРАМЕТРЫ КЛАСТЕРИЗАЦИИ ===
    CLUSTERING_EPS = 2.0  # параметр eps для DBSCAN
    CLUSTERING_MIN_SAMPLES = 3  # минимальное количество образцов для DBSCAN
    CLUSTERING_HIGH_RATIO = 0.8  # высокий коэффициент кластеризации
    
    # === ПАРАМЕТРЫ КАЛИБРОВКИ ===
    CALIBRATION_MIN_DISTANCE_FACTOR = 0.9  # коэффициент для расчета min_distance
    CALIBRATION_MIN_DISTANCE_FLOOR = 0.01  # минимальное значение min_distance
    CALIBRATION_BETA_HIGH = 0.8  # beta для высокой кластеризации
    CALIBRATION_BETA_LOW = 1.0  # beta для низкой кластеризации
    
    # === ПАРАМЕТРЫ ВИЗУАЛИЗАЦИИ ===
    COLORS = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    FIGURE_SIZE_NETWORK = (10, 10)  # размер графика для отображения сети
    FIGURE_SIZE_COMPARISON = (18, 8)  # размер графика для сравнения сетей
    FIGURE_SIZE_LOAD_ANALYSIS = (16, 12)  # размер графика для анализа нагрузки
    FIGURE_SIZE_TIME_SERIES = (14, 8)  # размер графика для временных рядов
    FIGURE_SIZE_SPATIAL_ANALYSIS = (14, 11)  # размер графика для пространственного анализа
    FIGURE_SIZE_PARAMS = (10, 6)  # размер графика для сравнения параметров
    
    # === ПАРАМЕТРЫ ПОЛЬЗОВАТЕЛЕЙ ===
    USER_DISTRIBUTION_SCALE = 1.0  # масштаб для кластерного распределения
    USER_NORMAL_SCALE_FACTOR = 2.5  # делитель для масштаба нормального распределения
    USER_NOISE_STD = 5  # стандартное отклонение для шума в синтетическом профиле
    
    # === ПАРАМЕТРЫ ПРОФИЛЕЙ НАГРУЗКИ ===
    LOAD_PROFILE_PATHS = {
        'default': 'test_env/load_profile_with_deviation.json',
        '12h': 'test_env/load_profile_with_deviation_12h.json',
        '6h': 'test_env/load_profile_with_deviation_6h.json'
    }
    
    # Параметры синтетического профиля нагрузки
    SYNTHETIC_LOAD_DURATION_HOURS = 12  # продолжительность синтетического профиля
    SYNTHETIC_LOAD_INTERVAL_MINUTES = 10  # интервал между записями
    SYNTHETIC_LOAD_BASE = 60  # базовая нагрузка
    SYNTHETIC_LOAD_VARIATION = 40  # амплитуда вариации нагрузки
    SYNTHETIC_LOAD_MIN = 20  # минимальная нагрузка
    SYNTHETIC_LOAD_MAX = 120  # максимальная нагрузка
    
    # === ПАРАМЕТРЫ АНАЛИЗА ===
    DEFAULT_MAX_TIME_POINTS = 100  # максимальное количество точек для анализа временных рядов
    COMPARISON_MAX_TIME_POINTS = 4200  # максимальное количество точек для сравнения
    HISTOGRAM_BINS = 20  # количество бинов для гистограмм
    
    # === ПАРАМЕТРЫ ГЕНЕРАЦИИ SEED ===
    DEFAULT_SEED = 123  # основной seed для воспроизводимости
    SEED_OFFSET = 1  # смещение seed для различных генераторов
    SYNTHETIC_DATA_SEED = 123  # seed для синтетических данных
    
    @classmethod
    def get_default_hcpp_params(cls):
        """Возвращает стандартные параметры HCPP"""
        return {
            'min_distance': cls.STANDARD_MIN_DISTANCE,
            'intensity_lambda': cls.STANDARD_INTENSITY_LAMBDA,
            'beta': cls.STANDARD_BETA
        }
    
    @classmethod
    def get_load_profile_path(cls, profile_type='default'):
        """Возвращает путь к профилю нагрузки"""
        return cls.LOAD_PROFILE_PATHS.get(profile_type, cls.LOAD_PROFILE_PATHS['default'])
    
    @classmethod
    def get_color_for_station(cls, station_idx):
        """Возвращает цвет для станции по индексу"""
        return cls.COLORS[(station_idx - 1) % len(cls.COLORS)]
    
    @classmethod
    def get_region_bounds(cls, region_name):
        """Возвращает границы региона для OpenCellID"""
        return cls.OPENCELLID_REGIONS.get(region_name)
    
    @classmethod
    def list_available_regions(cls):
        """Возвращает список доступных регионов"""
        return list(cls.OPENCELLID_REGIONS.keys())
    
    @classmethod
    def get_region_info(cls, region_name):
        """Возвращает информацию о регионе"""
        bounds = cls.get_region_bounds(region_name)
        if bounds:
            min_lat, max_lat, min_lon, max_lon = bounds
            return {
                'name': region_name,
                'bounds': bounds,
                'center_lat': (min_lat + max_lat) / 2,
                'center_lon': (min_lon + max_lon) / 2,
                'area_lat': max_lat - min_lat,
                'area_lon': max_lon - min_lon
            }
        return None


class NetworkUtils:
    """Утилитарные функции для работы с сетями - устраняет дублирование кода"""
    
    @staticmethod
    def set_random_seed(seed):
        """Устанавливает seed для генератора случайных чисел"""
        if seed is not None:
            np.random.seed(seed)
    
    @staticmethod
    def generate_users(ue_count, area_size=None, base_stations=None, 
                      distribution='clustered', fixed_seed=None):
        """
        Генерирует пользователей с заданным распределением.
        Если area_size не указан, вычисляет его по расположению станций.
        """
        NetworkUtils.set_random_seed(fixed_seed)
        
        effective_area_size = area_size
        if effective_area_size is None:
            if base_stations is not None and len(base_stations) > 0:
                # Вычисляем по bounding box станций, чтобы пользователи были внутри сети
                max_coord = max(np.max(base_stations[:, 0]), np.max(base_stations[:, 1]))
                effective_area_size = max_coord * 1.05  # 5% отступ
            else:
                effective_area_size = Config.AREA_SIZE

        users = []
        if distribution == 'clustered' and base_stations is not None and len(base_stations) > 0:
            users_per_station = np.random.multinomial(ue_count, np.ones(len(base_stations))/len(base_stations))
            for i, count in enumerate(users_per_station):
                station_pos = base_stations[i]
                user_positions = np.random.normal(
                    loc=station_pos, 
                    scale=effective_area_size / (len(base_stations) * Config.USER_NORMAL_SCALE_FACTOR), 
                    size=(count, 2)
                )
                users.extend(user_positions)
        else: # Равномерное распределение
            users = np.random.uniform(0, effective_area_size, size=(ue_count, 2))

        return np.array(users)
    
    @staticmethod
    def plot_network_base(ax, base_stations, users=None, title="Network", 
                         area_size=None, show_voronoi=True):
        """
        Базовая функция для отображения сети - убирает дублирование визуализации
        """
        if area_size is None:
            area_size = Config.AREA_SIZE
            
        # Диаграмма Вороного
        if show_voronoi and len(base_stations) >= 3:
            try:
                vor = Voronoi(base_stations)
                voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                               line_width=2, line_alpha=0.6, point_size=0, ax=ax)
            except Exception as e:
                print(f"Voronoi error: {e}")
        
        # Базовые станции
        ax.scatter(base_stations[:, 0], base_stations[:, 1], 
                   c='red', marker='x', s=100, linewidths=3, label='Base Stations')
        
        # Подписи станций
        for i, (x, y) in enumerate(base_stations):
            ax.text(x, y+0.03, f'BS{i+1}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        # Пользователи
        if users is not None:
            ax.scatter(users[:, 0], users[:, 1], 
                       alpha=0.5, s=10, c='blue', label='Users')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X coordinate (km)')
        ax.set_ylabel('Y coordinate (km)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(0, area_size)
        ax.set_ylim(0, area_size)
    
    @staticmethod
    def get_color_for_station(station_idx):
        """Возвращает цвет для станции по индексу"""
        return Config.get_color_for_station(station_idx)
    
    @staticmethod
    def calculate_network_stats(base_stations):
        """Рассчитывает статистики сети - убирает дублирование расчетов"""
        if len(base_stations) < 2:
            return {'mean_distance': 0, 'std_distance': 0, 'coverage_area': 0}
        
        # Расстояния между станциями
        distances = cdist(base_stations, base_stations)
        np.fill_diagonal(distances, np.inf)
        nearest_distances = np.min(distances, axis=1)
        
        # Площадь покрытия (convex hull)
        try:
            hull = ConvexHull(base_stations)
            coverage_area = hull.volume  # В 2D это площадь
        except:
            coverage_area = 0
        
        return {
            'mean_distance': np.mean(nearest_distances),
            'std_distance': np.std(nearest_distances),
            'coverage_area': coverage_area
        }


class UnifiedHCPPGenerator:
    """Основной класс для генерации HCPP сетей"""
    
    def __init__(self, area_size=None, num_base_stations=None, min_distance=None, 
                 intensity_lambda=None, beta=None, num_users=None, region_bounds=None, region_name=None):
        self.area_size = area_size or Config.AREA_SIZE
        self.num_base_stations = num_base_stations or Config.NUM_BASE_STATIONS
        self.min_distance = min_distance or Config.STANDARD_MIN_DISTANCE
        self.intensity_lambda = intensity_lambda or Config.STANDARD_INTENSITY_LAMBDA
        self.beta = beta or Config.STANDARD_BETA
        self.num_users = num_users or Config.NUM_USERS
        
        # Устанавливаем регион из названия или границ
        if region_name:
            self.region_bounds = Config.get_region_bounds(region_name)
            self.region_name = region_name
            if self.region_bounds is None:
                print(f"Warning: Unknown region '{region_name}'. Available regions: {Config.list_available_regions()}")
        else:
            self.region_bounds = region_bounds
            self.region_name = None
        
        self.base_stations = None
        self.users = None
        self.opencellid_data = None
        self.opencellid_patterns = None
        
        # OpenCellID API
        self.api_key = Config.OPENCELLID_API_KEY
        self.base_url = Config.OPENCELLID_BASE_URL
        
        # HCPP параметры
        self.hcpp_params = {
            'standard': {
                'min_distance': self.min_distance,
                'intensity_lambda': self.intensity_lambda,
                'beta': self.beta
            },
            'optimized': {
                'min_distance': self.min_distance,
                'intensity_lambda': self.intensity_lambda,
                'beta': self.beta
            }
        }

    def intensity_function(self, x):
        return self.intensity_lambda

    def interaction_function(self, x, y):
        distance = np.linalg.norm(x - y)
        if distance < self.min_distance:
            return 0
        return np.exp(-self.beta * distance)

    def calculate_density(self, points):
        if len(points) == 0:
            return 1.0
        
        intensity_product = np.prod([self.intensity_function(p) for p in points])
        
        interaction_product = 1.0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                interaction_product *= self.interaction_function(points[i], points[j])
        
        return intensity_product * interaction_product

    def generate_base_stations(self, fixed_seed=None, num_iterations=None):
        """
        Генерирует базовые станции используя Hard-Core Point Process (HCPP)
        с учетом всех параметров: min_distance, intensity_lambda и beta.
        Подгоняет площадь генерации, чтобы получить требуемое количество станций (num_base_stations),
        сохраняя при этом откалиброванную плотность (intensity_lambda).
        """
        NetworkUtils.set_random_seed(fixed_seed)

        # Рассчитываем эффективную площадь, чтобы соответствовать num_base_stations
        # при сохранении откалиброванной плотности.
        if self.intensity_lambda <= 0:
            self.base_stations = np.array([])
            print("Warning: intensity_lambda is non-positive. Generated 0 base stations.")
            return self.base_stations
            
        required_area = self.num_base_stations / self.intensity_lambda
        full_area = self.area_size ** 2
        effective_area = min(required_area, full_area)
        
        # Рассчитываем сторону квадрата для эффективной площади
        effective_side_length = np.sqrt(effective_area)
        self.effective_side_length = effective_side_length

        # Целевое количество точек - это то, что мы задали
        target_points = self.num_base_stations
        
        # Если требуемая площадь больше доступной, мы не сможем сгенерировать
        # target_points, поэтому скорректируем цель.
        if required_area > full_area:
            target_points = int(full_area * self.intensity_lambda)
            print(f"Warning: Calibrated intensity is too low to generate {self.num_base_stations} stations in the given area. "
                  f"Targeting {target_points} stations instead.")
        
        print(f"HCPP Generation: target={target_points}, intensity={self.intensity_lambda:.3f}, "
              f"effective_area={effective_area:.2f}km² (side: {effective_side_length:.2f}km)")
        
        current_points = []
        attempts = 0
        max_attempts = Config.MAX_GENERATION_ATTEMPTS
        
        # Сохраняем оригинальное min_distance для сброса, если будет релаксация
        original_min_distance = self.min_distance

        while len(current_points) < target_points and attempts < max_attempts:
            attempts += 1
            
            # Генерируем новую точку в пределах эффективной площади
            new_point = np.random.uniform(0, effective_side_length, 2)
            
            # Проверяем hard-core ограничение (минимальное расстояние)
            valid_hard_core = True
            for existing_point in current_points:
                if np.linalg.norm(new_point - existing_point) < self.min_distance:
                    valid_hard_core = False
                    break
            
            if not valid_hard_core:
                continue
            
            # Проверяем взаимодействие (beta параметр)
            if len(current_points) > 0:
                interaction_prob = 1.0
                for existing_point in current_points:
                    distance = np.linalg.norm(new_point - existing_point)
                    # Используем функцию взаимодействия только если точки не слишком близко
                    interaction = self.interaction_function(new_point, existing_point)
                    interaction_prob *= interaction
                
                if np.random.random() > interaction_prob:
                    continue
            
            current_points.append(new_point)
            
            # Ослабляем ограничения если не можем найти точки
            if attempts > Config.RELAXATION_THRESHOLD and len(current_points) < target_points:
                self.min_distance *= Config.RELAXATION_FACTOR
                print(f"Relaxing min_distance to {self.min_distance:.3f}")
        
        # Возвращаем min_distance к исходному значению после генерации
        self.min_distance = original_min_distance
        
        self.base_stations = np.array(current_points)
        print(f"Generated {len(self.base_stations)} base stations in {attempts} attempts")
        return self.base_stations

    def generate_users(self, ue_count, distribution='clustered', fixed_seed=None):
        """Использует NetworkUtils для генерации пользователей"""
        # Используем effective_side_length если он есть, чтобы пользователи не выходили за пределы
        area = getattr(self, 'effective_side_length', self.area_size)
        self.users = NetworkUtils.generate_users(
            ue_count, area, self.base_stations, distribution, fixed_seed
        )
        return self.users

    def visualize_distribution(self, show_users=True, title="HCPP Base Station Distribution"):
        if self.base_stations is None:
            raise ValueError("Base stations not generated yet")

        fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE_NETWORK)
        
        users_to_show = self.users if show_users else None

        # Используем effective_side_length если он есть, чтобы график был по размеру
        plot_area = getattr(self, 'effective_side_length', self.area_size)
        
        NetworkUtils.plot_network_base(ax, self.base_stations, users_to_show, 
                                     title, plot_area)
        
        plt.tight_layout()
        plt.show()

    def set_region(self, region_name):
        """Устанавливает регион для анализа OpenCellID данных"""
        self.region_bounds = Config.get_region_bounds(region_name)
        self.region_name = region_name
        if self.region_bounds is None:
            available_regions = Config.list_available_regions()
            print(f"Error: Unknown region '{region_name}'")
            print(f"Available regions: {available_regions}")
            return False
        else:
            region_info = Config.get_region_info(region_name)
            print(f"Region set to: {region_name}")
            print(f"  Center: {region_info['center_lat']:.3f}, {region_info['center_lon']:.3f}")
            print(f"  Area: {region_info['area_lat']:.3f}° lat x {region_info['area_lon']:.3f}° lon")
            return True
    
    def get_current_region_info(self):
        """Возвращает информацию о текущем регионе"""
        if self.region_name:
            return Config.get_region_info(self.region_name)
        elif self.region_bounds:
            min_lat, max_lat, min_lon, max_lon = self.region_bounds
            return {
                'name': 'custom',
                'bounds': self.region_bounds,
                'center_lat': (min_lat + max_lat) / 2,
                'center_lon': (min_lon + max_lon) / 2,
                'area_lat': max_lat - min_lat,
                'area_lon': max_lon - min_lon
            }
        return None

    def load_opencellid_data(self, radio_type='LTE', use_synthetic_fallback=None):
        """Загружает данные OpenCellID"""
        if use_synthetic_fallback is None:
            use_synthetic_fallback = Config.USE_SYNTHETIC_DATA_FALLBACK
        
        if self.region_bounds is None:
            if use_synthetic_fallback:
                print("No region set. Generating synthetic data...")
                self.opencellid_data = self._generate_synthetic_data()
            else:
                raise ValueError("No region set and synthetic data fallback disabled")
        else:
            region_info = self.get_current_region_info()
            print(f"Loading OpenCellID data for region: {region_info['name']}")
            print(f"  Center: {region_info['center_lat']:.3f}, {region_info['center_lon']:.3f}")
            print(f"  Area: {region_info['area_lat']:.3f}° lat x {region_info['area_lon']:.3f}° lon")
            
            try:
                self.opencellid_data = self._fetch_opencellid_data(radio_type)
                print(f"Successfully loaded {len(self.opencellid_data)} real stations from OpenCellID")
            except Exception as e:
                print(f"OpenCellID error: {e}")
                if use_synthetic_fallback:
                    print("Falling back to synthetic data...")
                    self.opencellid_data = self._generate_synthetic_data()
                else:
                    print("Synthetic data fallback disabled - stopping")
                    raise e
        
        return self.opencellid_data

    def _generate_synthetic_data(self):
        """Генерирует синтетические данные на основе реальных паттернов"""
        # Используем region_name для создания разных данных для разных регионов
        region_seed = Config.SYNTHETIC_DATA_SEED
        if self.region_name:
            # Создаем разные seed для разных регионов
            region_hash = hash(self.region_name) % 1000
            region_seed = Config.SYNTHETIC_DATA_SEED + region_hash
            print(f"  Генерация синтетических данных для региона '{self.region_name}' (seed: {region_seed})")
        else:
            print(f"  Генерация синтетических данных (стандартный seed: {region_seed})")
        
        np.random.seed(region_seed)
        
        n_clusters = Config.SYNTHETIC_CLUSTERS
        stations = []
        
        for cluster_id in range(n_clusters):
            center_x = np.random.uniform(1, self.area_size - 1)
            center_y = np.random.uniform(1, self.area_size - 1)
            
            cluster_size = np.random.randint(Config.SYNTHETIC_CLUSTER_SIZE_MIN, 
                                           Config.SYNTHETIC_CLUSTER_SIZE_MAX)
            
            for _ in range(cluster_size):
                radius = np.random.exponential(Config.SYNTHETIC_RADIUS_SCALE)
                angle = np.random.uniform(0, 2 * np.pi)
                
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                
                x = np.clip(x, 0, self.area_size)
                y = np.clip(y, 0, self.area_size)
                
                stations.append([x, y])
        
        return np.array(stations)

    def _fetch_opencellid_data(self, radio_type='LTE'):
        """Загружает данные из локального датасета OpenCellID"""
        if self.region_bounds is None:
            raise ValueError("Region bounds not set")
        
        min_lat, max_lat, min_lon, max_lon = self.region_bounds
        
        # Путь к локальному датасету OpenCellID
        dataset_path = os.path.join(os.path.dirname(__file__), 'opencellid_data', '250.csv')
        
        # Создаем директорию если её нет
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        # Проверяем, есть ли датасет
        if not os.path.exists(dataset_path):
            print(f"  Local dataset not found: {dataset_path}")
            print("  Please place OpenCellID dataset in opencellid_data/250.csv")
            print("  Generating synthetic data instead...")
            return self._generate_synthetic_data()
        
        print(f"  Loading OpenCellID data from local dataset: {dataset_path}")
        
        try:
            # Загружаем данные из CSV файла (без заголовков)
            import pandas as pd
            # Определяем колонки согласно официальному формату OpenCellID
            columns = ['radio', 'mcc', 'net', 'area', 'cell', 'unit', 'lon', 'lat', 'range', 'samples', 'changeable', 'created', 'updated', 'averageSignal']
            df = pd.read_csv(dataset_path, header=None, names=columns)
            
            # Фильтруем данные по региону и типу сети
            mask = (
                (df['lat'] >= min_lat) & (df['lat'] <= max_lat) &
                (df['lon'] >= min_lon) & (df['lon'] <= max_lon) &
                (df['radio'] == radio_type)
            )
            
            filtered_data = df[mask]
            
            if len(filtered_data) > 0:
                print(f"  Found {len(filtered_data)} stations in dataset for region {self.region_name}")
                
                # Конвертируем координаты в локальную систему
                stations = []
                for _, row in filtered_data.iterrows():
                    lat = float(row['lat'])
                    lon = float(row['lon'])
                    
                    # Преобразуем координаты в локальную систему
                    x = (lat - min_lat) / (max_lat - min_lat) * self.area_size
                    y = (lon - min_lon) / (max_lon - min_lon) * self.area_size
                    
                    stations.append([x, y])
                
                return np.array(stations)
            else:
                print(f"  No stations found in dataset for region {self.region_name}")
                print("  Generating synthetic data instead...")
                return self._generate_synthetic_data()
                
        except Exception as e:
            print(f"  Error loading dataset: {e}")
            print("  Generating synthetic data instead...")
            return self._generate_synthetic_data()
    
    def _download_opencellid_dataset(self, dataset_path):
        """Скачивает данные OpenCellID через API по частям"""
        print("  Downloading OpenCellID data through API...")
        print("  This may take a while...")
        
        try:
            import pandas as pd
            
            # Создаем пустой DataFrame
            all_data = []
            
            # Скачиваем данные по регионам
            regions = [
                ('moscow', (55.6, 55.9, 37.5, 37.8)),
                ('spb', (59.8, 60.1, 30.2, 30.5)),
                ('ekb', (56.7, 57.0, 60.4, 60.7)),
                ('kazan', (55.7, 56.0, 49.0, 49.3)),
                ('novosibirsk', (54.9, 55.2, 82.8, 83.1))
            ]
            
            for region_name, (min_lat, max_lat, min_lon, max_lon) in regions:
                print(f"  Downloading data for {region_name}...")
                
                # Скачиваем данные для разных типов сетей
                for radio_type in ['LTE', 'GSM', 'UMTS']:
                    try:
                        api_data = self._download_region_data(min_lat, max_lat, min_lon, max_lon, radio_type)
                        if api_data:
                            all_data.extend(api_data)
                            print(f"    Added {len(api_data)} {radio_type} stations for {region_name}")
                    except Exception as e:
                        print(f"    Error downloading {radio_type} data for {region_name}: {e}")
                        continue
            
            if all_data:
                # Создаем DataFrame
                df = pd.DataFrame(all_data)
                df.to_csv(dataset_path, index=False)
                
                print(f"  Dataset saved to: {dataset_path}")
                print(f"  Total records: {len(df)}")
            else:
                raise Exception("No data downloaded from any region")
            
        except Exception as e:
            raise Exception(f"Failed to download dataset: {e}")
    
    def _download_region_data(self, min_lat, max_lat, min_lon, max_lon, radio_type):
        """Скачивает данные для конкретного региона и типа сети"""
        params = {
            'key': self.api_key,
            'bbox': f"{min_lat},{min_lon},{max_lat},{max_lon}",
            'radio': radio_type,
            'limit': Config.OPENCELLID_LIMIT,
            'format': 'json'
        }
        
        url = f"{self.base_url}/cell/getInArea"
        
        try:
            response = requests.get(url, params=params, timeout=Config.OPENCELLID_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                # Проверяем на ошибки в ответе
                if isinstance(data, dict) and 'error' in data:
                    return None  # Пропускаем регионы с ошибками
                
                # Проверяем разные форматы ответа
                cells_data = None
                if isinstance(data, dict) and 'cells' in data and len(data['cells']) > 0:
                    cells_data = data['cells']
                elif isinstance(data, list) and len(data) > 0:
                    cells_data = data
                elif isinstance(data, dict) and 'data' in data and len(data['data']) > 0:
                    cells_data = data['data']
                
                return cells_data if cells_data else None
            else:
                return None
                
        except Exception as e:
            return None

    def analyze_spatial_patterns(self, plot=True):
        """Анализирует пространственные паттерны данных"""
        if self.opencellid_data is None:
            raise ValueError("Load OpenCellID data first")
        
        print("Analyzing spatial patterns...")
        
        coords = self.opencellid_data
        
        area = self.area_size ** 2
        density = len(coords) / area
        
        nn_distances = self._calculate_nearest_neighbor_distances(coords)
        mean_nn_distance = np.mean(nn_distances)
        
        clustering_stats = self._analyze_clustering(coords)
        
        recommended_params = {
            'min_distance': max(Config.CALIBRATION_MIN_DISTANCE_FLOOR, 
                               mean_nn_distance * Config.CALIBRATION_MIN_DISTANCE_FACTOR),
            'intensity_lambda': density,
            'beta': Config.CALIBRATION_BETA_HIGH if clustering_stats['cluster_ratio'] > Config.CLUSTERING_HIGH_RATIO else Config.CALIBRATION_BETA_LOW
        }
        
        patterns = {
            'density': density,
            'mean_nn_distance': mean_nn_distance,
            'clustering_stats': clustering_stats,
            'recommended_hcpp_params': recommended_params
        }
        
        self.opencellid_patterns = patterns
        
        if plot:
            self._plot_spatial_analysis(coords, patterns)
        
        print(f"Results:")
        print(f"  Density: {density:.3f} stations/km²")
        print(f"  Mean NN distance: {mean_nn_distance:.3f} km")
        print(f"  Clustering ratio: {clustering_stats['cluster_ratio']:.3f}")
        
        return patterns

    def _calculate_nearest_neighbor_distances(self, coords):
        """Расчет расстояний до ближайших соседей"""
        distances = cdist(coords, coords)
        np.fill_diagonal(distances, np.inf)
        return np.min(distances, axis=1)

    def _analyze_clustering(self, coords):
        """Анализ кластеризации станций"""
        clustering = DBSCAN(eps=Config.CLUSTERING_EPS, min_samples=Config.CLUSTERING_MIN_SAMPLES).fit(coords)
        
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        n_noise = np.sum(clustering.labels_ == -1)
        cluster_ratio = (len(coords) - n_noise) / len(coords)
        
        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_ratio': cluster_ratio
        }

    def _plot_spatial_analysis(self, coords, patterns):
        """Визуализация пространственного анализа"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=Config.FIGURE_SIZE_SPATIAL_ANALYSIS)
        
        # Пространственное распределение
        ax1.scatter(coords[:, 0], coords[:, 1], alpha=0.6, color='blue')
        ax1.set_title('Spatial Distribution')
        ax1.set_xlabel('X coordinate (km)')
        ax1.set_ylabel('Y coordinate (km)')
        ax1.grid(True, alpha=0.3)
        
        # Гистограмма расстояний
        nn_distances = self._calculate_nearest_neighbor_distances(coords)
        ax2.hist(nn_distances, bins=30, alpha=0.7, color='green')
        ax2.axvline(patterns['mean_nn_distance'], color='red', 
                   linestyle='--', label=f'Mean: {patterns["mean_nn_distance"]:.2f} km')
        ax2.set_title('Nearest Neighbor Distances')
        ax2.set_xlabel('Distance (km)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # Кластерный анализ
        clustering = DBSCAN(eps=Config.CLUSTERING_EPS, min_samples=Config.CLUSTERING_MIN_SAMPLES).fit(coords)
        colors = plt.cm.tab10(clustering.labels_)
        ax3.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.6)
        ax3.set_title('Cluster Analysis (DBSCAN)')
        ax3.set_xlabel('X coordinate (km)')
        ax3.set_ylabel('Y coordinate (km)')
        
        # Статистика
        ax4.axis('off')
        stats_text = f"""Analysis Statistics:
Total stations: {len(coords)}
Density: {patterns['density']:.3f} stations/km²
Mean distance: {patterns['mean_nn_distance']:.3f} km
Clusters: {patterns['clustering_stats']['n_clusters']}
In clusters: {patterns['clustering_stats']['cluster_ratio']*100:.1f}%
Noise: {(1-patterns['clustering_stats']['cluster_ratio'])*100:.1f}%

Recommended HCPP params:
min_distance: {patterns['recommended_hcpp_params']['min_distance']:.3f} km
intensity_lambda: {patterns['recommended_hcpp_params']['intensity_lambda']:.3f}
beta: {patterns['recommended_hcpp_params']['beta']:.3f}"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

    def calibrate_hcpp_params(self, use_opencellid=True, plot=True):
        """Калибровка параметров HCPP"""
        if use_opencellid:
            if self.opencellid_patterns is None:
                print("Patterns not analyzed. Running analysis...")
                self.analyze_spatial_patterns(plot=False)
            
            calibrated_params = self.opencellid_patterns['recommended_hcpp_params'].copy()
        else:
            calibrated_params = self.hcpp_params['standard'].copy()
        
        self.hcpp_params['optimized'] = calibrated_params
        
        # Обновляем основные параметры
        self.min_distance = calibrated_params['min_distance']
        self.intensity_lambda = calibrated_params['intensity_lambda']
        self.beta = calibrated_params['beta']
        
        if plot:
            self._plot_parameter_comparison()
        
        print(f"Calibrated parameters:")
        for key, value in calibrated_params.items():
            print(f"  {key}: {value:.3f}")
        
        return calibrated_params

    def _plot_parameter_comparison(self):
        """Сравнение стандартных и оптимизированных параметров"""
        fig, ax = plt.subplots(1, 1, figsize=Config.FIGURE_SIZE_PARAMS)
        
        params = ['min_distance', 'intensity_lambda', 'beta']
        standard_values = [self.hcpp_params['standard'][p] for p in params]
        optimized_values = [self.hcpp_params['optimized'][p] for p in params]
        
        x = np.arange(len(params))
        width = 0.35
        
        ax.bar(x - width/2, standard_values, width, label='Standard', alpha=0.8)
        ax.bar(x + width/2, optimized_values, width, label='Optimized', alpha=0.8)
        
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Values')
        ax.set_title('HCPP Parameter Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(params)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class LoadProfileUserDistribution:
    """Класс для распределения пользователей по профилю нагрузки"""
    
    def __init__(self, load_profile_path, num_base_stations=None, create_if_missing=False):
        self.load_profile_path = load_profile_path
        self.num_base_stations = num_base_stations or Config.NUM_BASE_STATIONS
        self.load_profile = None
        self.create_if_missing = create_if_missing
        self.load_profile_data()

    def load_profile_data(self):
        try:
            with open(self.load_profile_path, 'r') as f:
                self.load_profile = json.load(f)
            
            # Проверяем, что профиль не пустой и имеет правильную структуру
            if not self.load_profile or not isinstance(self.load_profile, list):
                raise ValueError("Load profile is empty or has incorrect format")
                
            # Проверяем, что записи имеют нужные поля
            if not all('ue_count' in record and 'timestamp' in record for record in self.load_profile):
                raise ValueError("Load profile records missing required fields (ue_count, timestamp)")
                
            print(f"Load profile loaded: {len(self.load_profile)} records")
        except FileNotFoundError:
            print(f"Error: Load profile file not found: {self.load_profile_path}")
            if self.create_if_missing:
                print("Creating synthetic load profile...")
                self.load_profile = self._create_synthetic_load_profile()
            else:
                self.load_profile = None
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in load profile file: {e}")
            self.load_profile = None
        except Exception as e:
            print(f"Error loading profile: {e}")
            self.load_profile = None

    def _create_synthetic_load_profile(self, duration_hours=None, interval_minutes=None):
        """Создаёт синтетический профиль нагрузки для тестирования"""
        import math
        
        duration_hours = duration_hours or Config.SYNTHETIC_LOAD_DURATION_HOURS
        interval_minutes = interval_minutes or Config.SYNTHETIC_LOAD_INTERVAL_MINUTES
        
        total_minutes = duration_hours * 60
        intervals = total_minutes // interval_minutes
        
        profile = []
        for i in range(intervals):
            timestamp = i * interval_minutes * 60  # в секундах
            # Синусоидальная нагрузка с пиком в середине периода
            t_normalized = i / intervals
            base_load = Config.SYNTHETIC_LOAD_BASE
            variation = Config.SYNTHETIC_LOAD_VARIATION * math.sin(t_normalized * 2 * math.pi)
            ue_count = int(base_load + variation + np.random.normal(0, Config.USER_NOISE_STD))
            ue_count = max(Config.SYNTHETIC_LOAD_MIN, min(Config.SYNTHETIC_LOAD_MAX, ue_count))
            
            profile.append({
                'timestamp': timestamp,
                'ue_count': ue_count
            })
        
        print(f"Created synthetic load profile with {len(profile)} records")
        return profile

    def get_ue_count_at_time(self, time_index):
        if self.load_profile and 0 <= time_index < len(self.load_profile):
            return {
                'ue_count': self.load_profile[time_index]['ue_count'],
                'timestamp': self.load_profile[time_index]['timestamp']
            }
        return None

    def generate_users(self, ue_count, base_stations, distribution='clustered', fixed_seed=None):
        """Использует NetworkUtils для генерации пользователей"""
        # Не передаем area_size, чтобы он был вычислен автоматически по станциям
        return NetworkUtils.generate_users(
            ue_count, None, base_stations, distribution, fixed_seed
        )

    def distribute_users(self, time_index, base_stations, fixed_seed=None, distribution='clustered'):
        time_data = self.get_ue_count_at_time(time_index)
        if time_data is None:
            return None, None
        
        ue_count = time_data['ue_count']
        timestamp = time_data['timestamp']
        user_distribution = defaultdict(list)
        
        NetworkUtils.set_random_seed(fixed_seed + time_index if fixed_seed else None)
        
        users = self.generate_users(ue_count, base_stations, distribution=distribution, fixed_seed=fixed_seed)
        
        vor = Voronoi(base_stations)
        
        for user_pos in users:
            distances = np.linalg.norm(base_stations - user_pos, axis=1)
            nearest_station = np.argmin(distances)
            
            region_idx = vor.point_region[nearest_station]
            region = vor.regions[region_idx]
            
            if self._point_in_polygon(user_pos, vor.vertices[region]):
                user_distribution[nearest_station + 1].append(user_pos.tolist())
            else:
                for i, station in enumerate(base_stations):
                    region_idx = vor.point_region[i]
                    region = vor.regions[region_idx]
                    if self._point_in_polygon(user_pos, vor.vertices[region]):
                        user_distribution[i + 1].append(user_pos.tolist())
                        break
                else:
                    user_distribution[nearest_station + 1].append(user_pos.tolist())
                    
        return user_distribution, timestamp

    def _point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def visualize_distribution(self, time_index, base_stations, distribution):
        fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE_NETWORK)
        
        # Вычисляем размер области по станциям для корректного масштаба
        if base_stations is not None and len(base_stations) > 0:
            plot_area = max(np.max(base_stations[:, 0]), np.max(base_stations[:, 1])) * 1.05
        else:
            plot_area = Config.AREA_SIZE

        # Диаграмма Вороного
        vor = Voronoi(base_stations)
        voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', 
                        line_width=2, line_alpha=0.6, point_size=0, ax=ax)
        
        # Базовые станции
        ax.scatter(base_stations[:, 0], base_stations[:, 1], 
                   c='red', marker='x', s=100, label='Base Stations')
        
        for i, station_pos in enumerate(base_stations):
            ax.text(station_pos[0], station_pos[1] + 0.1, f"BS {i+1}", fontsize=9, ha='center')
        
        # Пользователи по станциям
        for station_idx, users in distribution.items():
            if users:
                users = np.array(users)
                color = NetworkUtils.get_color_for_station(station_idx)
                ax.scatter(users[:, 0], users[:, 1], 
                           alpha=0.5, s=10, c=color,
                           label=f'Users on BS{station_idx}')
        
        ax.set_xlim(0, plot_area)
        ax.set_ylim(0, plot_area)
        ax.set_title(f"User Distribution at t={timestamp}s (Total: {len(all_users)})")
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()


class NetworkComparator:
    """Класс для сравнения сетей и распределений нагрузки"""
    
    @staticmethod
    def compare_hcpp_networks(standard_generator, optimized_generator, load_dist, 
                             fixed_seed=None, title_prefix="HCPP Network Comparison"):
        """Сравнивает стандартную и оптимизированную сети HCPP"""
        
        if fixed_seed is None:
            fixed_seed = Config.DEFAULT_SEED
        
        # ИСПРАВЛЕНО: используем разные seed для демонстрации различий в параметрах
        standard_stations = standard_generator.generate_base_stations(fixed_seed=fixed_seed)
        optimized_stations = optimized_generator.generate_base_stations(fixed_seed=fixed_seed + Config.SEED_OFFSET)
        
        # Создаем subplot для сравнения
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Config.FIGURE_SIZE_COMPARISON)
        
        # Стандартная сеть
        NetworkComparator._plot_network_on_axis(ax1, standard_stations, "Standard HCPP", 
                                               standard_generator, load_dist, fixed_seed)
        
        # Оптимизированная сеть
        NetworkComparator._plot_network_on_axis(ax2, optimized_stations, "Optimized HCPP", 
                                               optimized_generator, load_dist, fixed_seed)
        
        plt.suptitle(title_prefix, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Статистика сравнения
        print("=== Network Statistics Comparison ===")
        standard_stats = NetworkUtils.calculate_network_stats(standard_stations)
        optimized_stats = NetworkUtils.calculate_network_stats(optimized_stations)
        
        print(f"Standard HCPP:")
        for key, value in standard_stats.items():
            print(f"  {key}: {value:.3f}")
        
        print(f"\nOptimized HCPP:")
        for key, value in optimized_stats.items():
            print(f"  {key}: {value:.3f}")
        
        return standard_stations, optimized_stations
    
    @staticmethod
    def _plot_network_on_axis(ax, base_stations, title, generator, load_dist, fixed_seed):
        """Отрисовывает одну сеть на заданной оси"""
        # Используем effective_side_length если доступно, чтобы график был по размеру
        plot_area = getattr(generator, 'effective_side_length', generator.area_size)

        NetworkUtils.plot_network_base(
            ax, base_stations, users=None, title=title, 
            area_size=plot_area, show_voronoi=True
        )
    
    @staticmethod
    def compare_load_distributions(standard_stations, optimized_stations, load_dist, 
                                  fixed_seed=None, max_time_points=None):
        """Сравнивает распределения нагрузки между стандартной и оптимизированной сетями"""
        
        if fixed_seed is None:
            fixed_seed = Config.DEFAULT_SEED
        if max_time_points is None:
            max_time_points = Config.COMPARISON_MAX_TIME_POINTS
        
        # ИСПРАВЛЕНО: используем разные seed для разных сетей
        standard_load_data = NetworkComparator._collect_load_data(
            standard_stations, load_dist, fixed_seed, max_time_points)
        optimized_load_data = NetworkComparator._collect_load_data(
            optimized_stations, load_dist, fixed_seed + Config.SEED_OFFSET, max_time_points)
        
        # Визуализация сравнения
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=Config.FIGURE_SIZE_LOAD_ANALYSIS)
        
        # График нагрузки по времени для стандартной сети
        NetworkComparator._plot_load_over_time(ax1, standard_load_data, "Standard HCPP Load Distribution")
        
        # График нагрузки по времени для оптимизированной сети
        NetworkComparator._plot_load_over_time(ax2, optimized_load_data, "Optimized HCPP Load Distribution")
        
        # Гистограммы распределения нагрузки
        NetworkComparator._plot_load_histograms(ax3, standard_load_data, optimized_load_data)
        
        # Статистика сравнения
        NetworkComparator._plot_load_statistics(ax4, standard_load_data, optimized_load_data)
        
        plt.suptitle("Load Distribution Comparison: Standard vs Optimized HCPP", 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Численная статистика
        NetworkComparator._print_load_statistics(standard_load_data, optimized_load_data)
        
        return standard_load_data, optimized_load_data
    
    @staticmethod
    def _collect_load_data(base_stations, load_dist, fixed_seed, max_time_points):
        """Собирает данные нагрузки для сети"""
        if load_dist.load_profile is None:
            raise ValueError("Load profile not loaded. Please check the load profile file path.")
        
        total_points = len(load_dist.load_profile)
        
        if max_time_points is None:
            step = 1
            time_indices = range(0, total_points)
        else:
            step = max(1, total_points // max_time_points)
            time_indices = range(0, total_points, step)
        
        timestamps = []
        users_per_station = {i: [] for i in range(1, len(base_stations) + 1)}
        
        for time_idx in time_indices:
            distribution, timestamp = load_dist.distribute_users(
                time_idx, base_stations, fixed_seed=fixed_seed)
            
            if timestamp is not None:
                timestamps.append(timestamp)
                for station_idx in range(1, len(base_stations) + 1):
                    user_count = len(distribution.get(station_idx, []))
                    users_per_station[station_idx].append(user_count)
        
        hours = [t / 3600 for t in timestamps]
        
        return {
            'timestamps': hours,
            'users_per_station': users_per_station,
            'num_stations': len(base_stations)
        }
    
    @staticmethod
    def _plot_load_over_time(ax, load_data, title):
        """Отображает нагрузку по времени"""
        for station_idx in range(1, load_data['num_stations'] + 1):
            color = NetworkUtils.get_color_for_station(station_idx)
            user_counts = load_data['users_per_station'][station_idx]
            
            ax.plot(load_data['timestamps'], user_counts,
                    label=f'BS{station_idx} (avg: {np.mean(user_counts):.1f})',
                    color=color, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Number of Users')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def _plot_load_histograms(ax, standard_data, optimized_data):
        """Сравнивает распределения нагрузки в виде гистограмм"""
        
        # Собираем все значения нагрузки
        standard_loads = []
        optimized_loads = []
        
        for station_idx in range(1, standard_data['num_stations'] + 1):
            standard_loads.extend(standard_data['users_per_station'][station_idx])
            optimized_loads.extend(optimized_data['users_per_station'][station_idx])
        
        # Строим гистограммы
        ax.hist(standard_loads, bins=Config.HISTOGRAM_BINS, alpha=0.6, label='Standard HCPP', color='blue')
        ax.hist(optimized_loads, bins=Config.HISTOGRAM_BINS, alpha=0.6, label='Optimized HCPP', color='red')
        
        ax.set_xlabel('Users per Station')
        ax.set_ylabel('Frequency')
        ax.set_title('Load Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def _plot_load_statistics(ax, standard_data, optimized_data):
        """Отображает статистическое сравнение"""
        
        # Рассчитываем статистики для каждой станции
        standard_stats = []
        optimized_stats = []
        
        for station_idx in range(1, standard_data['num_stations'] + 1):
            std_loads = standard_data['users_per_station'][station_idx]
            opt_loads = optimized_data['users_per_station'][station_idx]
            
            standard_stats.append({
                'mean': np.mean(std_loads),
                'std': np.std(std_loads),
                'max': np.max(std_loads),
                'min': np.min(std_loads)
            })
            
            optimized_stats.append({
                'mean': np.mean(opt_loads),
                'std': np.std(opt_loads),
                'max': np.max(opt_loads),
                'min': np.min(opt_loads)
            })
        
        # Сравниваем средние значения
        stations = [f'BS{i}' for i in range(1, standard_data['num_stations'] + 1)]
        standard_means = [s['mean'] for s in standard_stats]
        optimized_means = [s['mean'] for s in optimized_stats]
        
        x = np.arange(len(stations))
        width = 0.35
        
        ax.bar(x - width/2, standard_means, width, label='Standard', alpha=0.8, color='blue')
        ax.bar(x + width/2, optimized_means, width, label='Optimized', alpha=0.8, color='red')
        
        ax.set_xlabel('Base Stations')
        ax.set_ylabel('Average Users')
        ax.set_title('Average Load per Station')
        ax.set_xticks(x)
        ax.set_xticklabels(stations)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def _print_load_statistics(standard_data, optimized_data):
        """Выводит численную статистику сравнения"""
        print("\n=== Load Distribution Statistics ===")
        
        for station_idx in range(1, standard_data['num_stations'] + 1):
            std_loads = standard_data['users_per_station'][station_idx]
            opt_loads = optimized_data['users_per_station'][station_idx]
            
            print(f"\nBS{station_idx}:")
            print(f"  Standard  - Mean: {np.mean(std_loads):.1f}, Std: {np.std(std_loads):.1f}, Range: [{np.min(std_loads)}-{np.max(std_loads)}]")
            print(f"  Optimized - Mean: {np.mean(opt_loads):.1f}, Std: {np.std(opt_loads):.1f}, Range: [{np.min(opt_loads)}-{np.max(opt_loads)}]")
            
            # Статистическое сравнение
            load_difference = np.mean(opt_loads) - np.mean(std_loads)
            std_difference = np.std(opt_loads) - np.std(std_loads)
            
            print(f"  Difference - Mean: {load_difference:+.1f}, Std: {std_difference:+.1f}")
        
        # Общая статистика
        all_standard = []
        all_optimized = []
        
        for station_idx in range(1, standard_data['num_stations'] + 1):
            all_standard.extend(standard_data['users_per_station'][station_idx])
            all_optimized.extend(optimized_data['users_per_station'][station_idx])
        
        print(f"\nOverall Network:")
        print(f"  Standard  - Mean: {np.mean(all_standard):.1f}, Std: {np.std(all_standard):.1f}")
        print(f"  Optimized - Mean: {np.mean(all_optimized):.1f}, Std: {np.std(all_optimized):.1f}")
        print(f"  Total difference: {np.mean(all_optimized) - np.mean(all_standard):+.1f} users/station")


def plot_users_per_station_over_time(network, load_profile_dist, base_stations, 
                                    fixed_seed=None, max_time_points=None):
    """Функция для отображения пользователей по станциям во времени"""
    if load_profile_dist.load_profile is None:
        raise ValueError("Load profile not loaded. Please check the load profile file path.")
        return
    
    if fixed_seed is None:
        fixed_seed = Config.DEFAULT_SEED
    if max_time_points is None:
        max_time_points = Config.DEFAULT_MAX_TIME_POINTS
    
    total_points = len(load_profile_dist.load_profile)
    
    # Если max_time_points=None, показываем все точки
    if max_time_points is None:
        step = 1
        time_indices = range(0, total_points)
        print(f"Showing all {total_points} time points")
    else:
        step = max(1, total_points // max_time_points)
        time_indices = range(0, total_points, step)
        print(f"Showing {len(time_indices)} time points out of {total_points}")
    
    timestamps = []
    users_per_station = {i: [] for i in range(1, len(base_stations) + 1)}
    
    for time_idx in time_indices:
        distribution, timestamp = load_profile_dist.distribute_users(
            time_idx, base_stations, fixed_seed=fixed_seed)
        
        if timestamp is not None:
            timestamps.append(timestamp)
            for station_idx in range(1, len(base_stations) + 1):
                user_count = len(distribution.get(station_idx, []))
                users_per_station[station_idx].append(user_count)
    
    hours = [t / 3600 for t in timestamps]
    
    plt.figure(figsize=Config.FIGURE_SIZE_TIME_SERIES)
    
    for station_idx in range(1, len(base_stations) + 1):
        color = NetworkUtils.get_color_for_station(station_idx)
        user_counts = users_per_station[station_idx]
        
        plt.plot(hours, user_counts, 
                label=f'BS{station_idx} (avg: {np.mean(user_counts):.1f})',
                color=color, linewidth=1)
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Number of Users')
    plt.title('Number of Users per Base Station Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Station statistics:")
    for station_idx in range(1, len(base_stations) + 1):
        counts = users_per_station[station_idx]
        if counts:
            print(f"   BS{station_idx}: mean={np.mean(counts):.1f}, "
                  f"min={np.min(counts)}, max={np.max(counts)}, "
                  f"std={np.std(counts):.1f}")


# === ОСНОВНЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ С OPENCELLID И HCPP ===

def analyze_opencellid_patterns(region_name=None, radio_type='LTE', show_plots=True, use_synthetic_fallback=None):
    """
    Анализирует паттерны OpenCellID для указанного региона
    
    Args:
        region_name: название региона из Config.OPENCELLID_REGIONS (по умолчанию Config.DEFAULT_REGION)
        radio_type: тип сети ('LTE', 'GSM', 'UMTS', 'CDMA')
        show_plots: показывать графики анализа
        use_synthetic_fallback: использовать синтетические данные если API недоступен (по умолчанию Config.USE_SYNTHETIC_DATA_FALLBACK)
    
    Returns:
        dict: результаты анализа с генератором, паттернами и калиброванными параметрами
    """
    if region_name is None:
        region_name = Config.DEFAULT_REGION
    
    print(f"=== Анализ паттернов OpenCellID для региона: {region_name} ===")
    
    # Создаем генератор для региона
    generator = UnifiedHCPPGenerator(region_name=region_name)
    
    # Загружаем данные OpenCellID
    print(f"Загрузка данных OpenCellID (тип: {radio_type})...")
    try:
        generator.load_opencellid_data(radio_type=radio_type, use_synthetic_fallback=use_synthetic_fallback)
        
        if generator.opencellid_data is None:
            print("Ошибка: не удалось загрузить данные OpenCellID")
            return None
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return None
    
    print(f"Загружено {len(generator.opencellid_data)} станций")
    
    # Анализируем пространственные паттерны
    print("Анализ пространственных паттернов...")
    patterns = generator.analyze_spatial_patterns(plot=show_plots)
    
    # Калибруем параметры HCPP
    print("Калибровка параметров HCPP...")
    calibrated_params = generator.calibrate_hcpp_params(plot=show_plots)
    
    return {
        'generator': generator,
        'patterns': patterns,
        'calibrated_params': calibrated_params,
        'region_name': region_name
    }


def compare_standard_vs_optimized_hcpp(region_name=None, radio_type='LTE', show_plots=True, use_synthetic_fallback=None):
    """
    Сравнивает стандартный и оптимизированный HCPP с данными OpenCellID
    
    Args:
        region_name: название региона из Config.OPENCELLID_REGIONS (по умолчанию Config.DEFAULT_REGION)
        radio_type: тип сети ('LTE', 'GSM', 'UMTS', 'CDMA')
        show_plots: показывать графики сравнения
        use_synthetic_fallback: использовать синтетические данные если API недоступен (по умолчанию Config.USE_SYNTHETIC_DATA_FALLBACK)
    
    Returns:
        dict: результаты сравнения с генераторами и распределением пользователей
    """
    if region_name is None:
        region_name = Config.DEFAULT_REGION
    
    print(f"=== Сравнение стандартного и оптимизированного HCPP для региона: {region_name} ===")
    
    # Создаем стандартный генератор (БЕЗ региона - всегда стандартные параметры)
    standard_generator = UnifiedHCPPGenerator()
    
    # Создаем оптимизированный генератор с калибровкой на OpenCellID
    optimized_generator = UnifiedHCPPGenerator(region_name=region_name)
    
    # Калибруем оптимизированный генератор
    try:
        optimized_generator.load_opencellid_data(radio_type=radio_type, use_synthetic_fallback=use_synthetic_fallback)
        optimized_generator.analyze_spatial_patterns(plot=False)
        calibrated_params = optimized_generator.calibrate_hcpp_params(plot=False)
    except Exception as e:
        print(f"Ошибка калибровки оптимизированного генератора: {e}")
        return None
    
    # Создаем распределение пользователей
    load_dist = LoadProfileUserDistribution(
        Config.get_load_profile_path('default'),
        create_if_missing=True
    )
    
    if show_plots:
        # Сравниваем сети визуально
        print("Сравнение топологий сетей...")
        NetworkComparator.compare_hcpp_networks(
            standard_generator, optimized_generator, load_dist,
            title_prefix=f"Standard vs Optimized HCPP: {region_name}"
        )
        
        # Сравниваем распределения нагрузки
        print("Сравнение распределений нагрузки...")
        standard_stations = standard_generator.generate_base_stations(fixed_seed=Config.DEFAULT_SEED)
        optimized_stations = optimized_generator.generate_base_stations(fixed_seed=Config.DEFAULT_SEED + Config.SEED_OFFSET)
        
        NetworkComparator.compare_load_distributions(
            standard_stations, optimized_stations, load_dist,
            fixed_seed=Config.DEFAULT_SEED
        )
    
    print(f"\nКалиброванные параметры для {region_name}:")
    for key, value in calibrated_params.items():
        print(f"  {key}: {value:.3f}")
    
    return {
        'standard_generator': standard_generator,
        'optimized_generator': optimized_generator,
        'load_distribution': load_dist,
        'calibrated_params': calibrated_params,
        'region_name': region_name
    }


# === УПРАВЛЕНИЕ СИНТЕТИЧЕСКИМИ ДАННЫМИ ===

def enable_synthetic_data(enabled=True):
    """Включить/выключить использование синтетических данных как fallback"""
    Config.USE_SYNTHETIC_DATA_FALLBACK = enabled
    status = "включено" if enabled else "выключено"
    print(f"Использование синтетических данных: {status}")


def disable_synthetic_data():
    """Выключить использование синтетических данных"""
    enable_synthetic_data(False)


def get_synthetic_data_status():
    """Получить текущий статус использования синтетических данных"""
    return Config.USE_SYNTHETIC_DATA_FALLBACK


# === ДЕМОНСТРАЦИЯ ВЛИЯНИЯ OPENCELLID КАЛИБРОВКИ ===

def demonstrate_opencellid_calibration_impact(region_name=None, show_plots=True):
    """
    Демонстрирует влияние OpenCellID калибровки на параметры HCPP
    
    Args:
        region_name: название региона для анализа
        show_plots: показывать графики сравнения
    
    Returns:
        dict: результаты демонстрации
    """
    if region_name is None:
        region_name = Config.DEFAULT_REGION
    
    print(f"=== Демонстрация влияния OpenCellID калибровки для региона: {region_name} ===")
    
    # 1. Создаем стандартный генератор (без калибровки)
    standard_generator = UnifiedHCPPGenerator()
    print(f"\n1. Стандартные параметры HCPP:")
    print(f"   min_distance: {standard_generator.min_distance:.3f}")
    print(f"   intensity_lambda: {standard_generator.intensity_lambda:.3f}")
    print(f"   beta: {standard_generator.beta:.3f}")
    
    # 2. Создаем оптимизированный генератор с калибровкой
    optimized_generator = UnifiedHCPPGenerator(region_name=region_name)
    
    # Загружаем и анализируем данные OpenCellID
    try:
        optimized_generator.load_opencellid_data(use_synthetic_fallback=True)
        patterns = optimized_generator.analyze_spatial_patterns(plot=show_plots)
        calibrated_params = optimized_generator.calibrate_hcpp_params(plot=show_plots)
        
        print(f"\n2. Калиброванные параметры HCPP (на основе {region_name}):")
        print(f"   min_distance: {calibrated_params['min_distance']:.3f}")
        print(f"   intensity_lambda: {calibrated_params['intensity_lambda']:.3f}")
        print(f"   beta: {calibrated_params['beta']:.3f}")
        
        # 3. Генерируем сети с разными параметрами
        print(f"\n3. Генерация сетей с разными параметрами...")
        
        # Стандартная сеть
        standard_stations = standard_generator.generate_base_stations(fixed_seed=Config.DEFAULT_SEED)
        
        # Оптимизированная сеть
        optimized_stations = optimized_generator.generate_base_stations(fixed_seed=Config.DEFAULT_SEED + Config.SEED_OFFSET)
        
        # 4. Сравниваем статистики сетей
        print(f"\n4. Сравнение статистик сетей:")
        standard_stats = NetworkUtils.calculate_network_stats(standard_stations)
        optimized_stats = NetworkUtils.calculate_network_stats(optimized_stations)
        
        print(f"   Стандартная сеть:")
        print(f"     Среднее расстояние: {standard_stats['mean_distance']:.3f} км")
        print(f"     Площадь покрытия: {standard_stats['coverage_area']:.3f} км²")
        
        print(f"   Оптимизированная сеть:")
        print(f"     Среднее расстояние: {optimized_stats['mean_distance']:.3f} км")
        print(f"     Площадь покрытия: {optimized_stats['coverage_area']:.3f} км²")
        
        # 5. Визуализация различий
        if show_plots:
            print(f"\n5. Визуализация различий...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Config.FIGURE_SIZE_COMPARISON)
            
            # Стандартная сеть
            NetworkUtils.plot_network_base(ax1, standard_stations, None, 
                                         "Standard HCPP\n(Default Parameters)", 
                                         getattr(standard_generator, 'effective_side_length', standard_generator.area_size))
            
            # Оптимизированная сеть
            NetworkUtils.plot_network_base(ax2, optimized_stations, None, 
                                         f"Optimized HCPP\n(OpenCellID Calibrated - {region_name})", 
                                         getattr(optimized_generator, 'effective_side_length', optimized_generator.area_size))
            
            plt.suptitle(f"Impact of OpenCellID Calibration: {region_name}", 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # График сравнения параметров
            fig, ax = plt.subplots(1, 1, figsize=Config.FIGURE_SIZE_PARAMS)
            
            params = ['min_distance', 'intensity_lambda', 'beta']
            standard_values = [standard_generator.min_distance, 
                             standard_generator.intensity_lambda, 
                             standard_generator.beta]
            optimized_values = [calibrated_params['min_distance'], 
                              calibrated_params['intensity_lambda'], 
                              calibrated_params['beta']]
            
            x = np.arange(len(params))
            width = 0.35
            
            ax.bar(x - width/2, standard_values, width, label='Standard', alpha=0.8, color='blue')
            ax.bar(x + width/2, optimized_values, width, label=f'Optimized ({region_name})', alpha=0.8, color='red')
            
            ax.set_xlabel('HCPP Parameters')
            ax.set_ylabel('Values')
            ax.set_title('Parameter Comparison: Standard vs OpenCellID Calibrated')
            ax.set_xticks(x)
            ax.set_xticklabels(params)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        print(f"\n=== Демонстрация завершена ===")
        print(f"OpenCellID калибровка успешно повлияла на параметры HCPP!")
        print(f"Различия в параметрах приводят к различным топологиям сетей.")
        
        return {
            'standard_generator': standard_generator,
            'optimized_generator': optimized_generator,
            'standard_stations': standard_stations,
            'optimized_stations': optimized_stations,
            'calibrated_params': calibrated_params,
            'patterns': patterns,
            'region_name': region_name
        }
        
    except Exception as e:
        print(f"Ошибка в демонстрации: {e}")
        return None


# === ПРИМЕР ИСПОЛЬЗОВАНИЯ ===

if __name__ == "__main__":
    print("Unified HCPP Generator - OpenCellID Pattern Analysis")
    print("=" * 60)
    
    try:
        enable_synthetic_data(True)  # Разрешаем использовать синтетические данные, если реальные не найдены
        region = Config.DEFAULT_REGION

        # --- Шаг 1: Анализ и калибровка ---
        print(f"\n1. Анализ OpenCellID и калибровка для региона: {region}...")
        optimized_generator = UnifiedHCPPGenerator(region_name=region)
        optimized_generator.load_opencellid_data(use_synthetic_fallback=True)
        # Показываем 4-панельный график анализа OpenCellID
        optimized_generator.analyze_spatial_patterns(plot=True) 
        # Показываем столбчатый график сравнения параметров
        optimized_generator.calibrate_hcpp_params(plot=False) # Не показываем старый график
        
        # --- Шаг 2: Создание стандартного генератора ---
        standard_generator = UnifiedHCPPGenerator()
        
        # --- Шаг 3: Визуальное сравнение ПАРАМЕТРОВ (Новый, улучшенный график) ---
        print("\n2. Визуальное сравнение параметров HCPP...")
        fig, ax = plt.subplots(1, 1, figsize=Config.FIGURE_SIZE_PARAMS)
        
        params = ['min_distance', 'intensity_lambda', 'beta']
        standard_values = [getattr(standard_generator, p) for p in params]
        optimized_values = [getattr(optimized_generator, p) for p in params]
        
        x = np.arange(len(params))
        width = 0.35
        
        ax.bar(x - width/2, standard_values, width, label='Standard', alpha=0.8, color='blue')
        ax.bar(x + width/2, optimized_values, width, label=f'Optimized ({region})', alpha=0.8, color='red')
        
        ax.set_xlabel('HCPP Parameters')
        ax.set_ylabel('Values')
        ax.set_title('HCPP Parameter Comparison: Standard vs Calibrated')
        ax.set_xticks(x)
        ax.set_xticklabels(params)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # --- Шаг 4: Генерация сетей и пользователей---
        print("\n3. Генерация стандартной и оптимизированной сетей и пользователей...")
        standard_stations = standard_generator.generate_base_stations(fixed_seed=Config.DEFAULT_SEED)
        standard_users = standard_generator.generate_users(Config.NUM_USERS, fixed_seed=Config.DEFAULT_SEED)
        
        optimized_stations = optimized_generator.generate_base_stations(fixed_seed=Config.DEFAULT_SEED + Config.SEED_OFFSET)
        optimized_users = optimized_generator.generate_users(Config.NUM_USERS, fixed_seed=Config.DEFAULT_SEED + Config.SEED_OFFSET)
        
        # --- Шаг 5: Визуальное сравнение сетей ---
        print("\n4. Визуальное сравнение сгенерированных сетей...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Config.FIGURE_SIZE_COMPARISON)
        
        plot_area_std = getattr(standard_generator, 'effective_side_length', standard_generator.area_size)
        NetworkUtils.plot_network_base(ax1, standard_stations, standard_users, "Standard HCPP", plot_area_std)
        
        plot_area_opt = getattr(optimized_generator, 'effective_side_length', optimized_generator.area_size)
        NetworkUtils.plot_network_base(ax2, optimized_stations, optimized_users, f"Optimized HCPP ({region})", plot_area_opt)
        
        plt.suptitle(f"Impact of OpenCellID Calibration: {region}", fontsize=16, fontweight='bold')
        # Исправляем "уплывшие" заголовки, давая место для suptitle
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # --- Шаг 6: Сравнение распределения нагрузки ---
        print("\n5. Сравнение распределения нагрузки...")
        load_dist = LoadProfileUserDistribution(Config.get_load_profile_path('default'), create_if_missing=True)
        NetworkComparator.compare_load_distributions(
            standard_stations, optimized_stations, load_dist,
            fixed_seed=Config.DEFAULT_SEED
        )
        
        print("\n✅ Демонстрация завершена!")

    except Exception as e:
        print(f"\n❌ Ошибка в основном процессе: {e}")
        import traceback
        traceback.print_exc()


 