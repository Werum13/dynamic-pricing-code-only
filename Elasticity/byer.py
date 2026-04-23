import pandas as pd
import numpy as np
from prophet import Prophet
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


class Automated_warehouse:

    def __init__(self, data, date=None):
        self.data = data
        self.date = date if date is not None else (data['DATE_'].max() if 'DATE_' in data.columns else None)
        self.model = None
        self.sales = None
        self.errors = []
        self.rest = 0
        self.z = 1.65
        self.lead_time = 1
        self.last_forecast_date = None  # Дата последнего прогноза
        self.forecast_week = {}  # Прогноз на 7 дней: {0: pred_day_0, 1: pred_day_1, ...}
        self.store = pd.DataFrame(columns=[
                'ds','pred','demand','sale_real',
                'order','available','rest','safety_stock'
            ])
        self._processed = False  # Флаг: был ли уже сделан initial pipeline

    def pipeline(self):
        """
        Полный пайплайн для инициализации (используется только первый раз).
        """
        if self._processed:
            return self.store
        
        p = None
        self.prepocessor()
        
        for d in self.data['ds']:
            self.df = self.data[self.data['ds'] < d]

            if len(self.df) < 10:
                continue

            self.modeling()
            if (p := self.predict(d)) is not None:
                break
        
        self._processed = True
        return p

    def update_one_day(self, day, actual_sales=None):
        """
        Обновляет состояние склада на один день (инкрементально).
        
        Параметры
        ---------
        day : pd.Timestamp или datetime
            Дата для обновления
        actual_sales : float, optional
            Фактические продажи за день. Если None, используется 0.
        
        Возвращает
        ----------
        dict : запись о дне (pred, demand, order, rest, etc.)
        """
        day = pd.to_datetime(day)
        
        # Проверяем, нужно ли переобучать модель (раз в 7 дней)
        days_since_forecast = None
        if self.last_forecast_date is not None:
            days_since_forecast = (day - self.last_forecast_date).days
        
        need_retrain = (
            self.model is None or 
            days_since_forecast is None or 
            days_since_forecast >= 7
        )
        
        if need_retrain:
            # Переобучаем модель и делаем прогноз на неделю
            self._retrain_and_forecast(day)
        
        # Получаем прогноз на текущий день
        day_index = (day - self.last_forecast_date).days if self.last_forecast_date else 0
        pred = self.forecast_week.get(day_index, 0)
        
        # Фактический спрос
        demand = actual_sales if actual_sales is not None else 0
        
        # Обновляем ошибку прогноза
        if demand > 0:
            error = demand - pred
            self.errors.append(error)
        
        # Рассчитываем safety stock
        sigma = np.std(self.errors) if len(self.errors) > 1 else 0
        safety_stock = self.z * sigma * np.sqrt(self.lead_time)
        
        # Рассчитываем заказ (только если переобучаем модель)
        order = 0
        if need_retrain:
            # Прогноз спроса на неделю
            week_demand = sum(self.forecast_week.values())
            order = int(max(week_demand + safety_stock - self.rest, 0))
        
        available = self.rest + order
        sale_real = min(available, demand)
        self.rest = available - sale_real
        
        # Добавляем запись в store
        row = {
            'ds': day,
            'pred': pred,
            'demand': demand,
            'sale_real': sale_real,
            'order': order,
            'available': available,
            'rest': self.rest,
            'safety_stock': safety_stock
        }
        
        self.store = pd.concat([self.store, pd.DataFrame([row])], ignore_index=True)
        
        return row

    def _retrain_and_forecast(self, day):
        """
        Переобучает модель и делает прогноз на 7 дней вперёд.
        """
        # Обучаем модель на всех данных до текущего дня
        # Убеждаемся, что данные预处理рованы
        if 'ds' not in self.data.columns:
            self.prepocessor()
        
        self.df = self.data[self.data['ds'] < day]
        
        if len(self.df) < 10:
            # Недостаточно данных для обучения
            self.forecast_week = {i: 0 for i in range(7)}
            self.last_forecast_date = day
            return
        
        self.modeling()
        
        # Делаем прогноз на 7 дней
        future = self.model.make_future_dataframe(periods=7, freq='D', include_history=False)
        forecast = self.model.predict(future)
        
        # Сохраняем прогноз на неделю
        self.forecast_week = {}
        for i in range(7):
            if i < len(forecast):
                self.forecast_week[i] = forecast['yhat'].iloc[i]
            else:
                self.forecast_week[i] = 0
        
        self.last_forecast_date = day

    def save_store_csv(self, path):
        """
        Сохраняет таблицу store в CSV-файл.

        Параметры
        ---------
        path : str или Path
            Путь к CSV-файлу
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.store.to_csv(path, index=False)
        print(f'  📊 Store сохранён в CSV: {path}')

    def load_store_csv(self, path):
        """
        Загружает таблицу store из CSV-файла.

        Параметры
        ---------
        path : str или Path
            Путь к CSV-файлу

        Возвращает
        ----------
        bool : True если загрузка успешна, False если файл не найден
        """
        path = Path(path)
        if not path.exists():
            return False

        try:
            self.store = pd.read_csv(path)
            self.store['ds'] = pd.to_datetime(self.store['ds'])

            # Восстанавливаем состояние из последней записи
            last_row = self.store.iloc[-1]
            self.rest = last_row['rest']
            self.last_forecast_date = pd.to_datetime(last_row['ds'])

            # Пересчитываем errors из store
            if 'demand' in self.store.columns and 'pred' in self.store.columns:
                valid = self.store[self.store['demand'] > 0]
                self.errors = (valid['demand'] - valid['pred']).tolist()

            print(f'  📊 Store загружен из CSV: {path} ({len(self.store)} записей)')
            return True
        except Exception as e:
            print(f'  ❌ Ошибка загрузки store из CSV: {e}')
            return False

    def save_state(self, path):
        """
        Сохраняет состояние склада в файл.

        Параметры
        ---------
        path : str или Path
            Путь к файлу для сохранения
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'store': self.store,
            'rest': self.rest,
            'errors': self.errors,
            'last_forecast_date': self.last_forecast_date,
            'forecast_week': self.forecast_week,
            'model': self.model,
            '_processed': self._processed,
            'z': self.z,
            'lead_time': self.lead_time
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        print(f'  💾 Состояние склада сохранено: {path}')

    def load_state(self, path):
        """
        Загружает состояние склада из файла.

        Параметры
        ---------
        path : str или Path
            Путь к файлу для загрузки

        Возвращает
        ----------
        bool : True если загрузка успешна, False если файл не найден
        """
        path = Path(path)

        if not path.exists():
            print(f'  ⚠️ Файл состояния не найден: {path}')
            return False

        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)

            self.store = state.get('store', self.store)
            self.rest = state.get('rest', 0)
            self.errors = state.get('errors', [])
            self.last_forecast_date = state.get('last_forecast_date')
            self.forecast_week = state.get('forecast_week', {})
            self.model = state.get('model')
            self._processed = state.get('_processed', False)
            self.z = state.get('z', 1.65)
            self.lead_time = state.get('lead_time', 1)

            print(f'  📦 Состояние склада загружено (последняя дата: {self.last_forecast_date})')
            return True

        except Exception as e:
            print(f'  ❌ Ошибка загрузки состояния: {e}')
            return False
        



    def prepocessor(self):
        self.data['DATE_'] = pd.to_datetime(self.data['DATE_'], format='%Y-%m-%d')
        self.data.rename(columns={'DATE_': 'ds', 'AMOUNT': 'y'}, inplace=True)
        self.data = self.data.fillna(0)

        date1 = self.data['ds'].min()
        date2 = self.data['ds'].max()
        mydates = pd.date_range(date1, date2).to_frame(name='ds')

        self.data = mydates.merge(self.data, on='ds', how='left')

        self.data = self.data.resample('W', on='ds').sum().reset_index()
        self.data = self.data.sort_values('ds')


    

    def modeling(self):

        model = Prophet(
            growth='linear',
            seasonality_mode='additive',
            uncertainty_samples=0
            )
        model.add_country_holidays(country_name='TR')
        self.model = model.fit(self.df)

    def predict(self, d, period=1):
        future = self.model.make_future_dataframe(periods=period, freq='W', include_history=False)
        forecast = self.model.predict(future)
        
        pred = forecast['yhat'].iloc[0] 

        if d < self.date:
            
            demand_series = self.data.loc[self.data['ds'] == d, 'y']
            demand = demand_series.iloc[0] if not demand_series.empty else 0
            
           
            error = demand - pred
            self.errors.append(error)
            
            sigma = np.std(self.errors) if len(self.errors) > 1 else 0
            safety_stock = self.z * sigma * np.sqrt(self.lead_time)

            
            order = int(max(pred + safety_stock - self.rest, 0))
            available = int(self.rest + order)
            sale_real = min(available, demand)
            self.rest = available - sale_real

            row = pd.DataFrame({
                'ds': [d],
                'pred': [pred],
                'demand': [demand],
                'sale_real': [sale_real],
                'order': [order],
                'available': [available],
                'rest': [self.rest],
                'safety_stock': [safety_stock]
            })

            self.store = pd.concat([self.store, row], ignore_index=True)
        else: 
            return self.store[['ds', 'pred']]
        
    
    def paint_grafic(self):

        dd = self.store[['ds', 'pred', 'demand']]

        # Настройка стиля
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")

        # Рисуем линию реального спроса
        sns.lineplot(data=dd, x='ds', y='demand', label='Реальный спрос (Demand)', marker='o', color='gray', alpha=0.5)

        # Рисуем линию прогноза
        sns.lineplot(data=dd, x='ds', y='pred', label='Прогноз (Prediction)', marker='s', color='blue')

        # Оформление
        plt.title('Сравнение прогноза и реального спроса на складе', fontsize=15)
        plt.xlabel('Дата (Недели)', fontsize=12)
        plt.ylabel('Количество (Units)', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def print_store(self):
        print(self.store)
        
# data = pd.read_csv('df_107.csv')
# data = data.drop('Unnamed: 0', axis=1)

# auto = Automated_warehouse(data, pd.Timestamp('2022-01-10'))
# s = auto.pipeline()

# auto.print_store()

# print(s)

# print(data.columns)

def init_warehouse(data, itemcode=None, base_dir=None, state_path=None, store_csv_path=None):
    """
    Инициализирует склад: загружает состояние из CSV/PKL или создаёт новый.

    Приоритет загрузки:
      1. CSV-файл store (warehouse_store_{itemcode}.csv) — если существует
      2. PKL-файл состояния (warehouse_state_{itemcode}.pkl) — если существует
      3. Инициализация нового склада

    Параметры
    ---------
    data : pd.DataFrame
        Исторические данные для обучения (должны иметь колонку DATE_)
    itemcode : int, optional
        ID товара (используется для авто-определения путей)
    base_dir : Path, optional
        Базовая директория проекта (для авто-определения путей)
    state_path : Path, optional
        Путь к PKL-файлу состояния
    store_csv_path : Path, optional
        Путь к CSV-файлу store

    Возвращает
    ----------
    Automated_warehouse : инициализированный объект склада
    """
    warehouse = Automated_warehouse(data)

    # 1. Пробуем загрузить из CSV
    if store_csv_path is not None:
        if warehouse.load_store_csv(store_csv_path):
            warehouse._processed = True
            return warehouse

    # 2. Пробуем загрузить из PKL
    if state_path is not None:
        if warehouse.load_state(state_path):
            return warehouse

    # 3. Инициализируем новый
    print('  🔄 Инициализация нового склада...')
    warehouse.pipeline()

    return warehouse


def update_warehouse_day(warehouse, day, actual_sales=None, state_path=None, store_csv_path=None):
    """
    Обновляет склад на один день и сохраняет состояние.

    Параметры
    ---------
    warehouse : Automated_warehouse
        Объект склада
    day : pd.Timestamp
        Дата обновления
    actual_sales : float, optional
        Фактические продажи
    state_path : Path, optional
        Путь к PKL-файлу для сохранения состояния
    store_csv_path : Path, optional
        Путь к CSV-файлу для сохранения store

    Возвращает
    ----------
    dict : запись о дне
    """
    row = warehouse.update_one_day(day, actual_sales)

    # Сохраняем CSV (всегда — инкрементально)
    if store_csv_path is not None:
        warehouse.save_store_csv(store_csv_path)

    # Сохраняем PKL (тяжёлая операция, только если указан путь)
    if state_path is not None:
        warehouse.save_state(state_path)

    return row
