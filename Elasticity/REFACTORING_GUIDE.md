# Рефакторинг пайплайна ценообразования: Q(P) вместо R(P)

## 1. Почему Q(P) лучше R(P)

### Математика проблемы

Текущий пайплайн обучает модель на выручке:

```
R̂(P) ≈ R(P) = P · Q(P)
```

Затем восстанавливает спрос обратным делением:

```python
pred['qlt'] = pred['pred_GMV'] / pred['UNITPRICE']  # pipeline.py, строка ~180
```

**Проблема 1 — смещённый градиент.** Производная выручки по цене:
```
dR/dP = Q(P) + P · dQ/dP
```
Первое слагаемое `Q(P) > 0` всегда тянет выручку вверх при росте цены.
Модель, обученная на `R`, недооценивает эластичность: ей "кажется",
что цена почти не влияет на спрос (она видит только разницу слагаемых).

**Проблема 2 — деление на маленькие цены.** При `P → 0`:
```
Q̂(P) = R̂(P) / P → ∞
```
Даже небольшой шум в `R̂` при малых ценах создаёт огромные выбросы спроса.

**Проблема 3 — интерпретируемость.** Закон спроса `dQ/dP < 0` невозможно
проверить без дополнительного вычисления из `R`. Для Q он проверяется напрямую.

### Правильная постановка

```
Обучаем:  Q̂(P, features)           ← модель спроса (DemandModel)
Считаем:  R(P) = P · Q̂(P)          ← выручка (аналитически)
Считаем:  π(P) = (P−MC) · Q̂(P)    ← прибыль (аналитически)
Считаем:  ε(P) = dQ/dP · P/Q̂      ← эластичность (числ. производная)
Находим:  P* = argmax π(P)          ← grid search + scipy
```

---

## 2. Изменения в ETL.py

Целевая переменная меняется с `GMV_7D` (выручка) на `AMOUNT_7D_target`
(скользящая сумма количества за следующие 7 дней).

ETL уже считает `AMOUNT_7D` (лаговый признак), но **с `future=False`**.
Нужно добавить `future=True`:

```python
# В методе window_features(), после существующего цикла:
for days in [7]:                # достаточно 7-дневного горизонта
    window = f'{days}D'
    df[f'AMOUNT_{days}D_target'] = _rolling(
        'AMOUNT', 'sum', window, 'ITEMCODE', future=True  # ← future!
    )
```

В `feature_selection()` — **не удалять** `AMOUNT_7D_target`.

---

## 3. DemandModel.py (заменяет Elasticity.py)

### Ключевые отличия

| | Elasticity.py (старый) | DemandModel.py (новый) |
|---|---|---|
| Target | `GMV_7D` (выручка) | `AMOUNT_7D_target` (количество) |
| Предсказание | `pred_GMV` | `pred_Q` + `pred_GMV = P·Q` |
| Эластичность | В `Evaluation.py` через `GMV/P` | Встроена в модель |
| clip | `[0, ∞)` для GMV | `[0, ∞)` для Q |

### Использование в pipeline.py

```python
from DemandModel import demand_model
from ETL_patch import etl_with_demand_target

# Замена etl()
df = etl_with_demand_target(df)

# Замена target
target   = train_df_full['AMOUNT_7D_target'].copy()   # ← Q, не R

# Замена elasticity()
model_demand = demand_model(train=train_df, target=target)
```

### Расчёт эластичности (формула центральных разностей)

```python
# Встроен в DemandModel.elasticity():
delta = 0.01 * P                                    # 1% от цены
Q_up  = model.predict(data с P+delta)['pred_Q']
Q_dn  = model.predict(data с P-delta)['pred_Q']
ε̂    = (Q_up - Q_dn) / (2·delta) · P / Q̂         # O(δ²) точность
```

**Интерпретация:**
- `ε < −1` — товар **эластичен**: снижение цены увеличивает выручку
- `−1 < ε < 0` — товар **неэластичен**: повышение цены увеличивает выручку
- `ε = −1` — единичная эластичность: выручка максимальна (= оптимум R)
- `ε ≥ 0` — **аномалия**: нарушение закона спроса, нужна проверка данных

---

## 4. PriceOptimizer.py (заменяет optimizer.py)

### Почему убираем PyTorch/Optuna

Старый `optimizer.py` использует `torch.softmax` + градиентный спуск
для **выбора** цены из дискретной сетки кандидатов.
Это:
- Делает выбор цены **чёрным ящиком** (нельзя объяснить)
- Добавляет лишние гиперпараметры (`tau`, `lambd`, `alpha`)
- Требует PyTorch там, где достаточно NumPy

### Явная оптимизация

```python
# Шаг 1: Grid search (глобальный максимум)
P_grid  = np.linspace(P_min, P_max, n_grid=300)
Q_grid  = model.predict(template с каждой ценой)['pred_Q']
Q_real  = np.minimum(Q_grid, S)                     # ограничение склада
π_grid  = (P_grid - MC) * Q_real                    # целевая функция
P_star  = P_grid[argmax(π_grid)]

# Шаг 2: Уточнение через scipy (локальная точность)
P_star = minimize_scalar(−π(P), bounds=(P*−δ, P*+δ), method='bounded')
```

### Ограничение склада

```
Q_реализованный(P) = min(Q̂(P), S)
```

При `Q̂(P) > S` (склад стал узким местом):
- Оптимальная цена сдвигается **вверх**: при той же выручке можно продать
  меньше единиц по более высокой цене, не исчерпывая склад.
- `OptimizationResult.stock_binding = True` сигнализирует об этом.

### Использование в pipeline.py

```python
from PriceOptimizer import price_optimizer

# Замена строк с optimizer() и price_condidates()
d = price_optimizer(
    demand_model = model_demand,
    row_features = today_template,   # 1 строка, без ценовой сетки
    stock        = warehouse_record['available'],
    objective    = 'profit',
    n_grid       = 300,
)

optimal_price = round(float(d['unitprice']), 2)
realized_qty  = int(round(float(d['quantity'])))
print(f"ε(P*) = {d['elasticity']:.3f}")
```

---

## 5. Диаграмма нового пайплайна

```
Исторические данные
        │
        ▼
DataPreprocessor                   (без изменений)
        │
        ▼
ETL + demand target                (+ AMOUNT_7D_target)
        │
        ├──── X (признаки) ──────► DemandModel.fit(X, Q_target)
        │                                  │
        │                          Q̂(P, features)
        │                                  │
        ▼                                  ▼
    today_row ──────────────► PriceOptimizer
                                   │
                            π(P) = (P-MC)·min(Q̂(P), S)
                                   │
                            P* = argmax π(P)  [grid + scipy]
                                   │
                            ε(P*) = dQ/dP · P*/Q̂(P*)
                                   │
                                   ▼
                          {price, quantity, profit, ε}
```

---

## 6. Типичные ошибки и улучшения

### Ошибки

**Деление на маленькие P при численном дифференцировании:**
```python
delta = max(0.01 * P, 1e-8)   # ← обязательно clip снизу
```

**Шум в данных → шумная dQ/dP:**
Использовать `DemandModel.enforce_monotonicity()` перед вычислением
эластичности — сглаживает кривую через PCHIP-интерполяцию.

**Мультиколлинеарность полиномиальных признаков:**
LassoCV с `StandardScaler` справляется, но при добавлении новых
признаков проверьте VIF (variance inflation factor).

**target leak в временном ряду:**
`AMOUNT_7D_target` считается со сдвигом `−7D` (вперёд).
Убедитесь, что `cutoff_date = day − 7D` при формировании обучающей выборки —
иначе target "смотрит в будущее" при обучении (data leakage).

### Улучшения

**Log-преобразование Q:** Если спрос сильно правоскошен:
```python
target_log = np.log1p(target)   # обучать на log(1+Q)
# при предсказании: Q̂ = expm1(model.predict())
```

**Монотонность как soft constraint:**
Добавить в LassoCV монотонный дополнительный признак `−UNITPRICE`
с принудительно положительным коэффициентом (isotonic regression слой).

**Сезонность:** Уже учтена через `dow_sin`, `dom_sin`, `doy_sin` в ETL.
При предсказании на `today_row` эти признаки берутся из текущей даты.

**Доверительный интервал P*:**
Используйте bootstrap по строкам обучающей выборки
(10–20 resample) и получите интервал `[P*_low, P*_high]`.
Это помогает обосновать решение в диплоне.

---

## 7. Минимальный чеклист изменений в pipeline.py

- [ ] `from ETL_patch import etl_with_demand_target` вместо `from ETL import etl`
- [ ] `from DemandModel import demand_model` вместо `from Elasticity import elasticity`
- [ ] `from PriceOptimizer import price_optimizer` вместо `from optimizer import optimizer`
- [ ] `target = train_df_full['AMOUNT_7D_target']` вместо `GMV_7D`
- [ ] Убрать `price_condidates()` — PriceOptimizer сам строит сетку
- [ ] Убрать `pred['qlt'] = pred['pred_GMV'] / pred['UNITPRICE']`
- [ ] Убрать импорт `optuna`, `torch` из optimizer
- [ ] В `_evaluate_elasticity_model` → заменить `target_col='GMV_7D'`
  на `target_col='AMOUNT_7D_target'`
