# Plan: X5 Next-Day ML Roadmap

## 0) Цель
Собрать устойчивый офлайн-процесс:
`ETL (sandbox) -> model-ready dataset -> обучение 3+ моделей -> отчёт -> сигнал long/short`.

## 1) Что использовать прямо сейчас (MVP)

### Данные
- `X5 1d` (обязательно)
- `IMOEX 1d` (контекст)
- `USDRUB 1d` (контекст)
- `5m` пока хранить, но в первой версии модели не использовать напрямую.

### Таргет
- `y = log(close_{t+1}/close_t)`
- `y_dir = 1[y > 0]`

### Модели
- `Ridge` (baseline regression)
- `LogisticRegression` (baseline direction)
- `LightGBMRegressor` (main)
- `LightGBM quantile upper/lower` (selector/filter)

### Метрики
- Regression: `MAE`, `RMSE`, `IC Pearson`, `IC Spearman`
- Direction: `accuracy`, `balanced accuracy`
- Strategy: `Sharpe`, `CAGR`, `Max Drawdown`, `Exposure`, `Hit Rate`

## 2) Что подходит для X5 из индикаторов (приоритет)

### Итерация 1 (добавить в первую очередь)
- `ATR(14)` и `ATR%`
- `OBV`
- `A/D Line (Chaikin)`

### Итерация 2
- `MFI(14)`
- intraday-агрегаты из `5m` (см. ниже)

### Итерация 3
- `VWAP daily` из `5m`
- intraday volatility/range features (агрегированные в день)

## 3) Порядок работы (рекомендуемый)

1. Обновить ETL и проверить целостность данных  
   Команды:
   - `make run` (обычно)
   - `make run-x5` / `make run-usdrub` (точечно)
   - `make run-full` (редко, при пересборке)

2. Подготовить 1D raw для ML  
   Команда:
   - `make ml-build-raw`

3. Собрать model-ready датасет  
   Команда:
   - `make ml-prepare`

4. Обучить модели и получить отчёт  
   Команды:
   - `make ml-train`
   - `make ml-report`

5. Проверить сигнал на последнюю дату  
   Команда:
   - `make ml-predict`

6. Только после этого менять фичи/параметры  
   Одно изменение за итерацию, затем снова пункты 3-5.

## 4) Что проверять перед тюнингом

- Диапазон дат и число строк в `ml-report`
- Доля пропусков после feature engineering
- Аномалии OHLCV:
  - `high < low`
  - нулевые объёмы
  - экстремальные выбросы доходности
- Реальная активность стратегии:
  - если `exposure ~ 0`, порог/selector слишком строгий

## 5) Тюнинг модели (в таком порядке)

1. `learning_rate` + `n_estimators` + early stopping  
2. `min_child_samples` (главный анти-overfit)  
3. `num_leaves` / `max_depth`  
4. `subsample` / `colsample_bytree`  
5. `reg_lambda`

Правило: сравнивать с baseline (`ridge/logreg`) и не ухудшать out-of-sample метрики.

## 6) Long/Short логика (рабочая схема)

- `pred > +thr => long`
- `pred < -thr => short`
- иначе `flat`

`thr` подбирать на validation по квантилям `|pred|` (например `0.7/0.8/0.9`), затем фиксировать и проверять только на test.

## 7) Когда подключать 5m

Подключать после стабилизации 1D модели.

Минимально полезные daily-агрегаты из `5m`:
- `daily_vwap`
- `intraday_range` (max-min внутри дня)
- `intraday_vol` (std 5m returns)
- `close_vs_vwap`

Это даёт сигнал режима дня без сложного intraday-бэктеста.

## 8) Практический цикл (каждый день)

1. `make run`
2. `make ml-build-raw`
3. `make ml-prepare`
4. `make ml-train`
5. `make ml-report`
6. `make ml-predict`

Если метрики деградируют:
- откатить последнее изменение фич/параметров,
- проверить данные и распределение таргета,
- повторить цикл.
