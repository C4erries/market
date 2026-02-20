# T-Invest Market ETL (Sandbox Only)

Data collector for instruments, candles, trading schedules, trading statuses, and dividends.
This repository is **sandbox-only** and intentionally blocks production API usage.
ML pipeline is fully offline and reads only local Parquet/CSV files (no API calls).

## Install

```bash
pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple
pip install pandas pyarrow
```

## Environment

Use only sandbox variables:

```bash
TINVEST_ENV=sandbox
TINVEST_SANDBOX_TOKEN=your_sandbox_token
```

Windows PowerShell:

```powershell
$env:TINVEST_ENV="sandbox"
$env:TINVEST_SANDBOX_TOKEN="your_sandbox_token"
```

Production tokens are not supported in this repository.

## Run

Incremental mode:

```bash
python download_data.py --symbols X5,IMOEX,USDRUB --intervals 1d,5m --start 2018-01-01 --end now --out ./data --mode incremental
```

Full reload:

```bash
python download_data.py --symbols X5,IMOEX,USDRUB --intervals 1d,5m --start max --end now --out ./data --mode full
```

## ML Pipeline (Offline)

### Install ML dependencies

```bash
make install-ml
```

### 1) Build model-ready dataset

```bash
python prepare_features.py \
  --x5 ./data/x5_1d.parquet \
  --imoex ./data/imoex_1d.parquet \
  --usdrub ./data/usdrub_1d.parquet \
  --calendar ./data/calendars/trading_schedules.parquet \
  --dividends ./data/corporate_actions/dividends_symbol=X5.parquet \
  --output ./data/model_ready/x5_next_day.parquet \
  --target-type log
```

What it does:
- merges local X5 + context by date;
- builds tabular features (returns lags, volatility, momentum, range, volume, calendar, context);
- creates target `y` (next-day return) and `y_dir`;
- drops rows with NaN from rolling windows and the last row without `t+1`.

### 2) Train and evaluate

```bash
python train_and_evaluate.py \
  --dataset ./data/model_ready/x5_next_day.parquet \
  --artifacts ./artifacts/ml \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --threshold-quantiles 0.6,0.7,0.8,0.9 \
  --cost-bps 0
```

Models:
- `Ridge` regression baseline;
- `LogisticRegression` direction baseline;
- `LightGBMRegressor` main model;
- LightGBM quantile models (`alpha=0.8` and `alpha=0.2`) for strong-signal selection.

Saved artifacts:
- models: `ridge.joblib`, `logreg.joblib`, `main_lgbm.joblib`, quantile models;
- configs/metrics: `feature_columns.json`, `strategy_config.json`, `metrics.json`, `run_report.json`;
- reports: threshold search table, test predictions, equity curves and plots.

### 3) Predict latest signal

```bash
python predict.py \
  --dataset ./data/model_ready/x5_next_day.parquet \
  --artifacts ./artifacts/ml
```

Output: predicted return, threshold, and signal (`long` / `short` / `flat`) for latest available date.

### Makefile shortcuts

```bash
make ml-prepare
make ml-train
make ml-predict
```

## Output

```text
data/
  instruments.parquet
  candles/
    symbol=X5/interval=5m/year=2025/...
    symbol=X5/interval=1d/year=2025/...
    symbol=IMOEX/interval=1d/year=2025/...
    symbol=USDRUB/interval=1d/year=2025/...
  calendars/
    trading_schedules.parquet
    trading_statuses.parquet
  corporate_actions/
    dividends_symbol=X5.parquet
```

## Safety Guards

- `TINVEST_ENV` must be `sandbox` (default is `sandbox`).
- `TINVEST_SANDBOX_TOKEN` is mandatory.
- Legacy env vars like `INVEST_TOKEN`, `TINVEST_TOKEN`, `TOKEN` are rejected.
- Client target is hardcoded and asserted as `INVEST_GRPC_API_SANDBOX`.
- Startup AST check blocks forbidden trading calls/imports.
- Production API mode is blocked by fail-fast guards.

## References

- Main repository (GitLab): https://opensource.tbank.ru/invest/invest-python
- Mirror/examples (GitHub): https://github.com/Tinkoff/invest-python
- Examples: https://tinkoff.github.io/invest-python/examples/
- Target sandbox/prod docs: https://russianinvestments.github.io/invest-python/
