# T-Invest Market ETL (Sandbox Only)

Data collector for instruments, candles, trading schedules, trading statuses, and dividends.
This repository is **sandbox-only** and intentionally blocks production API usage.
ML pipeline is fully offline and reads only local Parquet/CSV files (no API calls).

## Project Structure

```text
etl/              # ETL core modules (sandbox-only client, storage, guards, downloader)
ml_pipeline/      # offline ML data/model modules
scripts/          # CLI implementations for ML tasks
main.py           # optional root entrypoint for ETL
```

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
python -m etl.download_data --symbols MGNT,IMOEX,USDRUB --intervals 1d,5m --start 2018-01-01 --end now --out ./data --mode incremental
```

Full reload:

```bash
python -m etl.download_data --symbols MGNT,IMOEX,USDRUB --intervals 1d,5m --start max --end now --out ./data --mode full
```

## ML Pipeline (Offline)

### Install ML dependencies

```bash
make install-ml
```

### 1) Build model-ready dataset

```bash
python -m scripts.prepare_features \
  --main ./data/mgnt_1d.parquet \
  --imoex ./data/imoex_1d.parquet \
  --usdrub ./data/usdrub_1d.parquet \
  --calendar ./data/calendars/trading_schedules.parquet \
  --dividends ./data/corporate_actions/dividends_symbol=MGNT.parquet \
  --output ./data/model_ready/mgnt_next_day.parquet \
  --target-type log
```

What it does:
- merges local main symbol + context by date;
- builds tabular features (returns lags, volatility, momentum, range, volume, calendar, context);
- creates target `y` (next-day return) and `y_dir`;
- drops rows with NaN from rolling windows and the last row without `t+1`.

### 2) Train and evaluate

```bash
python -m scripts.train_and_evaluate \
  --dataset ./data/model_ready/mgnt_next_day.parquet \
  --artifacts ./artifacts/ml \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --threshold-quantiles 0.6,0.7,0.8,0.9 \
  --cost-bps 10 \
  --threshold-cost-multiplier 1.0 \
  --wf-enable 1 \
  --wf-folds 6 \
  --wf-expanding 1 \
  --selector-use-cost-rule 1 \
  --selector-alpha-low 0.1 \
  --selector-alpha-high 0.9 \
  --selector-risk-multiple 1.0
```

Models:
- `Ridge` regression baseline;
- `LogisticRegression` direction baseline;
- `LightGBMRegressor` main model (candidate selection by IC + anti-overfit penalties, not RMSE-driven);
- LightGBM quantile models (`alpha_low`/`alpha_high`, default `0.1/0.9`) for strong-signal selection.

Cost-aware trading setup:
- `cost_bps` is charged on turnover `abs(signal_t - signal_{t-1})` (entry/exit/flip aware);
- LGBM candidate selection uses the same cost by default (`selection_cost_bps = cost_bps`), optional override via CLI;
- `thr_min = cost_bps / 10000 * threshold_cost_multiplier`;
- threshold search is hard-limited by `threshold >= thr_min`;
- selector rule (default): direction comes from main model (`pred_main` vs `pred_main_threshold`), quantiles act as risk filter:
  long when `main > threshold` and `q_low > -risk_multiple * thr_min`, short when `main < -threshold` and `q_high < risk_multiple * thr_min`.

Saved artifacts:
- models: `ridge.joblib`, `logreg.joblib`, `main_lgbm.joblib`, quantile models;
- configs/metrics: `feature_columns.json`, `strategy_config.json`, `metrics.json`, `run_report.json`;
- reports: `threshold_search.csv`, `threshold_search_logreg.csv`, `lgbm_model_selection.csv`, `model_quality.csv`,
  `walk_forward_folds.csv`, `walk_forward_summary.csv`, test predictions, equity curves and plots.

### 3) Predict latest signal

```bash
python -m scripts.predict \
  --dataset ./data/model_ready/mgnt_next_day.parquet \
  --artifacts ./artifacts/ml
```

Output: predicted return, threshold, and signal (`long` / `short` / `flat`) for latest available date.

### Makefile shortcuts

```bash
make ml-prepare
make ml-train
make ml-predict
make ml-report
make ml-diagnostics
make ml-diagnostics-deep
make ml-data-view
make ml-model-plots
```

### Data and model plots

```bash
make ml-data-view
```

Saves data overview to `./artifacts/ml/data_view`:
- `main_price_volume.png`
- `context_normalized.png`
- `target_distribution.png`
- `missing_ratio.png`
- `top_feature_correlations.png`
- `data_summary.json`

```bash
make ml-model-plots
```

Saves model diagnostics to `./artifacts/ml/plots/model_diagnostics`:
- predictions vs actual (`pred_vs_actual.png`)
- prediction scatter (`pred_scatter.png`)
- residual histogram (`residuals_hist.png`)
- equity + drawdown (`equity_drawdown_main.png`)
- signal distribution (`signal_distribution_main.png`)
- selector signal distribution (`selector_signal_distribution.png`, if available)
- threshold search (`threshold_search.png`, if available)
- walk-forward Sharpe (`walk_forward_sharpe.png`, if available)
- walk-forward IC (`walk_forward_ic.png`, if available)
- feature importance (`feature_importance_main_lgbm.png`, if available)

## Output

```text
data/
  instruments.parquet
  candles/
    symbol=MGNT/interval=5m/year=2025/...
    symbol=MGNT/interval=1d/year=2025/...
    symbol=IMOEX/interval=1d/year=2025/...
    symbol=USDRUB/interval=1d/year=2025/...
  calendars/
    trading_schedules.parquet
    trading_statuses.parquet
  corporate_actions/
    dividends_symbol=MGNT.parquet
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
