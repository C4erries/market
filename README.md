# T-Invest Market ETL (Sandbox Only)

Data collector for instruments, candles, trading schedules, trading statuses, and dividends.
This repository is **sandbox-only** and intentionally blocks production API usage.

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

## References

- Main repository (GitLab): https://opensource.tbank.ru/invest/invest-python
- Mirror/examples (GitHub): https://github.com/Tinkoff/invest-python
- Examples: https://tinkoff.github.io/invest-python/examples/
- Target sandbox/prod docs: https://russianinvestments.github.io/invest-python/
