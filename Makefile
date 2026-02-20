ENV_FILE ?= .env

ifneq (,$(wildcard $(ENV_FILE)))
include $(ENV_FILE)
.EXPORT_ALL_VARIABLES:
endif

TINVEST_ENV ?= sandbox
ifdef TOKEN_SANDBOX
TINVEST_SANDBOX_TOKEN ?= $(TOKEN_SANDBOX)
endif

PYTHON ?= python
OUT ?= ./data
SYMBOLS ?= X5,IMOEX,USDRUB
INTERVALS ?= 1d,5m
START ?= 2018-01-01
END ?= now
MODE ?= incremental
RAW_X5 ?= ./data/candles_x5_1d.parquet
RAW_IMOEX ?= ./data/candles_imoex_1d.parquet
RAW_USDRUB ?= ./data/candles_usdrub_1d.parquet
RAW_CALENDAR ?=
RAW_DIVIDENDS ?=
ML_DATASET ?= ./data/model_ready/x5_next_day.parquet
ML_ARTIFACTS ?= ./artifacts/ml

.PHONY: help install install-dev install-ml env-check test compile check run run-full ml-prepare ml-train ml-predict clean

help:
	@echo "Available targets:"
	@echo "  (variables from .env are loaded automatically if .env exists)"
	@echo "  make install      - install runtime dependencies"
	@echo "  make install-dev  - install runtime + test dependencies"
	@echo "  make env-check    - print sandbox env vars"
	@echo "  make test         - run unit tests"
	@echo "  make compile      - run python compile checks"
	@echo "  make check        - run compile + tests"
	@echo "  make run          - run ETL in incremental mode"
	@echo "  make run-full     - run ETL in full mode"
	@echo "  make install-ml   - install ML dependencies"
	@echo "  make ml-prepare   - build model-ready dataset"
	@echo "  make ml-train     - train/evaluate models"
	@echo "  make ml-predict   - inference on latest row"
	@echo "  make clean        - remove Python cache files"

install:
	$(PYTHON) -m pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple
	$(PYTHON) -m pip install pandas pyarrow

install-dev: install
	$(PYTHON) -m pip install pytest

install-ml:
	$(PYTHON) -m pip install numpy pandas scikit-learn lightgbm matplotlib joblib pyarrow

env-check:
	$(PYTHON) -c "import os; from pathlib import Path; print('ENV_FILE=', '.env', 'exists' if Path('.env').exists() else 'missing'); print('TINVEST_ENV=', os.getenv('TINVEST_ENV', '<unset>')); print('TINVEST_SANDBOX_TOKEN=', 'set' if os.getenv('TINVEST_SANDBOX_TOKEN') else '<unset>')"

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

compile:
	$(PYTHON) -m compileall main.py etl ml_pipeline scripts tests

check: compile test

run:
	$(PYTHON) -m etl.download_data --symbols "$(SYMBOLS)" --intervals "$(INTERVALS)" --start "$(START)" --end "$(END)" --out "$(OUT)" --mode "$(MODE)"

run-full:
	$(PYTHON) -m etl.download_data --symbols "$(SYMBOLS)" --intervals "$(INTERVALS)" --start max --end "$(END)" --out "$(OUT)" --mode full

ml-prepare:
	$(PYTHON) -m scripts.prepare_features --x5 "$(RAW_X5)" --imoex "$(RAW_IMOEX)" --usdrub "$(RAW_USDRUB)" --output "$(ML_DATASET)"

ml-train:
	$(PYTHON) -m scripts.train_and_evaluate --dataset "$(ML_DATASET)" --artifacts "$(ML_ARTIFACTS)"

ml-predict:
	$(PYTHON) -m scripts.predict --dataset "$(ML_DATASET)" --artifacts "$(ML_ARTIFACTS)"

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]"
