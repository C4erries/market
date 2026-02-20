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
SYMBOL ?= X5

.PHONY: help install install-dev install-ml env-check test compile check run run-full run-symbol run-full-symbol run-x5 run-imoex run-usdrub run-full-x5 run-full-imoex run-full-usdrub ml-build-raw ml-prepare ml-train ml-predict ml-report clean

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
	@echo "  make run-symbol   - run ETL incremental for one symbol (use SYMBOL=X5/IMOEX/USDRUB)"
	@echo "  make run-full-symbol - run ETL full for one symbol (use SYMBOL=...)"
	@echo "  make run-x5 / run-imoex / run-usdrub - shortcuts for one-symbol incremental"
	@echo "  make run-full-x5 / run-full-imoex / run-full-usdrub - shortcuts for one-symbol full"
	@echo "  make install-ml   - install ML dependencies"
	@echo "  make ml-build-raw - build flat 1D parquet files from partitioned candles"
	@echo "  make ml-prepare   - build model-ready dataset"
	@echo "  make ml-train     - train/evaluate models"
	@echo "  make ml-predict   - inference on latest row"
	@echo "  make ml-report    - print compact report for all trained models"
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

run-symbol:
	$(PYTHON) -m etl.download_data --symbols "$(SYMBOL)" --intervals "$(INTERVALS)" --start "$(START)" --end "$(END)" --out "$(OUT)" --mode incremental

run-full-symbol:
	$(PYTHON) -m etl.download_data --symbols "$(SYMBOL)" --intervals "$(INTERVALS)" --start max --end "$(END)" --out "$(OUT)" --mode full

run-x5:
	$(MAKE) run-symbol SYMBOL=X5

run-imoex:
	$(MAKE) run-symbol SYMBOL=IMOEX

run-usdrub:
	$(MAKE) run-symbol SYMBOL=USDRUB

run-full-x5:
	$(MAKE) run-full-symbol SYMBOL=X5

run-full-imoex:
	$(MAKE) run-full-symbol SYMBOL=IMOEX

run-full-usdrub:
	$(MAKE) run-full-symbol SYMBOL=USDRUB

ml-build-raw:
	$(PYTHON) -c "from pathlib import Path; import pandas as pd; Path('$(RAW_X5)').parent.mkdir(parents=True, exist_ok=True); df=pd.read_parquet('$(OUT)/candles', filters=[('symbol','==','X5'),('interval','==','1d')]); df.to_parquet('$(RAW_X5)', index=False); print('saved', len(df), 'rows to', '$(RAW_X5)')"
	$(PYTHON) -c "from pathlib import Path; import pandas as pd; Path('$(RAW_IMOEX)').parent.mkdir(parents=True, exist_ok=True); df=pd.read_parquet('$(OUT)/candles', filters=[('symbol','==','IMOEX'),('interval','==','1d')]); df.to_parquet('$(RAW_IMOEX)', index=False); print('saved', len(df), 'rows to', '$(RAW_IMOEX)')"
	$(PYTHON) -c "from pathlib import Path; import pandas as pd; Path('$(RAW_USDRUB)').parent.mkdir(parents=True, exist_ok=True); df=pd.read_parquet('$(OUT)/candles', filters=[('symbol','==','USDRUB'),('interval','==','1d')]); df.to_parquet('$(RAW_USDRUB)', index=False); print('saved', len(df), 'rows to', '$(RAW_USDRUB)')"

ml-prepare:
	$(PYTHON) -m scripts.prepare_features --x5 "$(RAW_X5)" --imoex "$(RAW_IMOEX)" --usdrub "$(RAW_USDRUB)" --output "$(ML_DATASET)"

ml-train:
	$(PYTHON) -m scripts.train_and_evaluate --dataset "$(ML_DATASET)" --artifacts "$(ML_ARTIFACTS)"

ml-predict:
	$(PYTHON) -m scripts.predict --dataset "$(ML_DATASET)" --artifacts "$(ML_ARTIFACTS)"

ml-report:
	$(PYTHON) -m scripts.ml_report --artifacts "$(ML_ARTIFACTS)"

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]"
