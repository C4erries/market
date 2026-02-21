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
TARGET_SYMBOL ?= MGNT
SYMBOLS ?= $(TARGET_SYMBOL),IMOEX,USDRUB
INTERVALS ?= 1d,5m
START ?= 2018-01-01
END ?= now
MODE ?= incremental
RAW_MAIN ?= ./data/candles_mgnt_1d.parquet
RAW_X5 ?= $(RAW_MAIN)
RAW_IMOEX ?= ./data/candles_imoex_1d.parquet
RAW_USDRUB ?= ./data/candles_usdrub_1d.parquet
RAW_CALENDAR ?=
RAW_DIVIDENDS ?=
ML_DATASET ?= ./data/model_ready/mgnt_next_day.parquet
ML_ARTIFACTS ?= ./artifacts/ml
ML_DATA_VIEW_OUT ?= $(ML_ARTIFACTS)/data_view
ML_MODEL_PLOTS_OUT ?= $(ML_ARTIFACTS)/plots/model_diagnostics
ML_TRAIN_RATIO ?= 0.70
ML_VAL_RATIO ?= 0.15
ML_TEST_RATIO ?= 0.15
ML_THRESHOLD_QUANTILES ?= 0.6,0.7,0.8,0.9
ML_COST_BPS ?= 0
ML_THRESHOLD_COST_MULTIPLIER ?= 1.0
ML_RANDOM_STATE ?= 42
ML_WF_ENABLE ?= 1
ML_WF_FOLDS ?= 6
ML_WF_EXPANDING ?= 1
ML_WF_STEP_SIZE ?=
ML_SELECTOR_USE_COST_RULE ?= 1
ML_SELECTOR_ALPHA_LOW ?= 0.1
ML_SELECTOR_ALPHA_HIGH ?= 0.9
SYMBOL ?= $(TARGET_SYMBOL)

.PHONY: help install install-dev install-ml env-check test compile check run run-full run-symbol run-full-symbol run-mgnt run-x5 run-imoex run-usdrub run-full-mgnt run-full-x5 run-full-imoex run-full-usdrub ml-build-raw ml-prepare ml-train ml-predict ml-report ml-diagnostics ml-diagnostics-deep ml-data-view ml-model-plots clean

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
	@echo "  make run-symbol   - run ETL incremental for one symbol (use SYMBOL=MGNT/IMOEX/USDRUB)"
	@echo "  make run-full-symbol - run ETL full for one symbol (use SYMBOL=...)"
	@echo "  make run-mgnt / run-imoex / run-usdrub - shortcuts for one-symbol incremental"
	@echo "  make run-full-mgnt / run-full-imoex / run-full-usdrub - shortcuts for one-symbol full"
	@echo "  make install-ml   - install ML dependencies"
	@echo "  make ml-build-raw - build flat 1D parquet files from partitioned candles"
	@echo "  make ml-prepare   - build model-ready dataset"
	@echo "  make ml-train     - train/evaluate models (ML_COST_BPS / ML_THRESHOLD_COST_MULTIPLIER / ML_WF_* / ML_SELECTOR_*)"
	@echo "  make ml-predict   - inference on latest row"
	@echo "  make ml-report    - print compact report for all trained models"
	@echo "  make ml-diagnostics - show raw/model-ready date windows and row counts"
	@echo "  make ml-diagnostics-deep - run deep data/model diagnostics and save JSON report"
	@echo "  make ml-data-view - plot raw/model-ready data overview"
	@echo "  make ml-model-plots - plot model diagnostics from training artifacts"
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

run-mgnt:
	$(MAKE) run-symbol SYMBOL=MGNT

run-x5:
	$(MAKE) run-symbol SYMBOL=X5

run-imoex:
	$(MAKE) run-symbol SYMBOL=IMOEX

run-usdrub:
	$(MAKE) run-symbol SYMBOL=USDRUB

run-full-mgnt:
	$(MAKE) run-full-symbol SYMBOL=MGNT

run-full-x5:
	$(MAKE) run-full-symbol SYMBOL=X5

run-full-imoex:
	$(MAKE) run-full-symbol SYMBOL=IMOEX

run-full-usdrub:
	$(MAKE) run-full-symbol SYMBOL=USDRUB

ml-build-raw:
	$(PYTHON) -c "from pathlib import Path; import pandas as pd; Path('$(RAW_MAIN)').parent.mkdir(parents=True, exist_ok=True); df=pd.read_parquet('$(OUT)/candles', filters=[('symbol','==','$(TARGET_SYMBOL)'),('interval','==','1d')]); assert len(df)>0, '$(TARGET_SYMBOL) 1d is empty. Run make run-symbol SYMBOL=$(TARGET_SYMBOL) or make run-full'; df.to_parquet('$(RAW_MAIN)', index=False); print('saved', len(df), 'rows to', '$(RAW_MAIN)')"
	$(PYTHON) -c "from pathlib import Path; import pandas as pd; Path('$(RAW_IMOEX)').parent.mkdir(parents=True, exist_ok=True); df=pd.read_parquet('$(OUT)/candles', filters=[('symbol','==','IMOEX'),('interval','==','1d')]); assert len(df)>0, 'IMOEX 1d is empty. Run make run-imoex or make run-full'; df.to_parquet('$(RAW_IMOEX)', index=False); print('saved', len(df), 'rows to', '$(RAW_IMOEX)')"
	$(PYTHON) -c "from pathlib import Path; import pandas as pd; Path('$(RAW_USDRUB)').parent.mkdir(parents=True, exist_ok=True); df=pd.read_parquet('$(OUT)/candles', filters=[('symbol','==','USDRUB'),('interval','==','1d')]); assert len(df)>0, 'USDRUB 1d is empty. Run make run-usdrub or make run-full'; df.to_parquet('$(RAW_USDRUB)', index=False); print('saved', len(df), 'rows to', '$(RAW_USDRUB)')"

ml-prepare:
	$(PYTHON) -m scripts.prepare_features --main "$(RAW_MAIN)" --imoex "$(RAW_IMOEX)" --usdrub "$(RAW_USDRUB)" --output "$(ML_DATASET)"

ml-train:
	$(PYTHON) -m scripts.train_and_evaluate --dataset "$(ML_DATASET)" --artifacts "$(ML_ARTIFACTS)" --train-ratio "$(ML_TRAIN_RATIO)" --val-ratio "$(ML_VAL_RATIO)" --test-ratio "$(ML_TEST_RATIO)" --threshold-quantiles "$(ML_THRESHOLD_QUANTILES)" --random-state "$(ML_RANDOM_STATE)" --cost-bps "$(ML_COST_BPS)" --threshold-cost-multiplier "$(ML_THRESHOLD_COST_MULTIPLIER)" --wf-enable "$(ML_WF_ENABLE)" --wf-folds "$(ML_WF_FOLDS)" --wf-expanding "$(ML_WF_EXPANDING)" --selector-use-cost-rule "$(ML_SELECTOR_USE_COST_RULE)" --selector-alpha-low "$(ML_SELECTOR_ALPHA_LOW)" --selector-alpha-high "$(ML_SELECTOR_ALPHA_HIGH)" $(if $(ML_WF_STEP_SIZE),--wf-step-size "$(ML_WF_STEP_SIZE)",)

ml-predict:
	$(PYTHON) -m scripts.predict --dataset "$(ML_DATASET)" --artifacts "$(ML_ARTIFACTS)"

ml-report:
	$(PYTHON) -m scripts.ml_report --artifacts "$(ML_ARTIFACTS)"

ml-diagnostics:
	$(PYTHON) -m scripts.ml_diagnostics --raw-main "$(RAW_MAIN)" --raw-imoex "$(RAW_IMOEX)" --raw-usdrub "$(RAW_USDRUB)" --dataset "$(ML_DATASET)"

ml-diagnostics-deep:
	$(PYTHON) -m scripts.ml_diagnostics_deep --raw-main "$(RAW_MAIN)" --raw-imoex "$(RAW_IMOEX)" --raw-usdrub "$(RAW_USDRUB)" --dataset "$(ML_DATASET)" --artifacts "$(ML_ARTIFACTS)" --output "$(ML_ARTIFACTS)/reports/diagnostics_deep.json"

ml-data-view:
	$(PYTHON) -m scripts.data_view --main "$(RAW_MAIN)" --imoex "$(RAW_IMOEX)" --usdrub "$(RAW_USDRUB)" --dataset "$(ML_DATASET)" --out "$(ML_DATA_VIEW_OUT)"

ml-model-plots:
	$(PYTHON) -m scripts.model_plots --artifacts "$(ML_ARTIFACTS)" --out "$(ML_MODEL_PLOTS_OUT)"

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]"
