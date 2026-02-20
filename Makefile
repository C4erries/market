PYTHON ?= python
OUT ?= ./data
SYMBOLS ?= X5,IMOEX,USDRUB
INTERVALS ?= 1d,5m
START ?= 2018-01-01
END ?= now
MODE ?= incremental

.PHONY: help install install-dev env-check test compile check run run-full clean

help:
	@echo "Available targets:"
	@echo "  make install      - install runtime dependencies"
	@echo "  make install-dev  - install runtime + test dependencies"
	@echo "  make env-check    - print sandbox env vars"
	@echo "  make test         - run unit tests"
	@echo "  make compile      - run python compile checks"
	@echo "  make check        - run compile + tests"
	@echo "  make run          - run ETL in incremental mode"
	@echo "  make run-full     - run ETL in full mode"
	@echo "  make clean        - remove Python cache files"

install:
	$(PYTHON) -m pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple
	$(PYTHON) -m pip install pandas pyarrow

install-dev: install
	$(PYTHON) -m pip install pytest

env-check:
	$(PYTHON) -c "import os; print('TINVEST_ENV=', os.getenv('TINVEST_ENV', '<unset>')); print('TINVEST_SANDBOX_TOKEN=', 'set' if os.getenv('TINVEST_SANDBOX_TOKEN') else '<unset>')"

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

compile:
	$(PYTHON) -m compileall main.py download_data.py tinvest_client.py storage.py safety_guard.py tests

check: compile test

run:
	$(PYTHON) download_data.py --symbols "$(SYMBOLS)" --intervals "$(INTERVALS)" --start "$(START)" --end "$(END)" --out "$(OUT)" --mode "$(MODE)"

run-full:
	$(PYTHON) download_data.py --symbols "$(SYMBOLS)" --intervals "$(INTERVALS)" --start max --end "$(END)" --out "$(OUT)" --mode full

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]"
