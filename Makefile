# Makefile

.PHONY: help env install lint format test build run

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

env:            ## Create & activate conda env
	conda env create -f environment.yml
	conda activate stroke-env

install:        ## Install pip deps
	pip install -r requirements.txt -r dev-requirements.txt

lint:           ## Run linters
	black --check src tests
	flake8 src tests

format:         ## Auto-format code
	black src tests
	isort src tests

test:           ## Run all tests
	pytest --maxfail=1 --disable-warnings -q

build:          ## Build Docker image
	docker build -t stroke-api .

run:            ## Run Docker container locally
	docker rm -f stroke-api || true && \
	docker run --network host --name stroke-api -e MLFLOW_TRACKING_URI="http://127.0.0.1:5001" stroke-api
