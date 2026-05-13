SHELL := /bin/bash

.PHONY: up down logs build seed migrate test load fmt clean

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

build:
	docker compose build

migrate:
	docker compose exec backend alembic upgrade head

seed:
	docker compose exec backend python -m scripts.seed_db

test:
	docker compose exec backend pytest -q tests

load:
	cd locust && locust -f locustfile.py --headless -u 50 -r 5 -t 2m \
	  --host http://localhost:8000

fmt:
	ruff check --fix backend frontend prefect scripts
	ruff format backend frontend prefect scripts

clean:
	docker compose down -v
	rm -rf backend/__pycache__ frontend/__pycache__
