SHELL := /bin/bash
COMPOSE := docker compose

.PHONY: help setup models build up down restart logs ps health shell-backend rescan clean

help:
	@echo "make setup     - host prep (sudo): nvidia runtime, power mode, swap"
	@echo "make models    - download LLM / whisper / embedding models into ./models"
	@echo "make build     - build the CUDA images (slow: ~45-70 min on an Orin Nano)"
	@echo "make up        - start everything"
	@echo "make health    - service health + collection sizes"
	@echo "make logs      - follow all logs"
	@echo "make rescan    - re-ingest ./data/docs into Qdrant"

setup:
	sudo bash scripts/jetson_setup.sh
	bash scripts/set_lan_ip.sh

models:
	bash scripts/download_models.sh

build:
	$(COMPOSE) build

up:
	@test -f .env || cp .env.example .env
	$(COMPOSE) up -d
	@echo
	@grep -E '^SITE_ADDRESS=' .env

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart

logs:
	$(COMPOSE) logs -f --tail=100

ps:
	$(COMPOSE) ps

health:
	@curl -fsS http://localhost:$${HTTP_PORT:-8080}/api/health | python3 -m json.tool

rescan:
	@curl -fsS -X POST http://localhost:$${HTTP_PORT:-8080}/api/ingest/rescan | python3 -m json.tool

shell-backend:
	$(COMPOSE) exec backend bash

clean:
	$(COMPOSE) down -v
