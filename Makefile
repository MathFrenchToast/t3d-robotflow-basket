.PHONY: help lint validate test check

VENV_BIN := $(shell if [ -d .venv/bin ]; then echo ".venv/bin/"; else echo ""; fi)
PYTHON   := $(VENV_BIN)python
RUFF     := $(VENV_BIN)ruff
YAMLLINT := $(VENV_BIN)yamllint
PYTEST   := $(VENV_BIN)pytest

help:
	@echo "Targets disponibles :"
	@echo "  make lint      - Linter Python (ruff) et YAML (yamllint sur k8s/ et .github/workflows/)"
	@echo "  make validate  - Valider le rendu Kustomize des manifestes Kubernetes (k8s/)"
	@echo "  make test      - Exécuter les tests unitaires et API FastAPI (pytest src/test_ci.py)"
	@echo "  make check     - Exécuter lint + validate + test (identique à la CI GitHub Actions)"

lint:
	@echo "==> Running Python linter (ruff)..."
	$(RUFF) check src/
	@echo "==> Running YAML linter (yamllint)..."
	$(YAMLLINT) -c .yamllint.yml k8s/ .github/workflows/

validate:
	@echo "==> Validating Kubernetes manifests with kubectl kustomize..."
	kubectl kustomize k8s/ > /dev/null
	@echo "OK: k8s/ kustomization builds cleanly."

test:
	@echo "==> Running FastAPI & Homography CI test suite..."
	$(PYTEST) -v src/test_ci.py

check: lint validate test
	@echo "==> All local/CI checks passed successfully!"
