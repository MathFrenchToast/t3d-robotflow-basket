.PHONY: help install-ci lint validate test check

VENV_BIN := $(shell if [ -d .venv/bin ]; then echo ".venv/bin/"; else echo ""; fi)
PYTHON   := $(VENV_BIN)python
PIP      := $(VENV_BIN)pip
RUFF     := $(VENV_BIN)ruff
YAMLLINT := $(VENV_BIN)yamllint
PYTEST   := $(VENV_BIN)pytest

help:
	@echo "Targets disponibles :"
	@echo "  make install-ci - Installer les dépendances légères CI + linters/tests"
	@echo "  make lint       - Linter Python (ruff) et YAML (yamllint sur k8s/ et .github/workflows/)"
	@echo "  make validate   - Valider le rendu Kustomize des manifestes Kubernetes (k8s/)"
	@echo "  make test       - Exécuter les tests unitaires et API FastAPI (pytest src/test_ci.py)"
	@echo "  make check      - Exécuter lint + validate + test (identique à la CI GitHub Actions)"

install-ci:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install \
		"ruff>=0.9.0" \
		"yamllint>=1.35.0" \
		"pytest>=8.0.0" \
		"httpx>=0.27.0" \
		"fastapi>=0.115.14" \
		"uvicorn[standard]>=0.30.0" \
		"python-multipart>=0.0.18" \
		"python-dotenv>=1.0.1" \
		"requests>=2.32.0" \
		"numpy>=1.26.0" \
		"opencv-python-headless>=4.10.0.84" \
		"scikit-learn>=1.5.0" \
		"supervision>=0.25.0"
	$(PYTHON) -m pip install --no-deps "git+https://github.com/roboflow/sports.git@feat/basketball"

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
