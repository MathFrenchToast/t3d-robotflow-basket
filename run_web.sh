#!/bin/bash
# Run the T3D Multi-Sport AI Analyzer API & Web UI locally

PORT="${PORT:-8001}"
echo "Starting T3D Multi-Sport AI Analyzer API & Web UI..."
echo "Web UI:   http://localhost:${PORT}"
echo "API Docs: http://localhost:${PORT}/docs"
echo "Healthz:  http://localhost:${PORT}/healthz"
echo ""

PYTHONPATH=. uv run python -m uvicorn src.web.app:app --host 0.0.0.0 --port "${PORT}" --reload
