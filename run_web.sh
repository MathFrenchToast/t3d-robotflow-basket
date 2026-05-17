#!/bin/bash
# Run the Basketball AI web server

echo "Starting Basketball AI Analyzer Web Server..."
echo "Open http://localhost:8001 in your browser"
echo ""

# Using port 8001 to avoid conflict if vlm-test is also running on 8000
uv run python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8001 --reload
