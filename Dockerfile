FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# System dependencies for OpenCV headless and git (for roboflow/sports package)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lists/* /var/lib/apt/lists/*

# Install lightweight runtime dependencies for the FastAPI + VLM + 2D Homography pipeline
RUN pip install --no-cache-dir \
    "fastapi>=0.115.14" \
    "uvicorn[standard]>=0.30.0" \
    "python-multipart>=0.0.18" \
    "python-dotenv>=1.0.1" \
    "requests>=2.32.0" \
    "numpy>=1.26.0" \
    "opencv-python-headless>=4.10.0.84" \
    "scikit-learn>=1.5.0" \
    "supervision>=0.25.0" \
    && pip install --no-cache-dir --no-deps "git+https://github.com/roboflow/sports.git@feat/basketball"

COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
