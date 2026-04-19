# Basketball AI

A modular Python project to detect, track, and identify basketball players, inspired by the Roboflow Basketball AI notebook.

## Features
- **Player Detection**: Uses Roboflow Inference with YOLO-based models.
- **Tracking**: Implements ByteTrack for robust player tracking.
- **Modular Design**: Clean separation of configuration, model loading, and processing logic.
- **Fast Performance**: Optimized for RTX GPUs using `inference-gpu` and `uv`.

## Prerequisites
- NVIDIA GPU with drivers installed.
- [uv](https://github.com/astral-sh/uv) installed.

## Setup

1. Clone the repository and navigate to the project:
   ```bash
   git clone https://github.com/MathFrenchToast/t3d-robotflow-basket.git
   cd t3d-robotflow-basket
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your `ROBOFLOW_API_KEY`.

3. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

Run the tracking pipeline on a video:
```bash
uv run python -m src.main --source path/to/video.mp4 --target output.mp4
```

## Project Structure
- `src/config.py`: Thresholds and model IDs.
- `src/models.py`: Model loading logic.
- `src/tracking.py`: ByteTrack and ShotEventTracker initialization.
- `src/visualization.py`: Annotation and court drawing.
- `src/pipeline.py`: Main processing loop.
- `src/main.py`: CLI entry point.
