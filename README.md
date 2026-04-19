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

## Troubleshooting

### PyCUDA Build Errors
If `uv sync` fails with `pycuda` build errors:
1. **Install Dependencies**: `sudo apt install build-essential python3-dev nvidia-cuda-toolkit`
2. **Find CUDA Path**: Run `which nvcc`. 
   - If `which nvcc` is empty, you must install it: `sudo apt install nvidia-cuda-toolkit`.
   - Note: If `find` only shows headers in `/usr/include/linux/cuda.h`, these are **kernel headers** and are insufficient. You need the toolkit headers.
3. **Export and Sync**:
   ```bash
   export CUDA_INC_DIR=/usr/include  # or your specific path
   uv sync
   ```

## Project Structure
- `src/config.py`: Thresholds and model IDs.
- `src/models.py`: Model loading logic.
- `src/tracking.py`: ByteTrack and ShotEventTracker initialization.
- `src/visualization.py`: Annotation and court drawing.
- `src/pipeline.py`: Main processing loop.
- `src/main.py`: CLI entry point.
