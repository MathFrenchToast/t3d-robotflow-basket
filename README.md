# Basketball AI

A modular Python project to detect, track, and identify basketball players, inspired by the Roboflow Basketball AI notebook.
from this collab notebook from roboflow: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb#scrollTo=Ul99bONuZp-I

## Features
- **Player Detection**: Uses Roboflow Inference with YOLO-based models.
- **Tracking**: Implements ByteTrack for robust player tracking.
- **Modular Design**: Clean separation of configuration, model loading, and processing logic.
- **Fast Performance**: Optimized for RTX GPUs using `inference-gpu` and `uv`.

## Prerequisites (see below)
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
uv run python -m src.main --source test/basketball-s.mp4 --target output.mp4
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

### GPU Not Found (OVH / Turing Instances)
If `nvidia-smi` returns "No devices were found" or `dmesg` shows `NVRM: RmInitAdapter: Cannot initialize GSP firmware RM`, the GSP firmware is failing to initialize (common on Turing GPUs like RTX 5000 in virtualized environments).

1. **Disable GSP Firmware**:
   ```bash
   echo "options nvidia NVreg_EnableGpuFirmware=0" | sudo tee /etc/modprobe.d/nvidia-gsp.conf
   sudo update-initramfs -u
   sudo reboot
   ```

2. **Fix Permissions**:
   Ensure your user has access to the GPU devices:
   ```bash
   sudo usermod -aG video,render $USER
   # Log out and log back in
   ```

## Project Structure
- `src/config.py`: Thresholds and model IDs.
- `src/models.py`: Model loading logic.
- `src/tracking.py`: ByteTrack and ShotEventTracker initialization.
- `src/visualization.py`: Annotation and court drawing.
- `src/pipeline.py`: Main processing loop.
- `src/main.py`: CLI entry point.

## Comparison with Original Notebook

While this project is inspired by the Roboflow Basketball AI notebook, several architectural changes were made to convert it into a production-ready modular application:

1. **Tracking Mechanism**:
   - **Notebook**: Uses **SAM2 (Segment Anything Model 2)** which provides high-accuracy segmentation masks but is computationally intensive.
   - **This Project**: Uses **ByteTrack** with bounding boxes. This significantly improves processing speed (FPS) while maintaining robust tracking, suitable for real-time or batch processing on standard RTX GPUs.

2. **Code Structure**:
   - **Notebook**: A linear execution flow designed for experimentation.
   - **This Project**: Modularized into separate concerns (Models, Tracking, Visualization, Pipeline). Logic like jersey number matching and team validation is encapsulated into stateful classes to ensure stability across video frames.

3. **API & Library Drift**:
   - The `sports` library is under active development. This project implements fixes for API changes (e.g., `draw_points_on_court` parameter updates) that occurred after the notebook was published.
   - We use `inference-gpu` for optimized model serving, ensuring better utilization of NVIDIA hardware compared to standard CPU-based inference.

4. **Identification Logic**:
   - Adapted the mask-based IoU matching from the notebook to work with bounding box IoU, allowing jersey number recognition to function effectively without the overhead of SAM2 masks.

## Installing prerequisites

installing uv, and python 3.12 with it
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install 3.12
```

test if nvidia drivers are installed, on ubuntu 24 we target cuda 12.5
```
which nvcc
# if installed ok
# else

# 1. Install Build Tools and CUDA 12.5
sudo apt-get install -y build-essential cuda-toolkit-12-5

# 2. Set Up the Environment, update your path to point to the 12.5 version:

echo 'export PATH=/usr/local/cuda-12.5/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# 3. Verify gcc and cuda compiler
g++ --version
nvcc --version

# 4. Retry Sync
uv sync
```