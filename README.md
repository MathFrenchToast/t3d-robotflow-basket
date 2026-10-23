# T3D Multi-Sport AI Analyzer

A modular Python & FastAPI project to detect, classify, and project players (`T0`, `T1`), goalkeepers (`T0-GK`, `T1-GK`), referees (`REF`), balls (`BALL`), and field landmarks onto a 2D top-down radar across **Basketball**, **Football (Soccer)**, **Handball**, and **US Football (NFL)**.

---

## Features

- **Multi-Sport VLM Detection (`basket`, `foot`, `handball`, `usfootball`)**:
  - Unified OpenRouter (`openai/gpt-6-sol`, `google/gemini-2.5-pro`, `google/gemini-2.5-flash`) and Google Gemini Direct (`gemini-3.6-flash`) backends.
  - Native patch-grid alignment (`1536×864` for OpenAI vision models, normalized `0..1000` for Gemini) with adaptive resolution-aware post-processing.
- **Team & Goalkeeper Identification**:
  - Unsupervised jersey chromaticity clustering fused with Bayesian VLM team priors (`T0` vs `T1`).
  - Dedicated goalkeeper detection and defensive goal-area team assignment (`T0-GK` / `T1-GK`) in both JSON output and 2D radar view.
- **2D Court / Pitch Homography & Radar**:
  - Hybrid RANSAC homography combining ONNX keypoint models (when available), VLM court calibration landmarks, and goal/basket bbox anchors.
  - Automatic side-by-side (`[Annotated Image | 2D Radar]`) and overlay visualizations.
- **FastAPI Server & Interactive Web UI**:
  - Single-call `POST /api/analyze` returning **either structured JSON or the annotated JPEG image** (`side_by_side` or `annotated`).
  - Zero-cost instant re-rendering endpoint `POST /api/visualize` (`< 50ms`, `$0.00` LLM cost).
  - Kubernetes probes (`GET /healthz`, `GET /readyz`) and ready-to-deploy manifests in [`k8s/`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/k8s).

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MathFrenchToast/t3d-robotflow-basket.git
   cd t3d-robotflow-basket
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and configure your API keys:
   - `OPENROUTER_API_KEY`: Required for OpenRouter models (default: `openai/gpt-6-sol`).
   - `GOOGLE_API_KEY`: Required when using `--backend gemini-direct`.
   - `ROBOFLOW_API_KEY`: Optional/Required for Roboflow keypoint models or workflows.

3. **Install dependencies with [uv](https://github.com/astral-sh/uv):**
   ```bash
   uv sync
   ```

4. **(Optional) Link local test dataset:**
   If you have the shared test dataset directory (`../vlm-test/test` or `../t3d-test-data`), create a local symlink (ignored by Git):
   ```bash
   ln -sfn ../vlm-test/test tests
   ```

---

## Testing the FastAPI Server & Web UI Locally

### 1. Start the Local Server
```bash
./run_web.sh
```
By default, the server starts on port `8001` (override with `PORT=8000 ./run_web.sh`):
- **Interactive Web UI**: `http://localhost:8001`
- **Swagger / OpenAPI UI**: `http://localhost:8001/docs` (and ReDoc at `http://localhost:8001/redoc`)
- **Product Documentation Pages**:
  - **VLM Priors & RANSAC 2D Homography**: `http://localhost:8001/docs/vlm-priors`
  - **JSON Output Schema & HTTP Headers**: `http://localhost:8001/docs/output-schema`
- **Health / Readiness Probes**: `http://localhost:8001/healthz` and `http://localhost:8001/readyz`

### 2. Test the API with `curl`

#### A. Health & Readiness checks
```bash
curl -s http://localhost:8001/healthz | jq .
curl -s http://localhost:8001/readyz | jq .
```

#### B. Analyze an image and receive **JSON** (`output_format=json`, default)
```bash
curl -X POST http://localhost:8001/api/analyze \
  -F "image=@tests/usfootball/usfootball-01.png" \
  -F "sport=usfootball" \
  -F "backend=openrouter" \
  -F "model=openai/gpt-6-sol" \
  -F "vlm_priors=true" \
  -F "output_format=json" \
  -o result.json
```

#### C. Analyze an image and receive **the Image directly** (`output_format=image` or `annotated`)
- **`output_format=image`** (or `side_by_side` / `return_image=true`): returns the composite JPEG `[Annotated Image | 2D Radar]`.
- **`output_format=annotated`**: returns the annotated image with the radar overlaid in the top-left corner.
```bash
# Side-by-side image (Annotated Frame + 2D Radar)
curl -D - -X POST http://localhost:8001/api/analyze \
  -F "image=@tests/handball/handballNormal.png" \
  -F "sport=handball" \
  -F "output_format=image" \
  -o handball_side_by_side.jpg

# Annotated overlay image only
curl -X POST http://localhost:8001/api/analyze \
  -F "image=@tests/foot/foot1.png" \
  -F "sport=foot" \
  -F "output_format=annotated" \
  -o foot_annotated.jpg
```
> Note: Image responses also include metadata HTTP headers: `X-Detected-Sport`, `X-Homography-Source`, `X-Players-Count`, `X-Goalkeepers-Count`, `X-Processing-Time-Seconds`, `X-VLM-Inference-Seconds`, and `X-Estimated-Cost-USD` (see full specification at `/docs/output-schema`).

#### D. Re-render visualization from existing JSON (`POST /api/visualize`, `$0.00` LLM cost)
```bash
curl -X POST http://localhost:8001/api/visualize \
  -F "image=@tests/handball/handballNormal.png" \
  -F "analysis_json=<out/handball/handballNormal_out.json" \
  -F "sport=handball" \
  -F "view_mode=side_by_side" \
  -o viz_reprocessed.jpg
```

---

## Testing via CLI

### 1. Single Image Analysis (`src.analyze_image`)
```bash
PYTHONPATH=. .venv/bin/python -m src.analyze_image \
  --image tests/usfootball/usfootball-01.png \
  --sport usfootball \
  --backend openrouter \
  --model openai/gpt-6-sol \
  --vlm-priors \
  --output out/usfootball/usfootball-01_out.json \
  --debug-dir out/usfootball
```

### 2. Local Re-processing without API Calls (`--reprocess-dir`, `$0.00` cost)
Re-run post-processing, 2D homography, goalkeeper team assignment, and image rendering on already saved `*_out.json` files:
```bash
PYTHONPATH=. .venv/bin/python -m src.analyze_image \
  --reprocess-dir out/foot out/handball \
  --vlm-priors
```

### 3. Multi-Sport Batch Benchmark (`src.run_batch_sports`)
Run a full evaluation across all sports and generate a markdown telemetry report:
```bash
PYTHONPATH=. .venv/bin/python -m src.run_batch_sports \
  --backend openrouter \
  --model openai/gpt-6-sol \
  --vlm-priors
```

---

## Docker & Kubernetes Deployment (`api.t3d.frenchtoast.dev`)

### 1. How Docker Images Are Built (CI/CD Pipeline vs. Manual)
- **Automated GitHub Actions Pipeline ([`.github/workflows/docker-publish.yml`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/.github/workflows/docker-publish.yml))**:
  - Automatically builds and pushes the image to **GitHub Container Registry (`ghcr.io/mathfrenchtoast/t3d-robotflow-basket`)** on every push to `main` (and via manual `workflow_dispatch`).
  - Published tags:
    - `ghcr.io/mathfrenchtoast/t3d-robotflow-basket:latest` (on `main`)
    - `ghcr.io/mathfrenchtoast/t3d-robotflow-basket:sha-<short_commit>`
- **Local Build & Test (Optional)**:
  ```bash
  docker build -t ghcr.io/mathfrenchtoast/t3d-robotflow-basket:latest .
  docker run --rm -p 8001:8000 --env-file .env ghcr.io/mathfrenchtoast/t3d-robotflow-basket:latest
  ```

### 2. Namespace & Manifests Architecture (Kustomize, No Helm Needed)
All Kubernetes resources are managed via **Kustomize** in [`k8s/kustomization.yaml`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/k8s/kustomization.yaml) and strictly isolated in the **`t3d`** namespace:
- **Namespace Creation**: Included directly in [`k8s/kustomization.yaml`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/k8s/kustomization.yaml) via [`k8s/namespace.yaml`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/k8s/namespace.yaml) (and `CreateNamespace=true` in [`k8s/argocd-application.yaml`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/k8s/argocd-application.yaml)).
- **TLS & Gateway API**: [`k8s/gateway.yaml`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/k8s/gateway.yaml) provisions a dedicated `ClusterIssuer` (`letsencrypt-t3d-prod`), Let's Encrypt `Certificate` (`t3d-gateway-tls`), and Traefik `Gateway` (`t3d-gateway`) for `api.t3d.frenchtoast.dev`, routed by [`k8s/httproute.yaml`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/k8s/httproute.yaml).

### 3. Required Kubernetes Secrets (One-Time Setup on `vm-hostinger`)
Because secrets must never be committed to Git, two secrets must be created in the `t3d` namespace before the first deployment:

1. **Create the `t3d` namespace (or apply `k8s/namespace.yaml`):**
   ```bash
   ssh vm-hostinger "kubectl apply -f - <<EOF
   apiVersion: v1
   kind: Namespace
   metadata:
     name: t3d
   EOF"
   ```

2. **Secret #1 — `ghcr-secret` (to pull images from `ghcr.io`):**
   You can either copy the existing `ghcr-secret` already present on `vm-hostinger` or create it with a GitHub PAT (`read:packages`):
   ```bash
   # Option A: Copy existing ghcr-secret on vm-hostinger into namespace t3d
   ssh vm-hostinger "kubectl get secret ghcr-secret -n sf2 -o yaml \
     | sed 's/namespace: sf2/namespace: t3d/' \
     | kubectl apply -n t3d -f -"

   # Option B: Create from scratch with a GitHub token
   ssh vm-hostinger "kubectl -n t3d create secret docker-registry ghcr-secret \
     --docker-server=ghcr.io \
     --docker-username=MathFrenchToast \
     --docker-password=\$GITHUB_PAT"
   ```

3. **Secret #2 — `t3d-sport-api-secrets` (API keys used by the FastAPI container, see template [`k8s/secret.example.yaml`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/k8s/secret.example.yaml)):**
   ```bash
   # Inject keys directly from your local .env file to vm-hostinger
   source .env
   ssh vm-hostinger "kubectl -n t3d create secret generic t3d-sport-api-secrets \
     --from-literal=OPENROUTER_API_KEY='${OPENROUTER_API_KEY}' \
     --from-literal=GOOGLE_API_KEY='${GOOGLE_API_KEY}' \
     --from-literal=ROBOFLOW_API_KEY='${ROBOFLOW_API_KEY}' \
     --dry-run=client -o yaml | kubectl apply -f -"
   ```

### 4. Manual Initial Deployment on `vm-hostinger`
From your local workstation (since `k8s/` is in this repository), you can apply the Kustomization directly over SSH or after cloning on `vm-hostinger`:
```bash
kubectl kustomize k8s/ | ssh vm-hostinger "kubectl apply -f -"
```
Verify the deployment, TLS certificate, and route status:
```bash
ssh vm-hostinger "kubectl get pods,svc,httproute -n t3d && kubectl get gateway,certificate -n traefik"
```

### 5. How to Update the Deployment After Code or Manifest Evolutions

- **Case A — Code / API Evolution (New Commit Pushed to `main`)**:
  1. Push/merge your changes to `main`.
  2. GitHub Actions ([`.github/workflows/docker-publish.yml`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/.github/workflows/docker-publish.yml)) automatically builds and pushes the new `ghcr.io/mathfrenchtoast/t3d-robotflow-basket:latest` image.
  3. Trigger a zero-downtime rolling restart so Kubernetes pulls the new `:latest` image (`imagePullPolicy: Always`):
     ```bash
     ssh vm-hostinger "kubectl -n t3d rollout restart deployment/t3d-sport-api && kubectl -n t3d rollout status deployment/t3d-sport-api"
     ```
- **Case B — Kubernetes Manifest Evolution (Changes in `k8s/`)**:
  Re-apply the Kustomize manifests:
  ```bash
  kubectl kustomize k8s/ | ssh vm-hostinger "kubectl apply -f -"
  ```
- **Case C — GitOps Mode via ArgoCD (Optional)**:
  Apply [`k8s/argocd-application.yaml`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/k8s/argocd-application.yaml) once on `vm-hostinger`:
  ```bash
  ssh vm-hostinger "kubectl apply -f -" < k8s/argocd-application.yaml
  ```
  ArgoCD will then automatically sync any changes pushed to `k8s/` on `main` (`prune: true`, `selfHeal: true`).

---

## Project Structure

- [`src/analyze_image.py`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/src/analyze_image.py): Multi-sport image analysis pipeline, 2D homography orchestration, and visualization rendering.
- [`src/workflow_adapter.py`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/src/workflow_adapter.py): VLM prompt engineering, OpenRouter/Gemini API clients, coordinate normalization, and team classification.
- [`src/court_homography.py`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/src/court_homography.py): RANSAC 2D homography calculator, VLM landmark matching, and perspective fallback projection.
- [`src/sports_config.py`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/src/sports_config.py): Sport-specific dimensions, canonical landmark coordinates (in cm), and prompt priors.
- [`src/visualization.py`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/src/visualization.py): 2D court/pitch drawing (`basket`, `foot`, `handball`, `usfootball`) and bounding box annotations.
- [`src/web/app.py`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/src/web/app.py) & [`src/web/static/index.html`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/src/web/static/index.html): FastAPI server and interactive web client.
- [`k8s/`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/k8s): Kubernetes (`Namespace`, `Gateway`, `Certificate`, `Deployment`, `Service`, `HTTPRoute`) and ArgoCD manifests.
- [`doc/`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/doc): Architecture Decision Records ([`doc/adr.md`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/doc/adr.md)) and 2D homography pipeline documentation ([`doc/pipeline_2d_homography.md`](file:///home/mathieu/dev/projects/T3D/t3d-robotflow-basket/doc/pipeline_2d_homography.md)).

---

## Troubleshooting (Local GPU / Legacy Video Pipeline)

### PyCUDA Build Errors
If `uv sync` fails with `pycuda` build errors when installing local GPU dependencies:
1. **Install Dependencies**: `sudo apt install build-essential python3-dev nvidia-cuda-toolkit`
2. **Export and Sync**:
   ```bash
   export CUDA_INC_DIR=/usr/include
   uv sync
   ```

### GPU Not Found (OVH / Turing Instances)
If `nvidia-smi` returns "No devices were found" or `dmesg` shows `NVRM: RmInitAdapter: Cannot initialize GSP firmware RM`:
```bash
echo "options nvidia NVreg_EnableGpuFirmware=0" | sudo tee /etc/modprobe.d/nvidia-gsp.conf
sudo update-initramfs -u
sudo reboot
```
