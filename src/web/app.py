"""FastAPI application for T3D Multi-Sport Image Analysis & 2D Radar Homography."""

from contextlib import asynccontextmanager
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.analyze_image import analyze_image_workflow, render_workflow_visualizations
from src.sports_config import SPORTS_CONFIGS, resolve_sport
from src.visualization import SportsAnnotator
from src.workflow_adapter import WorkflowAdapter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


class AnalyzerRuntimeCache:
    """In-memory cache for WorkflowAdapter, SportsAnnotator, and court keypoint models."""

    def __init__(self) -> None:
        self.adapter: Optional[WorkflowAdapter] = None
        self.annotators: Dict[str, SportsAnnotator] = {}
        self.court_models: Dict[str, Any] = {}

    def initialize(self) -> None:
        self.adapter = WorkflowAdapter()
        for sport_key in SPORTS_CONFIGS:
            self.annotators[sport_key] = SportsAnnotator(sport=sport_key)

    def get_annotator(self, sport_name: str) -> SportsAnnotator:
        s_key = resolve_sport(sport_name).name
        if s_key not in self.annotators:
            self.annotators[s_key] = SportsAnnotator(sport=s_key)
        return self.annotators[s_key]

    def get_court_model(self, sport_name: str) -> Any:
        sport_cfg = resolve_sport(sport_name)
        model_id = sport_cfg.keypoint_model_id
        if not model_id:
            return None
        if model_id not in self.court_models:
            try:
                from inference import get_model
                self.court_models[model_id] = get_model(model_id)
            except Exception as exc:
                print(f"Court model lazy-load notice ({sport_cfg.name}): {exc}")
                self.court_models[model_id] = None
        return self.court_models[model_id]


runtime_cache = AnalyzerRuntimeCache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize reusable adapters and sport annotators on startup."""
    print("Initializing T3D Multi-Sport Analyzer runtime cache...")
    runtime_cache.initialize()
    print("✓ Runtime cache initialized and ready.")
    yield
    print("Shutting down T3D Multi-Sport Analyzer...")


app = FastAPI(
    title="T3D Sport Image Analyzer API",
    description="Multi-sport player/goalkeeper/referee/ball detection & 2D court homography API",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _encode_jpeg(img: np.ndarray, max_width: int = 2560, quality: int = 92) -> bytes:
    """Resize if needed and encode BGR numpy image as JPEG bytes."""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode output image as JPEG")
    return buf.tobytes()


@app.get("/healthz")
async def healthz():
    """Kubernetes liveness probe."""
    return {"status": "ok", "service": "t3d-sport-analyzer"}


@app.get("/readyz")
async def readyz():
    """Kubernetes readiness probe."""
    adapter = runtime_cache.adapter
    return {
        "status": "ready" if adapter is not None else "initializing",
        "supported_sports": list(SPORTS_CONFIGS.keys()),
        "openrouter_configured": bool(adapter and getattr(adapter, "openrouter_api_key", None)),
        "gemini_configured": bool(adapter and getattr(adapter, "google_api_key", None)),
        "roboflow_configured": bool(adapter and getattr(adapter, "api_key", None)),
    }


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    sport: str = Form("basket"),
    output_format: str = Form("json"),
    return_image: bool = Form(False),
    backend: str = Form("openrouter"),
    model: str = Form("openai/gpt-6-sol"),
    mode: str = Form("high"),
    vlm_priors: bool = Form(True),
    detect_landmarks: bool = Form(True),
):
    """
    Analyze a sports image and return either:
    - JSON (`output_format="json"` and `return_image=False`, default), or
    - JPEG image bytes (`output_format="image"` / `"side_by_side"` / `"annotated"` or `return_image=True`).
    """
    tmp_path: Optional[str] = None
    try:
        content = await image.read()
        if not content:
            return JSONResponse(status_code=400, content={"error": "Uploaded image is empty"})

        orig_name = image.filename or "upload.jpg"
        suffix = Path(orig_name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=f"{Path(orig_name).stem}_") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        resolved_sport = (
            resolve_sport(orig_name).name
            if sport.lower() == "auto"
            else resolve_sport(sport).name
        )
        annotator = runtime_cache.get_annotator(resolved_sport)
        court_model = runtime_cache.get_court_model(resolved_sport)
        effective_priors = bool(vlm_priors or detect_landmarks)

        output, _ = analyze_image_workflow(
            image_path=tmp_path,
            output_path=None,
            debug_dir=None,
            backend=backend,
            sport=resolved_sport,
            mode=mode,
            model_name=model,
            enable_vlm_priors=effective_priors,
            annotator=annotator,
            adapter=runtime_cache.adapter,
            court_model=court_model,
        )

        if output is None:
            return JSONResponse(status_code=400, content={"error": "Failed to analyze image"})

        # Keep clean original filename in image_info
        if "image_info" in output:
            output["image_info"]["filename"] = orig_name

        fmt = output_format.strip().lower()
        wants_image = return_image or fmt in ("image", "side_by_side", "sbs", "annotated", "overlay")

        if wants_image:
            frame = cv2.imread(tmp_path)
            if frame is None:
                return JSONResponse(status_code=400, content={"error": "Failed to decode uploaded image"})

            annotated_frame, side_by_side = render_workflow_visualizations(
                frame=frame,
                output=output,
                annotator=annotator,
            )
            target_img = annotated_frame if fmt in ("annotated", "overlay") else side_by_side
            jpeg_bytes = _encode_jpeg(target_img)

            perf = output.get("performance", {})
            usage = output.get("token_usage", {})
            headers = {
                "X-Detected-Sport": str(output.get("sport", resolved_sport)),
                "X-Homography-Source": str(output.get("homography", {}).get("source", "none")),
                "X-Players-Count": str(len(output.get("detections", {}).get("players", []))),
                "X-Goalkeepers-Count": str(len(output.get("detections", {}).get("goalkeepers", []))),
            }
            proc_time = perf.get("warm_pipeline_seconds")
            if proc_time is not None:
                headers["X-Processing-Time-Seconds"] = str(proc_time)
            vlm_time = perf.get("vlm_inference_seconds")
            if vlm_time is not None:
                headers["X-VLM-Inference-Seconds"] = str(vlm_time)
            cost = usage.get("cost_usd") if usage.get("cost_usd") is not None else usage.get("estimated_cost_usd")
            if cost is not None:
                headers["X-Estimated-Cost-USD"] = str(cost)

            return Response(content=jpeg_bytes, media_type="image/jpeg", headers=headers)

        return {
            "success": True,
            **output,
            "bboxes": output.get("detections", {}),
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/api/visualize")
async def visualize(
    image: UploadFile = File(...),
    analysis_json: Optional[str] = Form(None),
    bboxes_json: Optional[str] = Form(None),
    landmarks_json: Optional[str] = Form(None),
    sport: str = Form("basket"),
    view_mode: str = Form("side_by_side"),
):
    """
    Render annotated visualization (either `side_by_side` with 2D radar or `annotated` overlay)
    from an uploaded image and the JSON returned by `/api/analyze` (0 LLM cost).
    """
    try:
        content = await image.read()
        if not content:
            return JSONResponse(status_code=400, content={"error": "Uploaded image is empty"})

        np_arr = np.frombuffer(content, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse(status_code=400, content={"error": "Failed to decode image"})

        output_payload: Dict[str, Any] = {}
        if analysis_json and analysis_json != "null":
            output_payload = json.loads(analysis_json)
        else:
            det_data = json.loads(bboxes_json) if (bboxes_json and bboxes_json != "null") else {}
            lm_data = json.loads(landmarks_json) if (landmarks_json and landmarks_json != "null") else []
            if isinstance(det_data, dict) and "detections" in det_data:
                output_payload = det_data
            else:
                output_payload = {
                    "sport": resolve_sport(sport).name,
                    "detections": det_data if isinstance(det_data, dict) else {},
                    "landmarks": lm_data if isinstance(lm_data, list) else [],
                }

        if "sport" not in output_payload:
            output_payload["sport"] = resolve_sport(sport).name

        annotator = runtime_cache.get_annotator(output_payload["sport"])
        annotated_frame, side_by_side = render_workflow_visualizations(
            frame=frame,
            output=output_payload,
            annotator=annotator,
        )

        target_img = annotated_frame if view_mode.strip().lower() in ("annotated", "overlay") else side_by_side
        jpeg_bytes = _encode_jpeg(target_img)
        return Response(content=jpeg_bytes, media_type="image/jpeg")

    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON data"})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/docs/vlm-priors")
async def vlm_priors_doc():
    """Serve technical documentation page for VLM Probabilistic Priors & RANSAC 2D Homography."""
    return FileResponse(str(STATIC_DIR / "vlm_priors_doc.html"), media_type="text/html")


@app.get("/docs/output-schema")
async def output_schema_doc():
    """Serve technical documentation page for JSON Output Schema & HTTP Headers."""
    return FileResponse(str(STATIC_DIR / "output_schema_doc.html"), media_type="text/html")


@app.get("/")
async def root():
    """Serve interactive web test interface."""
    return FileResponse(str(STATIC_DIR / "index.html"), media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
