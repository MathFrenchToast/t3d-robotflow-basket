from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import json
import tempfile
import cv2
import os
import numpy as np
from pathlib import Path
import supervision as sv

from src.models import BasketballModels
from src.analyze_image import analyze_image
from src.visualization import BasketballAnnotator

app = FastAPI(title="Basketball AI Analyzer")

# Global models instance to avoid re-initialization
print("Initializing Basketball Models...")
models = BasketballModels()
annotator = BasketballAnnotator()
print("✓ Models loaded and ready.")

# Get the directory of the current file
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Create static dir if it doesn't exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)):
    """Analyze basketball image and return detections and landmarks."""
    try:
        # Save uploaded image to temp file
        suffix = Path(image.filename).suffix if image.filename else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Use our persistent 'models' and 'annotator' with the refactored analyze_image
        output, annotated_frame = analyze_image(
            image_path=tmp_path, 
            models=models, 
            annotator=annotator,
            debug_dir=None # Don't save to 'out' directory in web mode
        )
        
        if output is None:
            os.unlink(tmp_path)
            return JSONResponse(status_code=400, content={"error": "Failed to analyze image"})
        
        frame_h, frame_w = output["image_info"]["height"], output["image_info"]["width"]
            
        # Resize for display if too large
        if frame_w > 1280:
            scale = 1280 / frame_w
            annotated_frame = cv2.resize(annotated_frame, (1280, int(frame_h * scale)))

        # Save visualization to a temporary file to serve
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_viz:
            cv2.imwrite(tmp_viz.name, annotated_frame)
            viz_path = tmp_viz.name

        # Cleanup input image
        os.unlink(tmp_path)
        
        return {
            "success": True,
            "detections": output["detections"],
            "landmarks": output["landmarks"],
            "image_info": output["image_info"],
            "viz_id": os.path.basename(viz_path)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/viz/{viz_id}")
async def get_viz(viz_id: str):
    """Serve the generated visualization."""
    temp_dir = tempfile.gettempdir()
    viz_path = os.path.join(temp_dir, viz_id)
    if os.path.exists(viz_path):
        return FileResponse(viz_path, media_type="image/jpeg")
    return JSONResponse(status_code=404, content={"error": "Visualization not found"})

@app.get("/")
async def root():
    """Serve index page."""
    return FileResponse(str(STATIC_DIR / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
