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
        
        # We reuse the logic from analyze_image but use our persistent 'models'
        # To avoid re-importing everything, we can call it directly or refactor analyze_image
        # For now, let's call the function. We might need to pass the model if we refactor it.
        # But wait, analyze_image inside src/analyze_image.py creates its own BasketballModels.
        # I should refactor analyze_image to accept models as an optional argument.
        
        # Instead of calling analyze_image directly (which would recreate models), 
        # let's implement the core logic here or refactor.
        # Let's refactor analyze_image first in the next turn.
        
        # For this turn, I'll just implement it here using the global 'models'
        frame = cv2.imread(tmp_path)
        if frame is None:
            os.unlink(tmp_path)
            return JSONResponse(status_code=400, content={"error": "Failed to load image"})
        
        frame_h, frame_w, _ = frame.shape
        
        # 1. Inference
        player_results = models.player_model.infer(frame, confidence=0.25, iou_threshold=0.9)[0]
        all_detections = sv.Detections.from_inference(player_results)
        
        from src.config import PLAYER_CLASS_IDS, BALL_CLASS_ID, RIM_CLASS_ID, NUMBER_CLASS_ID, USE_SAM2, NUMBER_RECOGNITION_MODEL_PROMPT
        from src.pipeline import get_masked_crops
        
        player_detections = all_detections[np.isin(all_detections.class_id, PLAYER_CLASS_IDS)]
        ball_detections = all_detections[all_detections.class_id == BALL_CLASS_ID]
        rim_detections = all_detections[all_detections.class_id == RIM_CLASS_ID]
        number_detections = all_detections[all_detections.class_id == NUMBER_CLASS_ID]
        
        if USE_SAM2 and len(player_detections) > 0:
            masks = models.get_masks(frame, player_detections)
            if masks is not None:
                player_detections.mask = masks

        if len(player_detections) > 0:
            factor = 1.0 if USE_SAM2 else 0.4
            scaled_detections = sv.Detections(
                xyxy=sv.scale_boxes(xyxy=player_detections.xyxy, factor=factor),
                mask=player_detections.mask,
                confidence=player_detections.confidence,
                class_id=player_detections.class_id
            )
            player_crops = get_masked_crops(frame, scaled_detections)
            models.fit_teams(player_crops)
            team_ids = models.predict_teams(player_crops)
        else:
            team_ids = []

        player_numbers = [None] * len(player_detections)
        if len(number_detections) > 0 and len(player_detections) > 0:
            padded_boxes = sv.pad_boxes(xyxy=number_detections.xyxy, px=10, py=10)
            clipped_boxes = sv.clip_boxes(xyxy=padded_boxes, resolution_wh=(frame_w, frame_h))
            
            numbers = []
            for i, crop_box in enumerate(clipped_boxes):
                number_crop = sv.crop_image(frame, crop_box)
                if number_crop.size > 0:
                    res = models.number_model.infer(number_crop, prompt=NUMBER_RECOGNITION_MODEL_PROMPT)[0].response
                    numbers.append(res)
                else:
                    numbers.append(None)
            
            iou = sv.box_iou_batch(boxes_true=player_detections.xyxy, boxes_detection=number_detections.xyxy)
            for p_idx in range(len(player_detections)):
                best_n_idx = np.argmax(iou[p_idx])
                if iou[p_idx][best_n_idx] > 0.2:
                    player_numbers[p_idx] = numbers[best_n_idx]

        court_results = models.court_model.infer(frame, confidence=0.3)[0]
        keypoints = sv.KeyPoints.from_inference(court_results)
        
        # Prepare JSON response
        detections_out = {"players": [], "ball": [], "rim": []}
        for i in range(len(player_detections)):
            detections_out["players"].append({
                "bbox": player_detections.xyxy[i].tolist(),
                "team": int(team_ids[i]) if i < len(team_ids) else None,
                "number": player_numbers[i],
                "confidence": float(player_detections.confidence[i])
            })
        for i in range(len(ball_detections)):
            detections_out["ball"].append({"bbox": ball_detections.xyxy[i].tolist(), "confidence": float(ball_detections.confidence[i])})
        for i in range(len(rim_detections)):
            detections_out["rim"].append({"bbox": rim_detections.xyxy[i].tolist(), "confidence": float(rim_detections.confidence[i])})
            
        landmarks_out = []
        if len(keypoints) > 0:
            for i in range(len(keypoints.xy[0])):
                conf = float(keypoints.confidence[0][i])
                if conf > 0.3:
                    landmarks_out.append({"id": i, "x": float(keypoints.xy[0][i][0]), "y": float(keypoints.xy[0][i][1]), "confidence": conf})

        # Visualization
        labels = []
        for p in detections_out["players"]:
            team_label = f"T{p['team']}" if p['team'] is not None else ""
            num_label = f"#{p['number']}" if p['number'] is not None else ""
            labels.append(f"{team_label} {num_label}".strip())
            
        annotated_frame = annotator.annotate_frame(frame, player_detections, labels)
        if len(ball_detections) > 0:
            annotated_frame = annotator.box_annotator.annotate(annotated_frame, ball_detections)
        if len(rim_detections) > 0:
            annotated_frame = annotator.box_annotator.annotate(annotated_frame, rim_detections)
        if len(keypoints) > 0:
            annotated_frame = annotator.annotate_keypoints(annotated_frame, keypoints)
            
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
        
        # We'll return the JSON and a path to the visualization
        # In a real app we might serve the image as a separate request or base64
        # For simplicity, we'll return the image file in another endpoint or here
        # Let's return JSON with a 'viz_url' pointing to a temp serve
        
        # Actually, let's just return the image for now if requested, or both
        # The user wanted "complete json", so let's return JSON.
        
        return {
            "success": True,
            "detections": detections_out,
            "landmarks": landmarks_out,
            "image_info": {"width": frame_w, "height": frame_h},
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
