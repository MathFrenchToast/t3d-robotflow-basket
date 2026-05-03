from inference import get_model
import torch
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sports import TeamClassifier
from src.config import (
    PLAYER_DETECTION_MODEL_ID,
    NUMBER_RECOGNITION_MODEL_ID,
    KEYPOINT_DETECTION_MODEL_ID,
    USE_FAST_TEAM_CLASSIFIER,
    USE_SAM2,
    SAM2_MODEL_ID,
    ROBOFLOW_API_KEY
)

def load_player_detection_model():
    return get_model(PLAYER_DETECTION_MODEL_ID)

def load_number_recognition_model():
    return get_model(NUMBER_RECOGNITION_MODEL_ID)

def load_court_detection_model():
    return get_model(KEYPOINT_DETECTION_MODEL_ID)

def load_sam2_model():
    """Loads SAM2 model using the standard factory."""
    return get_model(SAM2_MODEL_ID)

class SimpleTeamClassifier:
    """A fast, color-based team classifier using K-Means."""
    def __init__(self):
        self.kmeans = KMeans(n_clusters=2, n_init=10)
        self.is_fitted = False

    def get_mean_color(self, crop):
        """Extracts the mean color of a player crop, ignoring black background (from masks)."""
        if crop is None or crop.size == 0:
            return np.array([0, 0, 0])
        
        # Focus on the center (jersey)
        h, w, _ = crop.shape
        center_crop = crop[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        
        # Find non-black pixels
        # A pixel is non-black if at least one channel is > 0
        non_black_mask = np.any(center_crop > 0, axis=-1)
        
        if np.any(non_black_mask):
            return np.mean(center_crop[non_black_mask], axis=0)
        else:
            # Fallback to standard mean if all pixels are black (shouldn't happen with valid crops)
            return np.mean(center_crop, axis=(0, 1))

    def fit(self, player_crops):
        if len(player_crops) < 2:
            return
        colors = [self.get_mean_color(crop) for crop in player_crops]
        self.kmeans.fit(colors)
        self.is_fitted = True

    def predict(self, player_crops):
        if not self.is_fitted or len(player_crops) == 0:
            return [0] * len(player_crops)
        colors = [self.get_mean_color(crop) for crop in player_crops]
        return self.kmeans.predict(colors)

class BasketballModels:
    def __init__(self):
        self.player_model = load_player_detection_model()
        self.number_model = load_number_recognition_model()
        self.court_model = load_court_detection_model()
        
        self.sam2_model = None
        if USE_SAM2:
            try:
                self.sam2_model = load_sam2_model()
            except Exception as e:
                print(f"Warning: Could not load SAM2 model: {e}. Falling back to bounding boxes.")
        
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"
            
        if USE_FAST_TEAM_CLASSIFIER:
            self.team_classifier = SimpleTeamClassifier()
        else:
            self.team_classifier = TeamClassifier(device=self.device)

    def fit_teams(self, player_crops):
        if len(player_crops) > 0:
            self.team_classifier.fit(player_crops)

    def predict_teams(self, player_crops):
        if len(player_crops) > 0:
            return self.team_classifier.predict(player_crops)
        return []

    def get_masks(self, frame, detections):
        """Generates masks for detections using SAM2 via the segment_image method."""
        if self.sam2_model is None or len(detections) == 0:
            return None
        
        try:
            # Prepare prompts in the format expected by segment_image
            # xyxy -> center_x, center_y, width, height
            prompts = []
            for box in detections.xyxy:
                x1, y1, x2, y2 = box
                prompts.append({
                    "box": {
                        "x": (x1 + x2) / 2,
                        "y": (y1 + y2) / 2,
                        "width": x2 - x1,
                        "height": y2 - y1
                    }
                })
            
            # segment_image returns (logits, scores, low_res_logits)
            results = self.sam2_model.segment_image(
                image=frame, 
                prompts={"prompts": prompts}
            )
            
            # Extract logits (the first element of the tuple)
            logits = results[0]
            # Convert logits to binary masks (threshold at 0.0)
            masks = (logits > 0.0)
            
            # If there's only one detection, the shape might be (1, H, W) 
            # or if multiple (N, H, W). Ensuring it's always (N, H, W).
            if masks.ndim == 2:
                masks = masks[None, ...]
                
            return masks
        except Exception as e:
            import traceback
            print(f"SAM2 segmentation failed with error type: {type(e).__name__}")
            print(f"Error details: {e}")
            # Keep traceback for deep debugging if needed on VM
            # traceback.print_exc() 
            return None
        return None
