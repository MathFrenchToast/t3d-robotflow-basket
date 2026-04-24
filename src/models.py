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
    """Loads SAM2 model by bypassing the buggy get_model factory if necessary."""
    try:
        # Try the standard way first
        return get_model(SAM2_MODEL_ID)
    except TypeError as e:
        if "multiple values for argument 'model_id'" in str(e):
            # Bypass the factory and use the model class directly
            try:
                from inference.models.sam2.segment_anything_2 import SegmentAnything2
                # We instantiate without the factory to avoid the double argument bug
                return SegmentAnything2(
                    model_id=SAM2_MODEL_ID, 
                    api_key=ROBOFLOW_API_KEY
                )
            except Exception as e2:
                print(f"Failed to bypass SAM2 factory: {e2}")
                raise e
        raise e

class SimpleTeamClassifier:
    """A fast, color-based team classifier using K-Means."""
    def __init__(self):
        self.kmeans = KMeans(n_clusters=2, n_init=10)
        self.is_fitted = False

    def get_mean_color(self, crop):
        """Extracts the mean color of a player crop, focusing on the center (jersey)."""
        if crop is None or crop.size == 0:
            return np.array([0, 0, 0])
        h, w, _ = crop.shape
        center_crop = crop[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
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
        if self.sam2_model is None or len(detections) == 0:
            return None
        
        try:
            outputs = self.sam2_model.infer(frame, bboxes=detections.xyxy.tolist())
            if isinstance(outputs, list):
                masks = [np.array(out.mask, dtype=bool) for out in outputs]
                return np.stack(masks) if masks else None
        except Exception as e:
            print(f"SAM2 inference failed: {e}")
            return None
        return None
