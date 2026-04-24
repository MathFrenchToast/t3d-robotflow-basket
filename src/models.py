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
    SAM2_MODEL_ID
)

def load_player_detection_model():
    return get_model(PLAYER_DETECTION_MODEL_ID)

def load_number_recognition_model():
    return get_model(NUMBER_RECOGNITION_MODEL_ID)

def load_court_detection_model():
    return get_model(KEYPOINT_DETECTION_MODEL_ID)

def load_sam2_model():
    return get_model(SAM2_MODEL_ID)

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
        # Focus on the middle 50% of the crop to avoid background noise
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
        self.sam2_model = load_sam2_model() if USE_SAM2 else None
        
        # Graceful handling of CUDA failure
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"
            
        if USE_FAST_TEAM_CLASSIFIER:
            self.team_classifier = SimpleTeamClassifier()
        else:
            self.team_classifier = TeamClassifier(device=self.device)

    def fit_teams(self, player_crops):
        """Fits the team classifier using the provided player crops."""
        if len(player_crops) > 0:
            self.team_classifier.fit(player_crops)

    def predict_teams(self, player_crops):
        """Predicts the team for each player crop."""
        if len(player_crops) > 0:
            return self.team_classifier.predict(player_crops)
        return []

    def get_masks(self, frame, detections):
        """Generates masks for detections using SAM2."""
        if self.sam2_model is None or len(detections) == 0:
            return None
        
        # SAM2 via inference can take multiple bboxes
        # We use the center of the boxes as points or the boxes themselves
        outputs = self.sam2_model.infer(frame, bboxes=detections.xyxy.tolist())
        if isinstance(outputs, list):
            masks = [np.array(out.mask, dtype=bool) for out in outputs]
            return np.stack(masks) if masks else None
        return None
