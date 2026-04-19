from inference import get_model
import torch
from sports import TeamClassifier
from src.config import (
    PLAYER_DETECTION_MODEL_ID,
    NUMBER_RECOGNITION_MODEL_ID,
    KEYPOINT_DETECTION_MODEL_ID
)

def load_player_detection_model():
    return get_model(model_id=PLAYER_DETECTION_MODEL_ID)

def load_number_recognition_model():
    return get_model(model_id=NUMBER_RECOGNITION_MODEL_ID)

def load_court_detection_model():
    return get_model(model_id=KEYPOINT_DETECTION_MODEL_ID)

class BasketballModels:
    def __init__(self):
        self.player_model = load_player_detection_model()
        self.number_model = load_number_recognition_model()
        self.court_model = load_court_detection_model()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
