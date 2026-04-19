from inference import get_model
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
