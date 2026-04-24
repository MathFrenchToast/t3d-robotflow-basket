import os
from dotenv import load_dotenv
import supervision as sv

# Suppress annoying inference warnings for models we don't use
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_YOLO_WORLD_ENABLED"] = "False"

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Model IDs
PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
NUMBER_RECOGNITION_MODEL_ID = "basketball-jersey-numbers-ocr/3"
KEYPOINT_DETECTION_MODEL_ID = "basketball-court-detection-2/14"
SAM2_MODEL_ID = "sam2/hvit-b" # Model identifier for SAM2

# Model Thresholds
PLAYER_DETECTION_MODEL_CONFIDENCE = 0.4
PLAYER_DETECTION_MODEL_IOU_THRESHOLD = 0.9

KEYPOINT_DETECTION_MODEL_CONFIDENCE = 0.3
KEYPOINT_DETECTION_MODEL_ANCHOR_CONFIDENCE = 0.5
NUMBER_RECOGNITION_MODEL_CONFIDENCE = 0.5
NUMBER_RECOGNITION_MODEL_PROMPT = "Read the number."

# Performance Flags
USE_FAST_TEAM_CLASSIFIER = True
USE_SAM2 = True

# Class IDs
BALL_IN_BASKET_CLASS_ID = 1
NUMBER_CLASS_ID = 2
JUMP_SHOT_CLASS_ID = 5
LAYUP_DUNK_CLASS_ID = 6

# Colors
TEAM_COLORS = {
    "TEAM_A": "#FFFFFF",
    "TEAM_B": "#000000"
}

PLAYER_COLOR_PALETTE = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
    "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

KEYPOINT_COLOR = sv.Color.from_hex('#FF1493')
