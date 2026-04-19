import supervision as sv
import cv2
import numpy as np
from sports.basketball import (
    CourtConfiguration,
    League,
    draw_court,
    draw_points_on_court,
    draw_paths_on_court,
    draw_made_and_miss_on_court
)
from src.config import PLAYER_COLOR_PALETTE, KEYPOINT_COLOR

class BasketballAnnotator:
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(color=PLAYER_COLOR_PALETTE, thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            color=PLAYER_COLOR_PALETTE, 
            text_color=sv.Color.BLACK
        )
        self.vertex_annotator = sv.VertexAnnotator(color=KEYPOINT_COLOR, radius=8)
        
        # Court setup
        self.court_config = CourtConfiguration(league=League.NBA)
        self.court_image = draw_court(config=self.court_config)

    def annotate_frame(self, frame, detections, labels):
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(), 
            detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, 
            detections=detections, 
            labels=labels
        )
        return annotated_frame

    def annotate_keypoints(self, frame, keypoints):
        return self.vertex_annotator.annotate(
            scene=frame.copy(),
            keypoints=keypoints
        )

    def draw_court_overlay(self, detections_xy):
        return draw_points_on_court(
            config=self.court_config,
            detections=detections_xy,
            court=self.court_image.copy()
        )
