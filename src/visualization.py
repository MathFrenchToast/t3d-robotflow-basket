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
            xy=detections_xy,
            court=self.court_image.copy()
        )

    def overlay_court(self, frame, court_image):
        fh, fw, _ = frame.shape
        ch, cw, _ = court_image.shape
        
        # Scaling the court image to be roughly 1/4 of the frame height
        scale_factor = (fh / 4.0) / ch
        new_cw = int(cw * scale_factor)
        new_ch = int(ch * scale_factor)
        resized_court = cv2.resize(court_image, (new_cw, new_ch))
        
        # Place in top-right corner with 20px padding
        padding = 20
        frame[padding:padding+new_ch, fw-new_cw-padding:fw-padding] = resized_court
        return frame
