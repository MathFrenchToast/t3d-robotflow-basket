import supervision as sv
import numpy as np
from tqdm import tqdm
from src.config import (
    PLAYER_DETECTION_MODEL_CONFIDENCE,
    PLAYER_DETECTION_MODEL_IOU_THRESHOLD,
    KEYPOINT_DETECTION_MODEL_CONFIDENCE
)
from src.models import BasketballModels
from src.tracking import initialize_trackers
from src.visualization import BasketballAnnotator
from sports.common import ViewTransformer

def run_pipeline(source_video_path: str, target_video_path: str):
    models = BasketballModels()
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    byte_tracker, shot_tracker, number_validator, team_validator = initialize_trackers(video_info.fps)
    annotator = BasketballAnnotator()
    
    frame_generator = sv.get_video_frames_generator(source_video_path)
    
    view_transformer = None
    is_team_classifier_fitted = False

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            # 1. Player & Ball Detection
            player_results = models.player_model.infer(
                frame, 
                confidence=PLAYER_DETECTION_MODEL_CONFIDENCE, 
                iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD
            )[0]
            detections = sv.Detections.from_inference(player_results)
            
            # 2. Tracking
            detections = byte_tracker.update_with_detections(detections)
            
            # 3. Team Classification Calibration (On first few detections)
            if not is_team_classifier_fitted and len(detections) > 4:
                # Scale boxes for better feature extraction from jerseys
                player_boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
                player_crops = [sv.crop_image(frame, box) for box in player_boxes]
                models.fit_teams(player_crops)
                is_team_classifier_fitted = True
            
            # 4. Court Detection (Keypoints)
            court_results = models.court_model.infer(
                frame, 
                confidence=KEYPOINT_DETECTION_MODEL_CONFIDENCE
            )[0]
            keypoints = sv.KeyPoints.from_inference(court_results)
            
            # 5. Annotation & Logic
            # Team Prediction
            if is_team_classifier_fitted:
                player_boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
                player_crops = [sv.crop_image(frame, box) for box in player_boxes]
                team_ids = models.predict_teams(player_crops)
                team_validator.update(tracker_ids=detections.tracker_id, values=team_ids)

            # Labels Generation
            validated_numbers = number_validator.get_validated(tracker_ids=detections.tracker_id)
            validated_teams = team_validator.get_validated(tracker_ids=detections.tracker_id)
            
            labels = []
            for tid, num, team in zip(detections.tracker_id, validated_numbers, validated_teams):
                team_label = f"T{team}" if team is not None else ""
                num_label = f"#{num}" if num is not None else f"ID{tid}"
                labels.append(f"{team_label} {num_label}")
            
            annotated_frame = annotator.annotate_frame(frame, detections, labels)
            annotated_frame = annotator.annotate_keypoints(annotated_frame, keypoints)
            
            sink.write_frame(annotated_frame)
