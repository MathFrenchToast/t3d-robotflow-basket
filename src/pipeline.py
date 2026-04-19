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
            
            # 3. Court Detection (Keypoints)
            court_results = models.court_model.infer(
                frame, 
                confidence=KEYPOINT_DETECTION_MODEL_CONFIDENCE
            )[0]
            keypoints = sv.KeyPoints.from_inference(court_results)
            
            # 4. Homography / View Transformation
            # Note: In a real scenario, we'd match detected keypoints to court coordinates
            if len(keypoints) > 0:
                # This is a simplified transformation logic
                # Actual logic would involve matching keypoint IDs to court landmarks
                pass

            # 5. Jersey Number Recognition (Optional/Heuristic)
            # In a full run, we'd crop player detections and run number_model.infer
            
            # 6. Annotation
            # Get validated numbers if any (currently defaults to tracker_id)
            validated_numbers = number_validator.get_validated(tracker_ids=detections.tracker_id)
            labels = [
                f"#{num if num is not None else tid}" 
                for tid, num in zip(detections.tracker_id, validated_numbers)
            ]
            
            annotated_frame = annotator.annotate_frame(frame, detections, labels)
            annotated_frame = annotator.annotate_keypoints(annotated_frame, keypoints)
            
            sink.write_frame(annotated_frame)
