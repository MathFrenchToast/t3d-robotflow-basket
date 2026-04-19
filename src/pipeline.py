import supervision as sv
import numpy as np
from tqdm import tqdm
from sports import ViewTransformer
from src.config import (
    PLAYER_DETECTION_MODEL_CONFIDENCE,
    PLAYER_DETECTION_MODEL_IOU_THRESHOLD,
    KEYPOINT_DETECTION_MODEL_CONFIDENCE,
    BALL_IN_BASKET_CLASS_ID,
    JUMP_SHOT_CLASS_ID,
    LAYUP_DUNK_CLASS_ID
)
from src.models import BasketballModels
from src.tracking import initialize_trackers
from src.visualization import BasketballAnnotator

def run_pipeline(source_video_path: str, target_video_path: str):
    models = BasketballModels()
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    byte_tracker, shot_tracker, number_validator, team_validator = initialize_trackers(video_info.fps)
    annotator = BasketballAnnotator()
    
    frame_generator = sv.get_video_frames_generator(source_video_path)
    
    is_team_classifier_fitted = False

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame_index, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):
            # 1. Inference
            player_results = models.player_model.infer(
                frame, 
                confidence=PLAYER_DETECTION_MODEL_CONFIDENCE, 
                iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD
            )[0]
            detections = sv.Detections.from_inference(player_results)

            court_results = models.court_model.infer(
                frame, 
                confidence=KEYPOINT_DETECTION_MODEL_CONFIDENCE
            )[0]
            keypoints = sv.KeyPoints.from_inference(court_results)
            
            # 2. Track Players
            # Filter for player class before tracking if necessary (depending on model)
            detections = byte_tracker.update_with_detections(detections)
            
            # 3. Shot Detection Logic
            has_jump_shot = JUMP_SHOT_CLASS_ID in detections.class_id
            has_layup_dunk = LAYUP_DUNK_CLASS_ID in detections.class_id
            has_ball_in_basket = BALL_IN_BASKET_CLASS_ID in detections.class_id
            
            shot_events = shot_tracker.update(
                frame_index=frame_index,
                has_jump_shot=has_jump_shot,
                has_layup_dunk=has_layup_dunk,
                has_ball_in_basket=has_ball_in_basket
            )
            
            # 4. Team Classification Calibration
            if not is_team_classifier_fitted and len(detections) > 4:
                player_boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
                player_crops = [sv.crop_image(frame, box) for box in player_boxes]
                models.fit_teams(player_crops)
                is_team_classifier_fitted = True
            
            # 5. Team & Jersey Identification
            if is_team_classifier_fitted:
                player_boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
                player_crops = [sv.crop_image(frame, box) for box in player_boxes]
                team_ids = models.predict_teams(player_crops)
                team_validator.update(tracker_ids=detections.tracker_id, values=team_ids)

            # 6. Annotation
            validated_numbers = number_validator.get_validated(tracker_ids=detections.tracker_id)
            validated_teams = team_validator.get_validated(tracker_ids=detections.tracker_id)
            
            labels = []
            for tid, num, team in zip(detections.tracker_id, validated_numbers, validated_teams):
                team_label = f"T{team}" if team is not None else ""
                num_label = f"#{num}" if num is not None else f"ID{tid}"
                labels.append(f"{team_label} {num_label}")
            
            annotated_frame = annotator.annotate_frame(frame, detections, labels)
            
            # 7. Court Transformation & Overlay
            if len(keypoints) > 0:
                view_transformer = ViewTransformer(
                    source=keypoints.xy[0],
                    target=np.array(annotator.court_config.vertices)
                )
                points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                transformed_points = view_transformer.transform_points(points=points)
                
                court_image = annotator.draw_court_overlay(transformed_points)
                annotated_frame = annotator.overlay_court(annotated_frame, court_image)

            # Add shot event overlay if needed
            if shot_events:
                # Logic to visualize shot events could be added here
                pass
                
            sink.write_frame(annotated_frame)
