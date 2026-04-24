import supervision as sv
import numpy as np
import cv2
import os
from tqdm import tqdm
from sports import ViewTransformer
from src.config import (
    PLAYER_DETECTION_MODEL_CONFIDENCE,
    PLAYER_DETECTION_MODEL_IOU_THRESHOLD,
    KEYPOINT_DETECTION_MODEL_CONFIDENCE,
    NUMBER_RECOGNITION_MODEL_CONFIDENCE,
    NUMBER_RECOGNITION_MODEL_PROMPT,
    BALL_IN_BASKET_CLASS_ID,
    NUMBER_CLASS_ID,
    JUMP_SHOT_CLASS_ID,
    LAYUP_DUNK_CLASS_ID,
    USE_SAM2
)
from src.models import BasketballModels
from src.tracking import initialize_trackers
from src.visualization import BasketballAnnotator

def coords_above_threshold(matrix: np.ndarray, threshold: float):
    A = np.asarray(matrix)
    rows, cols = np.where(A > threshold)
    return list(zip(rows.tolist(), cols.tolist()))

def get_masked_crops(frame, detections):
    """Extracts crops from the frame, and if masks are available, blacks out the background."""
    crops = []
    for i in range(len(detections)):
        box = detections.xyxy[i]
        crop = sv.crop_image(frame, box)
        
        if detections.mask is not None:
            mask = detections.mask[i]
            # Crop the mask to the same size as the player crop
            crop_mask = sv.crop_image(mask.astype(np.uint8) * 255, box)
            if crop_mask.shape[:2] != crop.shape[:2]:
                crop_mask = cv2.resize(crop_mask, (crop.shape[1], crop.shape[0]))
            
            # Apply mask to black out background
            crop = cv2.bitwise_and(crop, crop, mask=crop_mask)
            
        crops.append(crop)
    return crops

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
            all_detections = sv.Detections.from_inference(player_results)
            
            # Filter detections for players and numbers separately
            # Note: in this model, class_id might represent players, ball, etc.
            # We assume PLAYER_CLASS_ID is what's used for tracking. 
            # Looking at notebook, they use class_id == NUMBER_CLASS_ID for numbers.
            detections = all_detections[all_detections.class_id != NUMBER_CLASS_ID]
            number_detections = all_detections[all_detections.class_id == NUMBER_CLASS_ID]

            court_results = models.court_model.infer(
                frame, 
                confidence=KEYPOINT_DETECTION_MODEL_CONFIDENCE
            )[0]
            keypoints = sv.KeyPoints.from_inference(court_results)
            
            # 2. Track Players
            detections = byte_tracker.update_with_detections(detections)
            
            # 2.1 Refine with SAM2 Masks (Optional but recommended for accuracy)
            if USE_SAM2 and len(detections) > 0:
                masks = models.get_masks(frame, detections)
                if masks is not None:
                    detections.mask = masks

            # 3. Shot Detection Logic
            has_jump_shot = JUMP_SHOT_CLASS_ID in all_detections.class_id
            has_layup_dunk = LAYUP_DUNK_CLASS_ID in all_detections.class_id
            has_ball_in_basket = BALL_IN_BASKET_CLASS_ID in all_detections.class_id
            
            shot_events = shot_tracker.update(
                frame_index=frame_index,
                has_jump_shot=has_jump_shot,
                has_layup_dunk=has_layup_dunk,
                has_ball_in_basket=has_ball_in_basket
            )
            
            # 4. Team Classification Calibration
            if not is_team_classifier_fitted and len(detections) > 4:
                # In SAM2 mode, we don't scale down as much because masks already handle noise
                factor = 1.0 if USE_SAM2 else 0.4
                scaled_detections = sv.Detections(
                    xyxy=sv.scale_boxes(xyxy=detections.xyxy, factor=factor),
                    mask=detections.mask,
                    confidence=detections.confidence,
                    class_id=detections.class_id,
                    tracker_id=detections.tracker_id,
                    data=detections.data
                )
                player_crops = get_masked_crops(frame, scaled_detections)
                
                # Save calibration crops
                for i, crop in enumerate(player_crops):
                    if crop.size > 0:
                        cv2.imwrite(f"debug_crops/players/calib_{i}.png", crop)
                
                models.fit_teams(player_crops)
                is_team_classifier_fitted = True
            
            # 5. Team & Jersey Identification
            if is_team_classifier_fitted:
                factor = 1.0 if USE_SAM2 else 0.4
                scaled_detections = sv.Detections(
                    xyxy=sv.scale_boxes(xyxy=detections.xyxy, factor=factor),
                    mask=detections.mask,
                    confidence=detections.confidence,
                    class_id=detections.class_id,
                    tracker_id=detections.tracker_id,
                    data=detections.data
                )
                player_crops = get_masked_crops(frame, scaled_detections)
                
                team_ids = models.predict_teams(player_crops)
                team_validator.update(tracker_ids=detections.tracker_id, values=team_ids)
                
                # Save some prediction crops for debug (first 10 frames after calib)
                if frame_index < 50:
                    for i, (crop, tid, team) in enumerate(zip(player_crops, detections.tracker_id, team_ids)):
                        if crop.size > 0:
                            cv2.imwrite(f"debug_crops/players/frame_{frame_index}_id{tid}_team{team}.png", crop)

            # Jersey Number Recognition (every 5 frames to save GPU)
            if frame_index % 5 == 0 and len(number_detections) > 0 and len(detections) > 0:
                frame_h, frame_w, _ = frame.shape
                # Crop and Recognize
                padded_boxes = sv.pad_boxes(xyxy=number_detections.xyxy, px=10, py=10)
                clipped_boxes = sv.clip_boxes(xyxy=padded_boxes, resolution_wh=(frame_w, frame_h))
                
                numbers = []
                for i, crop_box in enumerate(clipped_boxes):
                    number_crop = sv.crop_image(frame, crop_box)
                    if number_crop.size > 0:
                        # Save number crop for debug
                        cv2.imwrite(f"debug_crops/numbers/frame_{frame_index}_n{i}.png", number_crop)
                        
                        # Use infer for SmolVLM based OCR model to ensure preprocessing is handled
                        res = models.number_model.infer(number_crop, prompt=NUMBER_RECOGNITION_MODEL_PROMPT)[0].response
                        numbers.append(res)
                    else:
                        numbers.append(None)
                
                # Match numbers with tracked players using Box IoU (or IoS if available)
                # Since we don't have masks here, we use box IoU
                iou = sv.box_iou_batch(
                    boxes_true=detections.xyxy,
                    boxes_detection=number_detections.xyxy
                )
                
                pairs = coords_above_threshold(iou, 0.2) # lower threshold for boxes
                if pairs:
                    valid_matched_player_ids = []
                    valid_matched_numbers = []
                    for p_idx, n_idx in pairs:
                        val = numbers[n_idx]
                        if val is not None:
                            valid_matched_player_ids.append(detections.tracker_id[p_idx])
                            valid_matched_numbers.append(val)
                    
                    if valid_matched_player_ids:
                        number_validator.update(tracker_ids=valid_matched_player_ids, values=valid_matched_numbers)

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

            sink.write_frame(annotated_frame)
