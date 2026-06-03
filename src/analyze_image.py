import cv2
import json
import argparse
import os
import time
import numpy as np
import supervision as sv
from pathlib import Path

# Add src to path if needed, but we expect to run as 'python -m src.analyze_image'
try:
    from src.models import BasketballModels
    from src.config import (
        PLAYER_DETECTION_MODEL_CONFIDENCE,
        PLAYER_DETECTION_MODEL_IOU_THRESHOLD,
        KEYPOINT_DETECTION_MODEL_CONFIDENCE,
        NUMBER_RECOGNITION_MODEL_CONFIDENCE,
        NUMBER_RECOGNITION_MODEL_PROMPT,
        PLAYER_CLASS_IDS,
        NUMBER_CLASS_ID,
        BALL_CLASS_ID,
        RIM_CLASS_ID,
        USE_SAM
    )
    from src.pipeline import get_masked_crops
except ImportError:
    # Fallback for direct script execution if not as module
    from models import BasketballModels
    from config import (
        PLAYER_DETECTION_MODEL_CONFIDENCE,
        PLAYER_DETECTION_MODEL_IOU_THRESHOLD,
        KEYPOINT_DETECTION_MODEL_CONFIDENCE,
        NUMBER_RECOGNITION_MODEL_CONFIDENCE,
        NUMBER_RECOGNITION_MODEL_PROMPT,
        PLAYER_CLASS_IDS,
        NUMBER_CLASS_ID,
        BALL_CLASS_ID,
        RIM_CLASS_ID,
        USE_SAM
    )
    from pipeline import get_masked_crops

def analyze_image(image_path, output_path=None, debug_dir="out", models=None, annotator=None, debug=False):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return None, None

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Failed to load image {image_path}")
        return None, None
    
    frame_h, frame_w, _ = frame.shape
    
    # Initialize models if not provided
    if models is None:
        print("Loading models...")
        models = BasketballModels()
    
    # Start timing after models are loaded
    start_time = time.perf_counter()
    
    # 1. Inference - Players and objects
    print("Detecting players and objects...")
    player_results = models.player_model.infer(
        frame, 
        confidence=PLAYER_DETECTION_MODEL_CONFIDENCE, 
        iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD
    )[0]
    all_detections = sv.Detections.from_inference(player_results).with_nms(
        threshold=0.8, class_agnostic=True
    )
    
    # Filter detections
    player_detections = all_detections[np.isin(all_detections.class_id, PLAYER_CLASS_IDS)]
    ball_detections = all_detections[all_detections.class_id == BALL_CLASS_ID]
    rim_detections = all_detections[all_detections.class_id == RIM_CLASS_ID]
    number_detections = all_detections[all_detections.class_id == NUMBER_CLASS_ID]
    
    # 2. SAM Masks (if enabled)
    if USE_SAM and len(player_detections) > 0:
        print("Generating SAM masks...")
        masks = models.get_masks(frame, player_detections)
        if masks is not None:
            player_detections.mask = masks

    # 3. Team Classification
    print("Classifying teams...")
    if len(player_detections) > 0:
        factor = 1.0 if USE_SAM else 0.4
        scaled_detections = sv.Detections(
            xyxy=sv.scale_boxes(xyxy=player_detections.xyxy, factor=factor),
            mask=player_detections.mask,
            confidence=player_detections.confidence,
            class_id=player_detections.class_id
        )
        player_crops = get_masked_crops(frame, scaled_detections)
        
        # Fit on this image's players
        models.fit_teams(player_crops)
        team_ids = models.predict_teams(player_crops)

        if debug and debug_dir:
            crops_dir = os.path.join(debug_dir, "crops")
            os.makedirs(crops_dir, exist_ok=True)
            image_stem = Path(image_path).stem
            for i, (crop, tid) in enumerate(zip(player_crops, team_ids)):
                if crop.size > 0:
                    # Isolate the jersey (roughly 15% to 60% of height, 20% to 80% of width)
                    h, w, _ = crop.shape
                    jersey_crop = crop[int(h*0.15):int(h*0.60), int(w*0.2):int(w*0.8)]
                    if jersey_crop.size > 0:
                        cv2.imwrite(os.path.join(crops_dir, f"{image_stem}_crop_{i}_team{tid}.png"), jersey_crop)
    else:
        team_ids = []

    # 4. Jersey Number Recognition
    print("Recognizing jersey numbers...")
    player_numbers = [None] * len(player_detections)
    if len(number_detections) > 0 and len(player_detections) > 0:
        padded_boxes = sv.pad_boxes(xyxy=number_detections.xyxy, px=10, py=10)
        clipped_boxes = sv.clip_boxes(xyxy=padded_boxes, resolution_wh=(frame_w, frame_h))
        
        numbers = []
        for i, crop_box in enumerate(clipped_boxes):
            number_crop = sv.crop_image(frame, crop_box)
            if number_crop.size > 0:
                res = models.number_model.infer(number_crop, prompt=NUMBER_RECOGNITION_MODEL_PROMPT)[0].response
                numbers.append(res)
            else:
                numbers.append(None)
        
        # Match with players using IoU
        iou = sv.box_iou_batch(
            boxes_true=player_detections.xyxy,
            boxes_detection=number_detections.xyxy
        )
        
        for p_idx in range(len(player_detections)):
            best_n_idx = np.argmax(iou[p_idx])
            if iou[p_idx][best_n_idx] > 0.2:
                player_numbers[p_idx] = numbers[best_n_idx]

    # 5. Court Keypoints
    print("Detecting court keypoints...")
    # Match video pipeline confidence threshold (0.3) for better coverage
    from src.config import KEYPOINT_DETECTION_MODEL_CONFIDENCE
    court_results = models.court_model.infer(
        frame, 
        confidence=KEYPOINT_DETECTION_MODEL_CONFIDENCE
    )[0]
    keypoints = sv.KeyPoints.from_inference(court_results)
    
    # End timing
    pipeline_time = time.perf_counter() - start_time
    print(f"Pipeline execution time: {pipeline_time:.3f} seconds")
    
    # Structure output
    output = {
        "image_info": {
            "path": image_path,
            "width": frame_w,
            "height": frame_h
        },
        "performance": {
            "pipeline_time_seconds": round(pipeline_time, 4)
        },
        "detections": {
            "players": [],
            "ball": [],
            "rim": []
        },
        "landmarks": []
    }
    
    # Map class IDs to names for readability
    class_id_to_name = {
        3: "player",
        4: "player_in_possession",
        5: "jump_shot",
        6: "layup_dunk",
        7: "player_shot_block"
    }
    
    for i in range(len(player_detections)):
        bbox = player_detections.xyxy[i].tolist()
        team = int(team_ids[i]) if i < len(team_ids) else None
        number = player_numbers[i]
        class_id = int(player_detections.class_id[i])
        role = class_id_to_name.get(class_id, "player")
        
        output["detections"]["players"].append({
            "bbox": bbox,
            "team": team,
            "number": number,
            "role": role,
            "confidence": float(player_detections.confidence[i])
        })
        
    for i in range(len(ball_detections)):
        output["detections"]["ball"].append({
            "bbox": ball_detections.xyxy[i].tolist(),
            "confidence": float(ball_detections.confidence[i])
        })
        
    for i in range(len(rim_detections)):
        output["detections"]["rim"].append({
            "bbox": rim_detections.xyxy[i].tolist(),
            "confidence": float(rim_detections.confidence[i])
        })
        
    # Landmark mapping logic
    # The model 'basketball-court-detection-2/14' returns 33 points matching the template
    model_mapping = list(range(33))
    
    if len(keypoints) > 0:
        for i in range(min(len(model_mapping), len(keypoints.xy[0]))):
            conf = float(keypoints.confidence[0][i])
            if conf > KEYPOINT_DETECTION_MODEL_CONFIDENCE:
                output["landmarks"].append({
                    "id": i,
                    "target_index": model_mapping[i],
                    "x": float(keypoints.xy[0][i][0]),
                    "y": float(keypoints.xy[0][i][1]),
                    "confidence": conf
                })

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"✓ Results saved to {output_path}")
    
    # 6. Visualization
    annotated_frame = frame.copy()
    if debug_dir or annotator:
        from src.visualization import BasketballAnnotator
        if annotator is None:
            annotator = BasketballAnnotator()
        
        # Prepare labels
        labels = []
        for p in output["detections"]["players"]:
            team_label = f"T{p['team']}" if p['team'] is not None else ""
            num_label = f"#{p['number']}" if p['number'] is not None else ""
            labels.append(f"{team_label} {num_label}".strip())
            
        annotated_frame = annotator.annotate_frame(frame, player_detections, labels)
        
        # Annotate ball and rim if any
        if len(ball_detections) > 0:
            annotated_frame = annotator.box_annotator.annotate(annotated_frame, ball_detections)
        if len(rim_detections) > 0:
            annotated_frame = annotator.box_annotator.annotate(annotated_frame, rim_detections)
            
        # 7. Court Transformation & Overlay (Same logic as pipeline.py)
        if len(keypoints) > 0:
            vertices_subset = np.array(annotator.court_config.vertices)
            source_indices = []
            target_indices = []
            
            for i in range(min(len(model_mapping), len(keypoints.xy[0]))):
                if keypoints.confidence[0][i] > KEYPOINT_DETECTION_MODEL_CONFIDENCE:
                    source_indices.append(i)
                    target_indices.append(model_mapping[i])

            if len(source_indices) >= 4:
                source = keypoints.xy[0][source_indices].astype(np.float32)
                target = vertices_subset[target_indices].astype(np.float32)

                m, inliers = cv2.findHomography(source, target, cv2.RANSAC, 5.0)

                if m is not None:
                    # Filter points to only show those within court boundaries
                    points = player_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                    reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
                    transformed_points = cv2.perspectiveTransform(reshaped_points, m)

                    if transformed_points is not None:
                        transformed_points = transformed_points.reshape(-1, 2)
                        max_x, max_y = vertices_subset.max(axis=0)
                        min_x, min_y = vertices_subset.min(axis=0)

                        player_colors = []
                        for tid in team_ids:
                            if tid == 0:
                                player_colors.append(sv.Color.WHITE)
                            elif tid == 1:
                                player_colors.append(sv.Color.BLACK)
                            else:
                                player_colors.append(sv.Color.GREY)

                        margin = 50
                        court_mask = (transformed_points[:, 0] >= min_x - margin) & (transformed_points[:, 0] <= max_x + margin)
                        court_mask &= (transformed_points[:, 1] >= min_y - margin) & (transformed_points[:, 1] <= max_y + margin)

                        active_points = transformed_points[court_mask]
                        active_colors = [player_colors[i] for i, m in enumerate(court_mask) if m]

                        if len(active_points) > 0:
                            court_image = annotator.draw_court_overlay(active_points, colors=active_colors)
                            annotated_frame = annotator.overlay_court(annotated_frame, court_image)
            
            # Optionally still annotate keypoints but with the higher threshold
            annotated_frame = annotator.annotate_keypoints(annotated_frame, keypoints)
            
        if debug_dir:
            image_stem = Path(image_path).stem
            output_image_path = os.path.join(debug_dir, f"{image_stem}_annotated.jpg")
            cv2.imwrite(output_image_path, annotated_frame)
            print(f"✓ Annotated image saved to {output_image_path}")
    
    return output, annotated_frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a single basketball image and output JSON")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output.json", help="Path to output JSON")
    parser.add_argument("--debug-dir", type=str, default="out", help="Directory for debug output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.debug_dir):
        os.makedirs(args.debug_dir)
        
    res = analyze_image(args.image, args.output, args.debug_dir)
    if res:
        print(f"Analysis complete. Found {len(res['detections']['players'])} players, {len(res['detections']['ball'])} balls, and {len(res['landmarks'])} landmarks.")
