import cv2
import json
import argparse
import os
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
        USE_SAM2
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
        USE_SAM2
    )
    from pipeline import get_masked_crops

def analyze_image(image_path, output_path=None, debug_dir="out"):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return None

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Failed to load image {image_path}")
        return None
    
    frame_h, frame_w, _ = frame.shape
    
    # Initialize models
    print("Loading models...")
    models = BasketballModels()
    
    # 1. Inference - Players and objects
    print("Detecting players and objects...")
    player_results = models.player_model.infer(
        frame, 
        confidence=PLAYER_DETECTION_MODEL_CONFIDENCE, 
        iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD
    )[0]
    all_detections = sv.Detections.from_inference(player_results)
    
    # Filter detections
    player_detections = all_detections[np.isin(all_detections.class_id, PLAYER_CLASS_IDS)]
    ball_detections = all_detections[all_detections.class_id == BALL_CLASS_ID]
    rim_detections = all_detections[all_detections.class_id == RIM_CLASS_ID]
    number_detections = all_detections[all_detections.class_id == NUMBER_CLASS_ID]
    
    # 2. SAM2 Masks (if enabled)
    if USE_SAM2 and len(player_detections) > 0:
        print("Generating SAM2 masks...")
        masks = models.get_masks(frame, player_detections)
        if masks is not None:
            player_detections.mask = masks

    # 3. Team Classification
    print("Classifying teams...")
    if len(player_detections) > 0:
        factor = 1.0 if USE_SAM2 else 0.4
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
    court_results = models.court_model.infer(
        frame, 
        confidence=KEYPOINT_DETECTION_MODEL_CONFIDENCE
    )[0]
    keypoints = sv.KeyPoints.from_inference(court_results)
    
    # Structure output
    output = {
        "image_info": {
            "path": image_path,
            "width": frame_w,
            "height": frame_h
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
        
    if len(keypoints) > 0:
        # Template for keypoint names if available, or just index
        # Based on pipeline.py mapping: model_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17]
        for i in range(len(keypoints.xy[0])):
            conf = float(keypoints.confidence[0][i])
            if conf > KEYPOINT_DETECTION_MODEL_CONFIDENCE:
                output["landmarks"].append({
                    "id": i,
                    "x": float(keypoints.xy[0][i][0]),
                    "y": float(keypoints.xy[0][i][1]),
                    "confidence": conf
                })

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"✓ Results saved to {output_path}")
    
    # 6. Visualization (Debug)
    if debug_dir:
        from src.visualization import BasketballAnnotator
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
            
        # Annotate keypoints
        if len(keypoints) > 0:
            annotated_frame = annotator.annotate_keypoints(annotated_frame, keypoints)
            
        image_stem = Path(image_path).stem
        output_image_path = os.path.join(debug_dir, f"{image_stem}_annotated.jpg")
        cv2.imwrite(output_image_path, annotated_frame)
        print(f"✓ Annotated image saved to {output_image_path}")
    
    return output

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
