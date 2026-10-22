import cv2
import json
import argparse
import os
import time
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import supervision as sv
from pathlib import Path

# Add src to path if needed, but we expect to run as 'python -m src.analyze_image'
try:
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
    from src.workflow_adapter import WorkflowAdapter, read_image_robust
    from src.court_homography import CourtHomographyCalculator
    from src.sports_config import resolve_sport, SportConfig
except ImportError:
    # Fallback for direct script execution if not as module
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
    from workflow_adapter import WorkflowAdapter, read_image_robust
    from court_homography import CourtHomographyCalculator
    from sports_config import resolve_sport, SportConfig

def analyze_image(image_path, output_path=None, debug_dir="out", models=None, annotator=None, debug=False):
    try:
        from src.models import BasketballModels
        from src.pipeline import get_masked_crops
    except ImportError:
        from models import BasketballModels
        from pipeline import get_masked_crops

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

    # 5. Court Transformation & Homography
    print("Detecting court keypoints...")
    from src.config import KEYPOINT_DETECTION_MODEL_CONFIDENCE
    court_results = models.court_model.infer(
        frame, 
        confidence=KEYPOINT_DETECTION_MODEL_CONFIDENCE
    )[0]
    keypoints = sv.KeyPoints.from_inference(court_results)
    
    court_mask = np.ones(len(player_detections), dtype=bool) # Default to True
    transformed_points = None
    model_mapping = list(range(33))
    
    if annotator is None:
        from src.visualization import BasketballAnnotator
        annotator = BasketballAnnotator()

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
                points = player_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
                transformed_points = cv2.perspectiveTransform(reshaped_points, m)

                if transformed_points is not None:
                    transformed_points = transformed_points.reshape(-1, 2)
                    max_x, max_y = vertices_subset.max(axis=0)
                    min_x, min_y = vertices_subset.min(axis=0)
                    margin = 50
                    court_mask = (transformed_points[:, 0] >= min_x - margin) & (transformed_points[:, 0] <= max_x + margin)
                    court_mask &= (transformed_points[:, 1] >= min_y - margin) & (transformed_points[:, 1] <= max_y + margin)

    # Refine teams: only 5 players per team max, prioritize on-court players
    if len(player_detections) > 0:
        team_counts = {0: 0, 1: 0}
        conf_indices = np.lexsort((-player_detections.confidence, -court_mask.astype(int)))
        refined_teams = [None] * len(player_detections)
        for idx in conf_indices:
            team = team_ids[idx]
            try:
                if team is not None:
                    team_int = int(team)
                    if team_counts.get(team_int, 0) < 5:
                        refined_teams[idx] = team_int
                        team_counts[team_int] += 1
            except (ValueError, TypeError):
                continue
        team_ids = refined_teams

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
        team = int(team_ids[i]) if i < len(team_ids) and team_ids[i] is not None else None
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
            
        # 7. Court Overlay
        if transformed_points is not None and len(transformed_points) > 0:
            player_colors = []
            for tid in team_ids:
                if tid == 0:
                    player_colors.append(sv.Color.WHITE)
                elif tid == 1:
                    player_colors.append(sv.Color.BLACK)
                else:
                    player_colors.append(sv.Color.GREY)

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

def analyze_image_workflow(
    image_path: str,
    output_path: Optional[str] = "output.json",
    debug_dir: str = "out",
    workflow_json: Optional[str] = None,
    workflow_id: str = "gemini-3-8-flash-object-detection",
    workspace_name: str = "wpt3dbasket",
    backend: str = "gemini-direct",
    sport: str = "auto",
    mode: str = "high",
    model_name: str = "gemini-3.6-flash",
    enable_vlm_priors: bool = False,
    annotator=None,
    court_model=None
):
    """
    Analyzes a sports image (basketball, soccer, handball, usfootball) using Roboflow VLM / Foundation Workflows
    or Google Gemini Direct API in high mode.
    When enable_vlm_priors=True (--vlm-priors), requests enriched semantic/landmark priors at Step 2
    and validates them via RANSAC homography (Step 4) and Bayesian team/role consensus (Step 5).
    """
    if not enable_vlm_priors and os.getenv("T3D_ENABLE_VLM_PRIORS", "").lower() in ("1", "true", "yes"):
        enable_vlm_priors = True

    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return None, None

    try:
        frame = read_image_robust(image_path)
    except Exception as e:
        print(f"Error: Failed to load image {image_path}: {e}")
        return None, None

    frame_h, frame_w, _ = frame.shape
    start_time = time.perf_counter()

    # Resolve sport
    if sport == "auto" or not sport:
        sport_cfg = resolve_sport(image_path)
    else:
        sport_cfg = resolve_sport(sport)
    print(f"Analyzing {sport_cfg.display_name} image (sport={sport_cfg.name}, mode={mode}, backend={backend}, model={model_name}, vlm_priors={enable_vlm_priors})...")

    adapter = WorkflowAdapter(workspace_name=workspace_name, workflow_id=workflow_id)

    # 1. Fetch or load detections (VLM Inference)
    t_vlm_start = time.perf_counter()
    prev_vlm_time = None
    prev_token_usage = None
    if workflow_json and os.path.exists(workflow_json):
        print(f"Loading pre-fetched workflow output from {workflow_json}...")
        with open(workflow_json, "r") as f:
            raw_workflow = json.load(f)
        if isinstance(raw_workflow, dict):
            prev_vlm_time = raw_workflow.get("performance", {}).get("vlm_inference_seconds")
            prev_token_usage = raw_workflow.get("token_usage")
    elif backend == "openrouter" or ("/" in model_name and adapter.openrouter_api_key):
        print(f"Running OpenRouter Unified API ({model_name}, mode={mode}, sport={sport_cfg.name}, vlm_priors={enable_vlm_priors})...")
        raw_workflow = adapter.call_openrouter(
            frame, sport=sport_cfg.name, mode=mode, model_name=model_name, enable_vlm_priors=enable_vlm_priors
        )
    elif backend == "gemini-direct":
        print(f"Running Google Gemini Direct API ({model_name}, mode={mode}, sport={sport_cfg.name}, vlm_priors={enable_vlm_priors})...")
        raw_workflow = adapter.call_gemini_direct(
            frame, sport=sport_cfg.name, mode=mode, model_name=model_name, enable_vlm_priors=enable_vlm_priors
        )
    elif backend == "openai-direct":
        print(f"Running OpenAI Direct API ({model_name}) for object detection...")
        raw_workflow = adapter.call_openai_direct(frame, model_name=model_name)
    elif backend == "gpt61-workflow":
        print(f"Running Roboflow Serverless GPT-6.1 Sol Workflow...")
        raw_workflow = adapter.call_gpt61_workflow(frame)
    else:
        print(f"Running Roboflow Serverless Workflow ({workspace_name}/{workflow_id})...")
        raw_workflow = adapter.call_workflow(frame)
    vlm_time = time.perf_counter() - t_vlm_start
    if prev_vlm_time is not None and float(prev_vlm_time) > 0.05:
        vlm_time = float(prev_vlm_time)

    # 2. Parse detections into standard categories
    parsed_dets = adapter.parse_predictions(raw_workflow, image_shape=(frame_h, frame_w))
    player_detections = parsed_dets["players"]
    goalkeeper_detections = parsed_dets.get("goalkeepers", sv.Detections.empty())
    referee_detections = parsed_dets["referees"]
    ball_detections = parsed_dets["ball"]
    rim_detections = parsed_dets["basket"]
    field_marks_detections = parsed_dets["field_marks"]

    print(f"Workflow detections: {len(player_detections)} players, {len(goalkeeper_detections)} goalkeepers, "
          f"{len(referee_detections)} referees, {len(ball_detections)} ball(s), "
          f"{len(rim_detections)} goal/basket(s), {len(field_marks_detections)} field mark(s)")

    # 3. Court Homography & Field Marks (works even if corners are not visible!)
    t_homo_start = time.perf_counter()
    calc = CourtHomographyCalculator(sport=sport_cfg.name, min_keypoints=4)
    landmarks = []
    keypoints = None
    matrix = None
    inliers = 0
    homography_source = "perspective_fallback"

    if sport_cfg.keypoint_model_id:
        if court_model is None or getattr(court_model, "model_id", None) != sport_cfg.keypoint_model_id:
            try:
                from inference import get_model
                court_model = get_model(sport_cfg.keypoint_model_id)
            except Exception as e:
                court_model = None

        if court_model is not None:
            try:
                court_res = court_model.infer(frame, confidence=0.3)[0]
                keypoints = sv.KeyPoints.from_inference(court_res)
                class_names = None
                if sport_cfg.name == "foot" and hasattr(court_res, "predictions") and len(court_res.predictions) > 0:
                    pred0 = court_res.predictions[0]
                    if hasattr(pred0, "keypoints"):
                        class_names = [kp.class_name for kp in pred0.keypoints]

                matrix, inliers = calc.compute_from_keypoints(
                    keypoints.xy[0], keypoints.confidence[0], class_names=class_names, confidence_threshold=0.3
                )
                if matrix is not None:
                    homography_source = "onnx_keypoints_ransac"
                for i in range(len(keypoints.xy[0])):
                    conf = float(keypoints.confidence[0][i])
                    if conf >= 0.3:
                        name_str = class_names[i] if class_names and i < len(class_names) else str(i)
                        landmarks.append({
                            "id": i,
                            "label": name_str,
                            "x": float(keypoints.xy[0][i][0]),
                            "y": float(keypoints.xy[0][i][1]),
                            "confidence": conf,
                            "source": "onnx_model"
                        })
            except Exception as e:
                print(f"Court keypoint inference notice ({sport_cfg.name}): {e}")

    feet_pts = (
        calc.compute_ground_contact_points(player_detections, (frame_h, frame_w), enable_vlm_priors=enable_vlm_priors)
        if len(player_detections) > 0 else None
    )
    gk_feet_pts = (
        calc.compute_ground_contact_points(goalkeeper_detections, (frame_h, frame_w), enable_vlm_priors=enable_vlm_priors)
        if len(goalkeeper_detections) > 0 else None
    )

    # Hybrid Step 4: VLM Landmark Priors + RANSAC Validation when --vlm-priors is enabled
    vlm_cal = {"visible_court_side": "full", "keypoints": []}
    if enable_vlm_priors:
        vlm_cal = adapter.parse_court_calibration(raw_workflow, image_shape=(frame_h, frame_w))
        vlm_kps = vlm_cal.get("keypoints", [])
        if vlm_kps or len(rim_detections) > 0:
            vlm_mat, vlm_inliers, matched_vlm_lms = calc.compute_from_vlm_landmarks(
                vlm_kps,
                test_points=feet_pts,
                goal_bboxes=rim_detections.xyxy if len(rim_detections) > 0 else None,
                image_size=(frame_w, frame_h)
            )
            if matrix is None and vlm_mat is not None:
                matrix = vlm_mat
                inliers = vlm_inliers
                homography_source = "vlm_landmarks_ransac"
                print(f"  ✓ [VLM-Priors] Computed validated RANSAC homography from {vlm_inliers} VLM landmarks ({sport_cfg.name})")
            landmarks.extend(matched_vlm_lms)

    # Transform player positions if homography computed, or fall back to robust perspective projection
    transformed_player_pts = None
    court_mask = np.ones(len(player_detections), dtype=bool)
    if len(player_detections) > 0 and feet_pts is not None:
        if matrix is not None:
            transformed_player_pts = calc.transform_points(feet_pts)
            if transformed_player_pts is not None:
                transformed_player_pts = calc.regularize_projected_players(
                    transformed_player_pts,
                    player_detections=player_detections,
                    ground_pts_px=feet_pts,
                    landmarks=landmarks,
                    frame_shape=(frame_h, frame_w),
                    visible_court_side=vlm_cal.get("visible_court_side", "full"),
                    enable_vlm_priors=enable_vlm_priors
                )
                base_margin = 85.0 if homography_source == "vlm_landmarks_ransac" else 0.0
                court_mask = calc.is_on_court(transformed_player_pts, margin=base_margin)
        else:
            transformed_player_pts = calc.estimate_perspective_projection(feet_pts, (frame_h, frame_w))
            court_mask = np.ones(len(player_detections), dtype=bool)

    # Transform goalkeeper positions onto the 2D pitch/court (preserving goal-area interior)
    transformed_gk_pts = None
    if len(goalkeeper_detections) > 0 and gk_feet_pts is not None:
        if matrix is not None:
            transformed_gk_pts = calc.transform_points(gk_feet_pts)
            if transformed_gk_pts is not None:
                transformed_gk_pts = np.array(transformed_gk_pts, dtype=np.float32)
                vis_side = str(vlm_cal.get("visible_court_side", "full")).lower()
                for gi in range(len(transformed_gk_pts)):
                    gx, gy = float(transformed_gk_pts[gi, 0]), float(transformed_gk_pts[gi, 1])
                    if vis_side == "left":
                        gx = float(np.clip(gx, 25.0, calc.court_length * 0.48))
                    elif vis_side == "right":
                        gx = float(np.clip(gx, calc.court_length * 0.52, calc.court_length - 25.0))
                    else:
                        gx = float(np.clip(gx, 25.0, calc.court_length - 25.0))
                    gy = float(np.clip(gy, 25.0, calc.court_width - 25.0))
                    transformed_gk_pts[gi] = [gx, gy]
        else:
            transformed_gk_pts = calc.estimate_perspective_projection(gk_feet_pts, (frame_h, frame_w))
    homo_time = time.perf_counter() - t_homo_start

    # 4. Classify Teams for Players based on dominant jersey chromaticity + optional Bayesian VLM consensus
    t_team_start = time.perf_counter()
    team_ids = adapter.classify_player_teams(
        frame, player_detections, court_mask=court_mask, enable_vlm_priors=enable_vlm_priors
    )

    active_court_mask = np.copy(court_mask)
    role_statuses = (
        list(player_detections.data.get("vlm_role_status", []))
        if (enable_vlm_priors and player_detections.data and "vlm_role_status" in player_detections.data)
        else ["active_on_court"] * len(player_detections)
    )
    jersey_numbers = (
        list(player_detections.data.get("jersey_number", []))
        if (enable_vlm_priors and player_detections.data and "jersey_number" in player_detections.data)
        else [""] * len(player_detections)
    )
    vlm_teams_raw = (
        list(player_detections.data.get("vlm_team", []))
        if (enable_vlm_priors and player_detections.data and "vlm_team" in player_detections.data)
        else [-1] * len(player_detections)
    )
    action_postures = (
        list(player_detections.data.get("action_posture", []))
        if (enable_vlm_priors and player_detections.data and "action_posture" in player_detections.data)
        else ["standing_running"] * len(player_detections)
    )
    ground_statuses = (
        list(player_detections.data.get("ground_contact_status", []))
        if (enable_vlm_priors and player_detections.data and "ground_contact_status" in player_detections.data)
        else ["visible_on_floor"] * len(player_detections)
    )
    ground_points_norm = (
        list(player_detections.data.get("ground_point_norm", []))
        if (enable_vlm_priors and player_detections.data and "ground_point_norm" in player_detections.data)
        else [None] * len(player_detections)
    )
    court_zone_hints = (
        list(player_detections.data.get("court_zone_hint", []))
        if (enable_vlm_priors and player_detections.data and "court_zone_hint" in player_detections.data)
        else [""] * len(player_detections)
    )

    if enable_vlm_priors and len(player_detections) > 0:
        inbound_mask = (
            calc.is_on_court(transformed_player_pts, margin=150.0)
            if transformed_player_pts is not None else np.ones(len(player_detections), dtype=bool)
        )
        for i, status in enumerate(role_statuses):
            if status == "bench_off_court":
                active_court_mask[i] = False
            elif status == "active_inbound" and inbound_mask[i]:
                active_court_mask[i] = True

    # Enforce basketball rule: at most 5 active players on court per team.
    if sport_cfg.name == "basket" and transformed_player_pts is not None and len(transformed_player_pts) > 0:
        for t in [0, 1]:
            t_indices = [i for i in range(len(player_detections)) if team_ids[i] == t and active_court_mask[i]]
            if len(t_indices) > 5:
                def dist_to_edge(idx):
                    # Protect active inbound passers identified by VLM priors from bench demotion
                    if enable_vlm_priors and idx < len(role_statuses) and role_statuses[idx] == "active_inbound":
                        return 9999.0
                    pt = transformed_player_pts[idx]
                    return min(pt[0], 2865.0 - pt[0], pt[1], 1524.0 - pt[1])
                sorted_t = sorted(t_indices, key=dist_to_edge, reverse=True)
                for demoted in sorted_t[5:]:
                    active_court_mask[demoted] = False
    team_time = time.perf_counter() - t_team_start

    if prev_vlm_time is not None and float(prev_vlm_time) > 0.05:
        pipeline_time = vlm_time + homo_time + team_time
    else:
        pipeline_time = time.perf_counter() - start_time

    # Print latency breakdown
    print(f"\n--- Latency Breakdown (Warm Request) ---")
    print(f"  VLM Inference ({backend}): {vlm_time*1000:.1f}ms ({vlm_time:.3f}s)")
    print(f"  Court Homography ({homography_source}): {homo_time*1000:.1f}ms ({homo_time:.3f}s)")
    print(f"  Team Classification:         {team_time*1000:.1f}ms ({team_time:.3f}s)")
    print(f"  Total Warm Pipeline Latency: {pipeline_time*1000:.1f}ms ({pipeline_time:.3f}s)")

    # 5. Build structured JSON output
    output = {
        "sport": sport_cfg.name,
        "mode": mode,
        "vlm_priors_enabled": enable_vlm_priors,
        "image_info": {
            "path": image_path,
            "width": frame_w,
            "height": frame_h
        },
        "performance": {
            "backend": backend,
            "mode": mode,
            "vlm_priors_enabled": enable_vlm_priors,
            "vlm_inference_seconds": round(vlm_time, 4),
            "court_homography_seconds": round(homo_time, 4),
            "team_classification_seconds": round(team_time, 4),
            "warm_pipeline_seconds": round(pipeline_time, 4)
        },
        "homography": {
            "computed": matrix is not None,
            "source": homography_source,
            "visible_court_side": vlm_cal.get("visible_court_side", "full") if enable_vlm_priors else "full",
            "inliers_count": inliers,
            "matrix": matrix.tolist() if matrix is not None else None
        },
        "detections": {
            "players": [],
            "goalkeepers": [],
            "referees": [],
            "ball": [],
            "goal_or_basket": [],
            "field_marks": []
        },
        "landmarks": landmarks
    }

    for i in range(len(player_detections)):
        court_pos = None
        if transformed_player_pts is not None and i < len(transformed_player_pts):
            court_pos = [round(float(coord), 1) for coord in transformed_player_pts[i]]
        ground_px = None
        if feet_pts is not None and i < len(feet_pts):
            ground_px = [round(float(coord), 1) for coord in feet_pts[i]]
        t_val = team_ids[i] if i < len(team_ids) else None
        p_entry = {
            "bbox": player_detections.xyxy[i].tolist(),
            "ground_point_px": ground_px,
            "team": t_val,
            "label": f"T{t_val}" if t_val is not None else "T0",
            "role": "player",
            "confidence": float(player_detections.confidence[i]),
            "court_position_cm": court_pos,
            "is_on_court": bool(active_court_mask[i]) if i < len(active_court_mask) else True
        }
        if enable_vlm_priors:
            p_entry["vlm_team_prior"] = int(vlm_teams_raw[i]) if i < len(vlm_teams_raw) and vlm_teams_raw[i] in (0, 1) else None
            p_entry["vlm_role_status"] = str(role_statuses[i]) if i < len(role_statuses) else "active_on_court"
            p_entry["jersey_number"] = str(jersey_numbers[i]) if i < len(jersey_numbers) and jersey_numbers[i] else None
            p_entry["action_posture"] = str(action_postures[i]) if i < len(action_postures) else "standing_running"
            p_entry["ground_contact_status"] = str(ground_statuses[i]) if i < len(ground_statuses) else "visible_on_floor"
            raw_gpn = ground_points_norm[i] if i < len(ground_points_norm) else None
            p_entry["ground_point_norm"] = [round(float(v), 1) for v in raw_gpn] if isinstance(raw_gpn, (list, tuple, np.ndarray)) and len(raw_gpn) == 2 else None
            p_entry["court_zone_hint"] = str(court_zone_hints[i]) if i < len(court_zone_hints) and court_zone_hints[i] else None
        output["detections"]["players"].append(p_entry)

    # Determine goalkeeper team IDs (T0-GK vs T1-GK) from defensive goal-area geometry + VLM prior alignment
    gk_team_ids: List[int] = []
    gk_vlm_teams = (
        list(goalkeeper_detections.data.get("vlm_team", []))
        if (goalkeeper_detections.data and "vlm_team" in goalkeeper_detections.data)
        else [-1] * len(goalkeeper_detections)
    )
    gk_jersey_numbers = (
        list(goalkeeper_detections.data.get("jersey_number", []))
        if (goalkeeper_detections.data and "jersey_number" in goalkeeper_detections.data)
        else [""] * len(goalkeeper_detections)
    )
    gk_action_postures = (
        list(goalkeeper_detections.data.get("action_posture", []))
        if (goalkeeper_detections.data and "action_posture" in goalkeeper_detections.data)
        else ["defensive_stance"] * len(goalkeeper_detections)
    )
    gk_ground_statuses = (
        list(goalkeeper_detections.data.get("ground_contact_status", []))
        if (goalkeeper_detections.data and "ground_contact_status" in goalkeeper_detections.data)
        else ["visible_on_floor"] * len(goalkeeper_detections)
    )

    # Check if VLM team IDs are inverted relative to final chromatic team_ids
    same_polarity_votes = 0
    inv_polarity_votes = 0
    for idx_p, vt_p in enumerate(vlm_teams_raw):
        if vt_p in (0, 1) and idx_p < len(team_ids) and team_ids[idx_p] in (0, 1):
            if int(vt_p) == int(team_ids[idx_p]):
                same_polarity_votes += 1
            else:
                inv_polarity_votes += 1
    invert_vlm_polarity = inv_polarity_votes > same_polarity_votes

    for i in range(len(goalkeeper_detections)):
        assigned_t = None
        if transformed_gk_pts is not None and i < len(transformed_gk_pts) and transformed_player_pts is not None and len(transformed_player_pts) > 0:
            gx, gy = float(transformed_gk_pts[i][0]), float(transformed_gk_pts[i][1])
            goal_pt = np.array([0.0, calc.court_width * 0.5] if gx < (calc.court_length * 0.5) else [calc.court_length, calc.court_width * 0.5])
            team_dists = {}
            for cand_t in (0, 1):
                cand_pts = [
                    transformed_player_pts[p_idx]
                    for p_idx in range(len(player_detections))
                    if p_idx < len(team_ids) and team_ids[p_idx] == cand_t and active_court_mask[p_idx]
                ]
                if cand_pts:
                    dists = sorted(float(np.linalg.norm(pt - goal_pt)) for pt in cand_pts)
                    top_k = dists[:min(3, len(dists))]
                    team_dists[cand_t] = float(np.mean(top_k))
            if 0 in team_dists and 1 in team_dists:
                assigned_t = 0 if team_dists[0] <= team_dists[1] else 1
            elif 0 in team_dists:
                assigned_t = 0
            elif 1 in team_dists:
                assigned_t = 1

        if assigned_t is None and i < len(gk_vlm_teams) and gk_vlm_teams[i] in (0, 1):
            raw_gt = int(gk_vlm_teams[i])
            assigned_t = (1 - raw_gt) if invert_vlm_polarity else raw_gt

        if assigned_t is None:
            assigned_t = 0 if (gk_feet_pts is not None and gk_feet_pts[i][0] < frame_w * 0.5) else 1
        gk_team_ids.append(int(assigned_t))

    # Enforce physical rule: at most 1 active goalkeeper per half-court / goal
    gk_active_mask = np.ones(len(goalkeeper_detections), dtype=bool)
    if len(goalkeeper_detections) > 1 and transformed_gk_pts is not None and len(transformed_gk_pts) == len(goalkeeper_detections):
        for half_is_left in (True, False):
            half_indices = [
                gi for gi in range(len(goalkeeper_detections))
                if (transformed_gk_pts[gi][0] < calc.court_length * 0.5) == half_is_left
            ]
            if len(half_indices) > 1:
                goal_center = np.array([0.0 if half_is_left else calc.court_length, calc.court_width * 0.5])
                half_indices_sorted = sorted(
                    half_indices,
                    key=lambda gi: float(np.linalg.norm(transformed_gk_pts[gi] - goal_center))
                )
                for extra_gi in half_indices_sorted[1:]:
                    gk_active_mask[extra_gi] = False

    # If 2 active goalkeepers are on opposite sides of the pitch/court, ensure they have opposite team IDs
    active_gk_indices = [gi for gi in range(len(goalkeeper_detections)) if gk_active_mask[gi]]
    if len(active_gk_indices) == 2 and transformed_gk_pts is not None:
        i0, i1 = active_gk_indices[0], active_gk_indices[1]
        if (transformed_gk_pts[i0][0] < calc.court_length * 0.5) != (transformed_gk_pts[i1][0] < calc.court_length * 0.5):
            if gk_team_ids[i0] == gk_team_ids[i1]:
                gk_team_ids[i1] = 1 - gk_team_ids[i0]

    for i in range(len(goalkeeper_detections)):
        gk_court_pos = None
        if transformed_gk_pts is not None and i < len(transformed_gk_pts):
            gk_court_pos = [round(float(coord), 1) for coord in transformed_gk_pts[i]]
        gk_ground_px = None
        if gk_feet_pts is not None and i < len(gk_feet_pts):
            gk_ground_px = [round(float(coord), 1) for coord in gk_feet_pts[i]]
        gk_t = gk_team_ids[i] if i < len(gk_team_ids) else 0
        gk_entry = {
            "bbox": goalkeeper_detections.xyxy[i].tolist(),
            "ground_point_px": gk_ground_px,
            "team": gk_t,
            "label": f"T{gk_t}-GK",
            "role": "goalkeeper",
            "confidence": float(goalkeeper_detections.confidence[i]),
            "court_position_cm": gk_court_pos,
            "is_on_court": bool(gk_active_mask[i])
        }
        if enable_vlm_priors:
            gk_entry["vlm_team_prior"] = int(gk_vlm_teams[i]) if i < len(gk_vlm_teams) and gk_vlm_teams[i] in (0, 1) else None
            gk_entry["vlm_role_status"] = "goalkeeper"
            gk_entry["jersey_number"] = str(gk_jersey_numbers[i]) if i < len(gk_jersey_numbers) and gk_jersey_numbers[i] else None
            gk_entry["action_posture"] = str(gk_action_postures[i]) if i < len(gk_action_postures) and gk_action_postures[i] else "defensive_stance"
            gk_entry["ground_contact_status"] = str(gk_ground_statuses[i]) if i < len(gk_ground_statuses) and gk_ground_statuses[i] else "visible_on_floor"
        output["detections"]["goalkeepers"].append(gk_entry)

    for i in range(len(referee_detections)):
        output["detections"]["referees"].append({
            "bbox": referee_detections.xyxy[i].tolist(),
            "confidence": float(referee_detections.confidence[i])
        })

    for i in range(len(ball_detections)):
        output["detections"]["ball"].append({
            "bbox": ball_detections.xyxy[i].tolist(),
            "confidence": float(ball_detections.confidence[i])
        })

    for i in range(len(rim_detections)):
        output["detections"]["goal_or_basket"].append({
            "bbox": rim_detections.xyxy[i].tolist(),
            "confidence": float(rim_detections.confidence[i])
        })

    for i in range(len(field_marks_detections)):
        output["detections"]["field_marks"].append({
            "bbox": field_marks_detections.xyxy[i].tolist(),
            "confidence": float(field_marks_detections.confidence[i])
        })

    if "usage" in raw_workflow:
        output["token_usage"] = raw_workflow["usage"]
    elif prev_token_usage:
        output["token_usage"] = prev_token_usage

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"✓ Results saved to {output_path}")

    # 6. Visualization & 2D radar overlay
    annotated_frame = frame.copy()
    if debug_dir:
        from src.visualization import SportsAnnotator
        if annotator is None or getattr(annotator, "sport", None) != sport_cfg.name:
            annotator = SportsAnnotator(sport=sport_cfg.name)

        # Team colors: Cyan/Blue for T0, Pink/Magenta or Dark for T1
        team_palette = sv.ColorPalette([sv.Color.from_hex("#00BFFF"), sv.Color.from_hex("#FF1493")])
        team_box_annotator = sv.BoxAnnotator(color=team_palette, thickness=3)
        team_label_annotator = sv.LabelAnnotator(color=team_palette, text_color=sv.Color.WHITE)

        player_dets_colored = sv.Detections(
            xyxy=player_detections.xyxy,
            confidence=player_detections.confidence,
            class_id=np.array([t if t is not None else 0 for t in team_ids], dtype=int)
        )
        labels = [
            f"T{t}" if active_court_mask[i] else f"T{t} (Bench)"
            for i, t in enumerate(team_ids)
        ]
        annotated_frame = team_box_annotator.annotate(annotated_frame, player_dets_colored)
        annotated_frame = team_label_annotator.annotate(annotated_frame, player_dets_colored, labels)

        # Goalkeepers (labeled T0-GK / T1-GK)
        if len(goalkeeper_detections) > 0:
            gk_palette = sv.ColorPalette([sv.Color.from_hex("#39FF14")])
            gk_box_annotator = sv.BoxAnnotator(color=gk_palette, thickness=3)
            gk_label_annotator = sv.LabelAnnotator(color=gk_palette, text_color=sv.Color.BLACK)
            gk_labels = [
                f"T{gk_team_ids[i]}-GK" if i < len(gk_team_ids) else "T0-GK"
                for i in range(len(goalkeeper_detections))
            ]
            annotated_frame = gk_box_annotator.annotate(annotated_frame, goalkeeper_detections)
            annotated_frame = gk_label_annotator.annotate(annotated_frame, goalkeeper_detections, gk_labels)

        # Referees
        if len(referee_detections) > 0:
            ref_labels = ["REF"] * len(referee_detections)
            annotated_frame = annotator.box_annotator.annotate(annotated_frame, referee_detections)
            annotated_frame = annotator.label_annotator.annotate(annotated_frame, referee_detections, ref_labels)

        # Ball and goal/basket
        if len(ball_detections) > 0:
            ball_palette = sv.ColorPalette([sv.Color.from_hex("#FF4500")])
            ball_box_annotator = sv.BoxAnnotator(color=ball_palette, thickness=3)
            ball_label_annotator = sv.LabelAnnotator(color=ball_palette, text_color=sv.Color.WHITE)
            annotated_frame = ball_box_annotator.annotate(annotated_frame, ball_detections)
            annotated_frame = ball_label_annotator.annotate(annotated_frame, ball_detections, ["BALL"] * len(ball_detections))

        if len(rim_detections) > 0:
            goal_label = "BASKET" if sport_cfg.name == "basket" else ("GOALPOST" if sport_cfg.name == "usfootball" else "GOAL")
            goal_palette = sv.ColorPalette([sv.Color.from_hex("#00FF7F")])
            goal_box_annotator = sv.BoxAnnotator(color=goal_palette, thickness=3)
            goal_label_annotator = sv.LabelAnnotator(color=goal_palette, text_color=sv.Color.BLACK)
            annotated_frame = goal_box_annotator.annotate(annotated_frame, rim_detections)
            annotated_frame = goal_label_annotator.annotate(annotated_frame, rim_detections, [goal_label] * len(rim_detections))

        # 2D Court / Pitch Radar Overlay for ALL sports (including Goalkeepers T0-GK / T1-GK)
        court_overlay = None
        has_active_players = transformed_player_pts is not None and len(transformed_player_pts) > 0
        active_gk_pts = (
            transformed_gk_pts[gk_active_mask]
            if (transformed_gk_pts is not None and len(transformed_gk_pts) == len(gk_active_mask))
            else None
        )
        active_gk_teams = [gk_team_ids[i] for i in range(len(gk_team_ids)) if i < len(gk_active_mask) and gk_active_mask[i]]
        has_active_gks = active_gk_pts is not None and len(active_gk_pts) > 0
        if has_active_players or has_active_gks:
            active_pts = np.empty((0, 2), dtype=np.float32)
            active_colors = []
            if has_active_players:
                colors = [sv.Color.from_hex("#00BFFF") if t == 0 else sv.Color.from_hex("#FF1493") for t in team_ids]
                active_pts = transformed_player_pts[active_court_mask]
                active_colors = [colors[i] for i, m in enumerate(active_court_mask) if m]
            if len(active_pts) > 0 or has_active_gks:
                court_overlay = annotator.draw_court_overlay(
                    active_pts,
                    colors=active_colors,
                    gk_xy=active_gk_pts if has_active_gks else None,
                    gk_teams=active_gk_teams if has_active_gks else None
                )
                annotated_frame = annotator.overlay_court(annotated_frame, court_overlay)

        def _draw_vlm_landmarks(img_canvas: np.ndarray, lm_list: list) -> np.ndarray:
            for lm in lm_list:
                if lm.get("source") in ("vlm_prior", "goal_bbox_anchor"):
                    lx, ly = int(round(lm.get("x", 0))), int(round(lm.get("y", 0)))
                    lbl = str(lm.get("label", "")).replace("right_", "R_").replace("left_", "L_")
                    color_bgr = (0, 255, 255) if lm.get("source") == "vlm_prior" else (0, 255, 128)
                    cv2.circle(img_canvas, (lx, ly), 7, (0, 0, 0), -1)
                    cv2.circle(img_canvas, (lx, ly), 5, color_bgr, -1)
                    cv2.putText(img_canvas, lbl, (lx + 8, ly - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img_canvas, lbl, (lx + 8, ly - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_bgr, 1, cv2.LINE_AA)
            return img_canvas

        # Keypoints overlay
        if keypoints is not None:
            annotated_frame = annotator.annotate_keypoints(annotated_frame, keypoints)
        if landmarks:
            annotated_frame = _draw_vlm_landmarks(annotated_frame, landmarks)

        os.makedirs(debug_dir, exist_ok=True)
        stem = Path(image_path).stem
        out_img = os.path.join(debug_dir, f"{stem}_workflow_annotated.jpg")
        cv2.imwrite(out_img, annotated_frame)
        print(f"✓ Annotated image saved to {out_img}")

        # Side-by-side composite view: [Annotated frame with bboxes] | [2D Pitch/Court Radar]
        try:
            if court_overlay is not None:
                fh, fw = frame.shape[:2]
                ch, cw = court_overlay.shape[:2]
                scale = fh / float(ch)
                court_resized = cv2.resize(court_overlay, (int(cw * scale), fh))
                
                # Left side: complete annotated frame with bboxes and labels
                left_annotated = team_box_annotator.annotate(frame.copy(), player_dets_colored)
                left_annotated = team_label_annotator.annotate(left_annotated, player_dets_colored, labels)
                if len(goalkeeper_detections) > 0:
                    left_annotated = gk_box_annotator.annotate(left_annotated, goalkeeper_detections)
                    left_annotated = gk_label_annotator.annotate(left_annotated, goalkeeper_detections, gk_labels)
                if len(referee_detections) > 0:
                    left_annotated = annotator.box_annotator.annotate(left_annotated, referee_detections)
                    left_annotated = annotator.label_annotator.annotate(left_annotated, referee_detections, ref_labels)
                if len(ball_detections) > 0:
                    left_annotated = ball_box_annotator.annotate(left_annotated, ball_detections)
                    left_annotated = ball_label_annotator.annotate(left_annotated, ball_detections, ["BALL"] * len(ball_detections))
                if len(rim_detections) > 0:
                    left_annotated = goal_box_annotator.annotate(left_annotated, rim_detections)
                    left_annotated = goal_label_annotator.annotate(left_annotated, rim_detections, [goal_label] * len(rim_detections))
                if keypoints is not None:
                    left_annotated = annotator.annotate_keypoints(left_annotated, keypoints)
                if landmarks:
                    left_annotated = _draw_vlm_landmarks(left_annotated, landmarks)
                side_by_side = np.hstack([left_annotated, court_resized])
            else:
                side_by_side = np.hstack([frame, annotated_frame])

            sh, sw = side_by_side.shape[:2]
            if sw > 2560:
                scale_w = 2560.0 / sw
                side_by_side = cv2.resize(side_by_side, (2560, int(sh * scale_w)))
            sbs_img = os.path.join(debug_dir, f"{stem}_side_by_side.jpg")
            cv2.imwrite(sbs_img, side_by_side)
            print(f"✓ Side-by-side image saved to {sbs_img}")
        except Exception as e:
            print(f"Notice: side_by_side generation: {e}")

    return output, annotated_frame


def render_workflow_visualizations(
    frame: np.ndarray,
    output: Dict[str, Any],
    annotator=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render both the annotated frame (with top-left 2D radar overlay) and the
    side-by-side frame ([annotated frame | full-height 2D radar]) in memory
    directly from an image `frame` and the structured `output` dictionary
    produced by `analyze_image_workflow`.
    """
    from src.visualization import SportsAnnotator

    sport_name = resolve_sport(output.get("sport", "basket")).name
    if annotator is None or getattr(annotator, "sport", None) != sport_name:
        annotator = SportsAnnotator(sport=sport_name)

    dets_dict = output.get("detections", {})
    players_list = dets_dict.get("players", [])
    gks_list = dets_dict.get("goalkeepers", [])
    refs_list = dets_dict.get("referees", [])
    balls_list = dets_dict.get("ball", [])
    goals_list = dets_dict.get("goal_or_basket", [])
    landmarks = output.get("landmarks", [])

    team_palette = sv.ColorPalette([sv.Color.from_hex("#00BFFF"), sv.Color.from_hex("#FF1493")])
    team_box_annotator = sv.BoxAnnotator(color=team_palette, thickness=3)
    team_label_annotator = sv.LabelAnnotator(color=team_palette, text_color=sv.Color.WHITE)

    def _annotate_base(canvas: np.ndarray) -> np.ndarray:
        out_canvas = canvas.copy()
        # 1. Players
        if players_list:
            p_xyxy = np.array([p["bbox"] for p in players_list], dtype=np.float32)
            p_conf = np.array([p.get("confidence", 0.9) for p in players_list], dtype=np.float32)
            p_teams = [p.get("team", 0) if p.get("team") in (0, 1) else 0 for p in players_list]
            p_dets = sv.Detections(
                xyxy=p_xyxy,
                confidence=p_conf,
                class_id=np.array(p_teams, dtype=int),
            )
            p_labels = [
                f"T{p_teams[i]}" if p.get("is_on_court", True) else f"T{p_teams[i]} (Bench)"
                for i, p in enumerate(players_list)
            ]
            out_canvas = team_box_annotator.annotate(out_canvas, p_dets)
            out_canvas = team_label_annotator.annotate(out_canvas, p_dets, p_labels)

        # 2. Goalkeepers (T0-GK / T1-GK)
        if gks_list:
            gk_xyxy = np.array([g["bbox"] for g in gks_list], dtype=np.float32)
            gk_conf = np.array([g.get("confidence", 0.9) for g in gks_list], dtype=np.float32)
            gk_dets = sv.Detections(
                xyxy=gk_xyxy,
                confidence=gk_conf,
                class_id=np.zeros(len(gks_list), dtype=int),
            )
            gk_palette = sv.ColorPalette([sv.Color.from_hex("#39FF14")])
            gk_box_annotator = sv.BoxAnnotator(color=gk_palette, thickness=3)
            gk_label_annotator = sv.LabelAnnotator(color=gk_palette, text_color=sv.Color.BLACK)
            gk_labels = [g.get("label", f"T{g.get('team', 0)}-GK") for g in gks_list]
            out_canvas = gk_box_annotator.annotate(out_canvas, gk_dets)
            out_canvas = gk_label_annotator.annotate(out_canvas, gk_dets, gk_labels)

        # 3. Referees
        if refs_list:
            r_xyxy = np.array([r["bbox"] for r in refs_list], dtype=np.float32)
            r_conf = np.array([r.get("confidence", 0.9) for r in refs_list], dtype=np.float32)
            r_dets = sv.Detections(
                xyxy=r_xyxy,
                confidence=r_conf,
                class_id=np.zeros(len(refs_list), dtype=int),
            )
            out_canvas = annotator.box_annotator.annotate(out_canvas, r_dets)
            out_canvas = annotator.label_annotator.annotate(out_canvas, r_dets, ["REF"] * len(refs_list))

        # 4. Ball
        if balls_list:
            b_xyxy = np.array([b["bbox"] for b in balls_list], dtype=np.float32)
            b_conf = np.array([b.get("confidence", 0.9) for b in balls_list], dtype=np.float32)
            b_dets = sv.Detections(
                xyxy=b_xyxy,
                confidence=b_conf,
                class_id=np.zeros(len(balls_list), dtype=int),
            )
            ball_palette = sv.ColorPalette([sv.Color.from_hex("#FF4500")])
            ball_box_annotator = sv.BoxAnnotator(color=ball_palette, thickness=3)
            ball_label_annotator = sv.LabelAnnotator(color=ball_palette, text_color=sv.Color.WHITE)
            out_canvas = ball_box_annotator.annotate(out_canvas, b_dets)
            out_canvas = ball_label_annotator.annotate(out_canvas, b_dets, ["BALL"] * len(balls_list))

        # 5. Goal / Basket / Goalpost
        if goals_list:
            g_xyxy = np.array([g["bbox"] for g in goals_list], dtype=np.float32)
            g_conf = np.array([g.get("confidence", 0.9) for g in goals_list], dtype=np.float32)
            g_dets = sv.Detections(
                xyxy=g_xyxy,
                confidence=g_conf,
                class_id=np.zeros(len(goals_list), dtype=int),
            )
            goal_label = "BASKET" if sport_name == "basket" else ("GOALPOST" if sport_name == "usfootball" else "GOAL")
            goal_palette = sv.ColorPalette([sv.Color.from_hex("#00FF7F")])
            goal_box_annotator = sv.BoxAnnotator(color=goal_palette, thickness=3)
            goal_label_annotator = sv.LabelAnnotator(color=goal_palette, text_color=sv.Color.BLACK)
            out_canvas = goal_box_annotator.annotate(out_canvas, g_dets)
            out_canvas = goal_label_annotator.annotate(out_canvas, g_dets, [goal_label] * len(goals_list))

        # 6. Landmarks
        for lm in landmarks:
            lx, ly = int(round(lm.get("x", 0))), int(round(lm.get("y", 0)))
            lbl = str(lm.get("label", "")).replace("right_", "R_").replace("left_", "L_")
            src = lm.get("source", "")
            color_bgr = (0, 255, 255) if src == "vlm_prior" else ((0, 255, 128) if src == "goal_bbox_anchor" else (0, 165, 255))
            cv2.circle(out_canvas, (lx, ly), 7, (0, 0, 0), -1)
            cv2.circle(out_canvas, (lx, ly), 5, color_bgr, -1)
            cv2.putText(out_canvas, lbl, (lx + 8, ly - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out_canvas, lbl, (lx + 8, ly - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_bgr, 1, cv2.LINE_AA)

        return out_canvas

    left_annotated = _annotate_base(frame)
    annotated_frame = left_annotated.copy()

    # Build 2D court/pitch radar
    active_pts_list = []
    active_colors = []
    for p in players_list:
        pos = p.get("court_position_cm")
        if p.get("is_on_court", True) and isinstance(pos, (list, tuple)) and len(pos) == 2:
            active_pts_list.append(pos)
            t = p.get("team", 0)
            active_colors.append(sv.Color.from_hex("#00BFFF") if t == 0 else sv.Color.from_hex("#FF1493"))

    active_gk_pts_list = []
    active_gk_teams = []
    for g in gks_list:
        pos = g.get("court_position_cm")
        if g.get("is_on_court", True) and isinstance(pos, (list, tuple)) and len(pos) == 2:
            active_gk_pts_list.append(pos)
            active_gk_teams.append(int(g.get("team", 0)))

    court_overlay = None
    if active_pts_list or active_gk_pts_list:
        active_pts_arr = np.array(active_pts_list, dtype=np.float32) if active_pts_list else np.empty((0, 2), dtype=np.float32)
        active_gk_arr = np.array(active_gk_pts_list, dtype=np.float32) if active_gk_pts_list else None
        court_overlay = annotator.draw_court_overlay(
            active_pts_arr,
            colors=active_colors,
            gk_xy=active_gk_arr,
            gk_teams=active_gk_teams if active_gk_teams else None,
        )
        annotated_frame = annotator.overlay_court(annotated_frame, court_overlay)

    if court_overlay is not None:
        fh, fw = frame.shape[:2]
        ch, cw = court_overlay.shape[:2]
        scale = fh / float(ch)
        court_resized = cv2.resize(court_overlay, (int(cw * scale), fh))
        side_by_side = np.hstack([left_annotated, court_resized])
    else:
        side_by_side = np.hstack([frame, annotated_frame])

    sh, sw = side_by_side.shape[:2]
    if sw > 2560:
        scale_w = 2560.0 / sw
        side_by_side = cv2.resize(side_by_side, (2560, int(sh * scale_w)))

    return annotated_frame, side_by_side

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze sports images (basket, foot, handball, usfootball) and output JSON")
    parser.add_argument("--image", type=str, required=False, default=None, help="Path to input image")
    parser.add_argument("--reprocess-dir", type=str, nargs="+", default=None, help="Reprocess all *_out.json files in the given directory/directories locally (0 API cost)")
    parser.add_argument("--output", type=str, default="output.json", help="Path to output JSON")
    parser.add_argument("--debug-dir", type=str, default="out", help="Directory for debug output")
    parser.add_argument("--sport", type=str, default="auto", choices=["auto", "basket", "foot", "handball", "usfootball"], help="Sport: 'auto' (detect from path), 'basket', 'foot', 'handball', 'usfootball'")
    parser.add_argument("--mode", type=str, default="high", choices=["high", "default"], help="Gemini mode: 'high' (MEDIA_RESOLUTION_HIGH + thinkingLevel high) or 'default'")
    parser.add_argument("--workflow", type=str, default=None, help="Roboflow Workflow ID (e.g. gemini-3-8-flash-object-detection)")
    parser.add_argument("--workflow-json", type=str, default=None, help="Path to pre-fetched workflow JSON output file")
    parser.add_argument("--workspace", type=str, default="wpt3dbasket", help="Roboflow workspace name")
    parser.add_argument("--backend", type=str, default="gemini-direct", choices=["gemini-direct", "openrouter", "workflow", "openai-direct", "gpt61-workflow"], help="Backend: 'gemini-direct', 'openrouter', 'workflow', 'openai-direct', or 'gpt61-workflow'")
    parser.add_argument("--model", type=str, default="gemini-3.6-flash", help="VLM model name (default: gemini-3.6-flash, or e.g. google/gemini-2.5-pro with --backend openrouter)")
    parser.add_argument("--vlm-priors", action="store_true", default=False, help="Enable Hybrid VLM Probabilistic Priors (team_id, role_status, court_calibration keypoints)")
    
    args = parser.parse_args()
    
    if args.reprocess_dir:
        from src.visualization import SportsAnnotator
        shared_court_models = {}
        shared_annotators = {}
        total_reprocessed = 0
        for rdir in args.reprocess_dir:
            if not os.path.isdir(rdir):
                continue
            json_files = sorted(Path(rdir).glob("*_out.json"))
            for jf in json_files:
                with open(jf, "r") as f:
                    jdata = json.load(f)
                img_path = jdata.get("image_info", {}).get("path")
                if not img_path or not os.path.exists(img_path):
                    print(f"Skipping {jf}: image_path '{img_path}' not found.")
                    continue
                sp = jdata.get("sport", args.sport)
                md = jdata.get("mode", args.mode)
                vp = bool(jdata.get("vlm_priors_enabled", True))
                bk = jdata.get("performance", {}).get("backend", "openrouter")
                mdl = jdata.get("token_usage", {}).get("model", args.model)
                if "/" in str(mdl):
                    bk = "openrouter"
                sport_cfg_r = resolve_sport(sp)
                if sp not in shared_annotators:
                    shared_annotators[sp] = SportsAnnotator(sport=sport_cfg_r.name)
                if sp not in shared_court_models and sport_cfg_r.keypoint_model_id:
                    try:
                        from inference import get_model
                        shared_court_models[sp] = get_model(sport_cfg_r.keypoint_model_id)
                    except Exception:
                        shared_court_models[sp] = None
                res, _ = analyze_image_workflow(
                    image_path=img_path,
                    output_path=str(jf),
                    debug_dir=str(jf.parent),
                    workflow_json=str(jf),
                    backend=bk,
                    sport=sp,
                    mode=md,
                    model_name=mdl,
                    enable_vlm_priors=vp,
                    court_model=shared_court_models.get(sp),
                    annotator=shared_annotators.get(sp)
                )
                total_reprocessed += 1
        print(f"\n✓ Locally reprocessed {total_reprocessed} JSON/image outputs across {args.reprocess_dir} (0 API cost).")
    else:
        if not args.image:
            parser.error("--image is required unless --reprocess-dir is specified.")

        if not os.path.exists(args.debug_dir):
            os.makedirs(args.debug_dir)

        if args.workflow or args.workflow_json or args.backend in ["gemini-direct", "openrouter", "openai-direct", "gpt61-workflow"]:
            wf_id = args.workflow if args.workflow else "gemini-3-8-flash-object-detection"
            res, _ = analyze_image_workflow(
                image_path=args.image,
                output_path=args.output,
                debug_dir=args.debug_dir,
                workflow_json=args.workflow_json,
                workflow_id=wf_id,
                workspace_name=args.workspace,
                backend=args.backend,
                sport=args.sport,
                mode=args.mode,
                model_name=args.model,
                enable_vlm_priors=args.vlm_priors
            )
        else:
            res, _ = analyze_image(args.image, args.output, args.debug_dir)
            
        if res:
            print(f"Analysis complete. Sport: {res.get('sport')}. Found {len(res['detections']['players'])} players, {len(res['detections']['ball'])} balls, and {len(res['landmarks'])} landmarks.")


