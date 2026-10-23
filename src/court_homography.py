import sys
import types
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import supervision as sv

for _mod_name, _attr_name in (("sports.common.team", "TeamClassifier"), ("sports.common.path", "clean_paths")):
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        setattr(_stub, _attr_name, None)
        sys.modules[_mod_name] = _stub

try:
    from sports.basketball import CourtConfiguration, League
except ImportError:
    CourtConfiguration = None
    League = None

try:
    from sports.soccer.annotators import SoccerPitchConfiguration
except ImportError:
    SoccerPitchConfiguration = None


class CourtHomographyCalculator:
    """
    Computes court/pitch homography from partial field marks and keypoints across sports.
    Supports Basketball (NBA/FIBA), Soccer (FIFA/Roboflow), Handball, and American Football.
    """

    def __init__(self, sport: str = "basket", min_keypoints: int = 4, reproj_threshold: float = 15.0):
        self.sport = sport.lower()
        self.min_keypoints = min_keypoints
        self.reproj_threshold = reproj_threshold
        self.homography_matrix: Optional[np.ndarray] = None
        self.inliers: Optional[np.ndarray] = None
        
        if self.sport == "basket":
            if CourtConfiguration is not None and League is not None:
                self.court_config = CourtConfiguration(league=League.NBA)
            else:
                from src.visualization import SportsAnnotator
                self.court_config = SportsAnnotator(sport="basket").court_config
            self.vertices = np.array(self.court_config.vertices, dtype=np.float32)
            self.court_length, self.court_width = 2865.0, 1524.0
            self.labels = None
        elif self.sport == "foot":
            if SoccerPitchConfiguration is not None:
                self.court_config = SoccerPitchConfiguration()
            else:
                from src.visualization import SportsAnnotator
                self.court_config = SportsAnnotator(sport="foot").court_config
            self.vertices = np.array(self.court_config.vertices, dtype=np.float32)
            self.court_length, self.court_width = 12000.0, 7000.0
            self.labels = list(self.court_config.labels)
        elif self.sport == "handball":
            self.court_config = None
            self.court_length, self.court_width = 4000.0, 2000.0
            self.vertices = np.array([
                [0, 0], [4000, 0], [4000, 2000], [0, 2000],
                [2000, 0], [2000, 2000]
            ], dtype=np.float32)
            self.labels = None
        elif self.sport == "usfootball":
            self.court_config = None
            self.court_length, self.court_width = 10973.0, 4877.0
            self.vertices = np.array([
                [0, 0], [10973, 0], [10973, 4877], [0, 4877],
                [5486.5, 0], [5486.5, 4877]
            ], dtype=np.float32)
            self.labels = None
        else:
            self.court_config = None
            self.court_length, self.court_width = 2865.0, 1524.0
            self.vertices = np.array([[0, 0], [2865, 0], [2865, 1524], [0, 1524]], dtype=np.float32)
            self.labels = None

    def compute_from_keypoints(
        self, 
        keypoint_xy: np.ndarray, 
        confidences: np.ndarray, 
        class_names: Optional[List[str]] = None,
        confidence_threshold: float = 0.3
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Computes the homography matrix from detected keypoints.
        Supports both index-based matching (basketball) and label-based matching (soccer).
        """
        source_points = []
        target_points = []
        
        # Soccer label-based matching ('01' -> vertex)
        if self.sport == "foot" and class_names is not None and self.labels is not None:
            label_to_vertex = {lbl: vertex for lbl, vertex in zip(self.labels, self.vertices)}
            for i, name in enumerate(class_names):
                conf = float(confidences[i])
                if conf >= confidence_threshold and name in label_to_vertex:
                    source_points.append(keypoint_xy[i])
                    target_points.append(label_to_vertex[name])
        else:
            num_points = min(len(keypoint_xy), len(self.vertices))
            for i in range(num_points):
                conf = float(confidences[i])
                if conf >= confidence_threshold:
                    source_points.append(keypoint_xy[i])
                    target_points.append(self.vertices[i])
                    
        if len(source_points) < self.min_keypoints:
            self.homography_matrix = None
            return None, 0
            
        src = np.array(source_points, dtype=np.float32)
        dst = np.array(target_points, dtype=np.float32)
        
        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.reproj_threshold)
        
        if matrix is not None:
            inliers_count = int(np.sum(mask)) if mask is not None else 0
            if inliers_count >= self.min_keypoints:
                self.homography_matrix = matrix
                self.inliers = mask
                return matrix, inliers_count
                
        self.homography_matrix = None
        return None, 0

    def _resolve_vlm_landmark_cm(self, name: str) -> Optional[Tuple[float, float]]:
        """
        Resolves a canonical VLM landmark name (or dynamic US Football yard-line label)
        to its exact physical (X_cm, Y_cm) coordinates on the court/pitch.
        """
        if not name:
            return None
        clean = name.strip().lower().replace("-", "_").replace(" ", "_")

        try:
            from src.sports_config import resolve_sport
        except ImportError:
            from sports_config import resolve_sport

        sport_cfg = resolve_sport(self.sport)
        if clean in sport_cfg.canonical_landmarks_cm:
            return sport_cfg.canonical_landmarks_cm[clean]

        # Dynamic parser for American Football yard-line + lateral zone labels
        if self.sport == "usfootball":
            zone_y_map = {
                "far_sideline": 0.0,
                "top_sideline": 0.0,
                "far_numbers": 823.0,
                "top_numbers": 823.0,
                "far_hash": 1829.0,
                "top_hash": 1829.0,
                "center": 2438.5,
                "middle": 2438.5,
                "near_hash": 3048.0,
                "bottom_hash": 3048.0,
                "near_numbers": 4054.0,
                "bottom_numbers": 4054.0,
                "near_sideline": 4877.0,
                "bottom_sideline": 4877.0,
            }
            matched_y = None
            for z_key, z_val in zone_y_map.items():
                if clean.endswith(z_key):
                    matched_y = z_val
                    prefix = clean[: -len(z_key)].rstrip("_")
                    break
            if matched_y is not None:
                import re
                if "goal" in prefix:
                    yard_num = 0.0
                else:
                    m = re.search(r"(\d+)", prefix)
                    yard_num = float(m.group(1)) if m else None
                if yard_num is not None and 0.0 <= yard_num <= 50.0:
                    if yard_num == 50.0:
                        x_cm = 60.0 * 91.44
                    elif "right" in prefix:
                        x_cm = (110.0 - yard_num) * 91.44
                    else:
                        x_cm = (10.0 + yard_num) * 91.44
                    return (round(x_cm, 1), matched_y)

        return None

    def compute_from_vlm_landmarks(
        self,
        vlm_keypoints: List[Dict[str, Any]],
        test_points: Optional[np.ndarray] = None,
        goal_bboxes: Optional[np.ndarray] = None,
        image_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[Optional[np.ndarray], int, List[Dict[str, Any]]]:
        """
        Computes and geometrically validates a RANSAC homography matrix from VLM semantic landmarks.
        Enforces 2D non-collinearity (SVD), horizon positivity (w > 0.15), Jacobian orientation,
        and provides a 6-DoF RANSAC Affine fallback for tight attack-zone views.
        """
        source_points = []
        target_points = []
        matched_landmarks = []
        existing_labels = set()

        for idx, kp in enumerate(vlm_keypoints or []):
            name = str(kp.get("name") or kp.get("label") or "").strip()
            target_cm = self._resolve_vlm_landmark_cm(name)
            if target_cm is None:
                continue
            x_px = kp.get("x")
            y_px = kp.get("y")
            if x_px is None or y_px is None:
                continue
            source_points.append([float(x_px), float(y_px)])
            target_points.append([float(target_cm[0]), float(target_cm[1])])
            existing_labels.add(name.lower())
            matched_landmarks.append({
                "id": idx,
                "label": name,
                "x": float(x_px),
                "y": float(y_px),
                "target_cm": [float(target_cm[0]), float(target_cm[1])],
                "confidence": float(kp.get("confidence", 0.85)),
                "source": "vlm_prior"
            })

        # Supplement with floor contact points of detected goalposts in handball when goal bbox is available
        if self.sport == "handball" and goal_bboxes is not None and len(goal_bboxes) > 0 and image_size is not None:
            fw, fh = image_size
            gx1, gy1, gx2, gy2 = [float(v) for v in goal_bboxes[0]]
            gw = max(1.0, gx2 - gx1)
            gh = max(1.0, gy2 - gy1)
            is_right_goal = ((gx1 + gx2) / 2.0) > (fw * 0.5)
            side_prefix = "right" if is_right_goal else "left"
            goal_x_cm = 4000.0 if is_right_goal else 0.0
            if f"{side_prefix}_goalpost_far" not in existing_labels:
                far_post_x = gx1 + 0.12 * gw if is_right_goal else gx2 - 0.12 * gw
                far_post_y = gy1 + 0.64 * gh
                source_points.append([far_post_x, far_post_y])
                target_points.append([goal_x_cm, 850.0])
                matched_landmarks.append({
                    "id": len(matched_landmarks),
                    "label": f"{side_prefix}_goalpost_far",
                    "x": round(far_post_x, 1),
                    "y": round(far_post_y, 1),
                    "target_cm": [goal_x_cm, 850.0],
                    "confidence": 0.90,
                    "source": "goal_bbox_anchor"
                })
            if f"{side_prefix}_goalpost_near" not in existing_labels:
                near_post_x = gx2 - 0.10 * gw if is_right_goal else gx1 + 0.10 * gw
                near_post_y = gy2
                source_points.append([near_post_x, near_post_y])
                target_points.append([goal_x_cm, 1150.0])
                matched_landmarks.append({
                    "id": len(matched_landmarks),
                    "label": f"{side_prefix}_goalpost_near",
                    "x": round(near_post_x, 1),
                    "y": round(near_post_y, 1),
                    "target_cm": [goal_x_cm, 1150.0],
                    "confidence": 0.90,
                    "source": "goal_bbox_anchor"
                })

        if len(source_points) < 3:
            return None, 0, matched_landmarks

        src = np.array(source_points, dtype=np.float32)
        dst = np.array(target_points, dtype=np.float32)

        # Ensure target points span both X and Y dimensions
        span_x = float(np.ptp(dst[:, 0]))
        span_y = float(np.ptp(dst[:, 1]))
        if span_x < 100.0 or span_y < 100.0:
            return None, 0, matched_landmarks

        def _is_geometrically_valid(H_cand: np.ndarray, inlier_mask: Optional[np.ndarray]) -> bool:
            if H_cand is None or not np.all(np.isfinite(H_cand)):
                return False
            # 1. Check 2D rank / non-collinearity of inlier target points via SVD
            if inlier_mask is not None:
                inlier_idx = inlier_mask.ravel().astype(bool)
                dst_in = dst[inlier_idx]
            else:
                dst_in = dst
            if len(dst_in) >= 3:
                centered = dst_in - np.mean(dst_in, axis=0, keepdims=True)
                _, s_vals, _ = np.linalg.svd(centered, full_matrices=False)
                if len(s_vals) < 2 or float(s_vals[1]) < 65.0:
                    return False

            # 2. Check homogeneous depth positivity (no horizon crossing w <= 0)
            pts_to_check = src
            if test_points is not None and len(test_points) > 0:
                pts_to_check = np.vstack([src, test_points.astype(np.float32)])
            w_vals = H_cand[2, 0] * pts_to_check[:, 0] + H_cand[2, 1] * pts_to_check[:, 1] + H_cand[2, 2]
            if np.any(w_vals <= 0.15) or ( float(np.max(w_vals)) / max(1e-6, float(np.min(w_vals))) > 12.0 ):
                return False

            # 3. Check orientation consistency at centroid of points (right in image -> +X, down in image -> +Y)
            cx, cy = float(np.mean(pts_to_check[:, 0])), float(np.mean(pts_to_check[:, 1]))
            probe = np.array([[[cx, cy]], [[cx + 100.0, cy]], [[cx, cy + 100.0]]], dtype=np.float32)
            proj_probe = cv2.perspectiveTransform(probe, H_cand).reshape(3, 2)
            dx_du = proj_probe[1, 0] - proj_probe[0, 0]
            dy_dv = proj_probe[2, 1] - proj_probe[0, 1]
            if dx_du <= 0.0 or dy_dv <= 0.0:
                return False

            # 4. Check player foot projection bounds
            if test_points is not None and len(test_points) > 0:
                reshaped = test_points.reshape(-1, 1, 2).astype(np.float32)
                proj = cv2.perspectiveTransform(reshaped, H_cand).reshape(-1, 2)
                margin_x = 0.20 * self.court_length
                margin_y = 0.20 * self.court_width
                valid_x = (proj[:, 0] >= -margin_x) & (proj[:, 0] <= self.court_length + margin_x)
                valid_y = (proj[:, 1] >= -margin_y) & (proj[:, 1] <= self.court_width + margin_y)
                if float(np.mean(valid_x & valid_y)) < 0.70:
                    return False
            return True

        def _strict_inside_ratio(H_cand: np.ndarray) -> float:
            if test_points is None or len(test_points) == 0:
                return 1.0
            reshaped = test_points.reshape(-1, 1, 2).astype(np.float32)
            proj = cv2.perspectiveTransform(reshaped, H_cand).reshape(-1, 2)
            inside = (proj[:, 0] >= 0.0) & (proj[:, 0] <= self.court_length) & (proj[:, 1] >= 0.0) & (proj[:, 1] <= self.court_width)
            return float(np.mean(inside))

        # Candidate A: 6-DoF RANSAC Affine Homography (horizon-safe, immune to corner blow-up on single attack-zone views)
        aff_candidate = None
        aff_mask = None
        aff_inliers = 0
        if len(src) >= 3:
            aff_2x3, a_mask = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=120.0)
            a_inliers = int(np.sum(a_mask)) if a_mask is not None else 0
            if aff_2x3 is not None and a_inliers >= 3:
                aff_mat = np.vstack([aff_2x3, [0.0, 0.0, 1.0]]).astype(np.float64)
                if _is_geometrically_valid(aff_mat, a_mask):
                    aff_candidate = aff_mat
                    aff_mask = a_mask
                    aff_inliers = a_inliers

        # Candidate B: Full 8-DoF RANSAC Projective Homography
        proj_candidate = None
        proj_mask = None
        proj_inliers = 0
        if len(src) >= 4:
            for reproj in (80.0, 110.0, 55.0):
                matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, reproj)
                inliers_count = int(np.sum(mask)) if mask is not None else 0
                if matrix is not None and inliers_count >= 4 and _is_geometrically_valid(matrix, mask):
                    if inliers_count > proj_inliers:
                        proj_candidate = matrix
                        proj_mask = mask
                        proj_inliers = inliers_count
                        if proj_inliers >= 6:
                            break

        # On localized attack-zone views (span_x < 45% of court length), prefer 6-DoF Affine if it keeps >= as many players strictly inside the court
        if aff_candidate is not None and proj_candidate is not None:
            if span_x < 0.45 * self.court_length and _strict_inside_ratio(aff_candidate) >= _strict_inside_ratio(proj_candidate):
                self.homography_matrix = aff_candidate
                self.inliers = aff_mask
                return aff_candidate, aff_inliers, matched_landmarks
            if _strict_inside_ratio(proj_candidate) >= _strict_inside_ratio(aff_candidate):
                self.homography_matrix = proj_candidate
                self.inliers = proj_mask
                return proj_candidate, proj_inliers, matched_landmarks
            self.homography_matrix = aff_candidate
            self.inliers = aff_mask
            return aff_candidate, aff_inliers, matched_landmarks

        if proj_candidate is not None:
            self.homography_matrix = proj_candidate
            self.inliers = proj_mask
            return proj_candidate, proj_inliers, matched_landmarks

        if aff_candidate is not None:
            self.homography_matrix = aff_candidate
            self.inliers = aff_mask
            return aff_candidate, aff_inliers, matched_landmarks

        return None, 0, matched_landmarks

    def compute_ground_contact_points(
        self,
        player_detections: sv.Detections,
        frame_shape: Tuple[int, int],
        enable_vlm_priors: bool = False
    ) -> np.ndarray:
        """
        Computes the true floor contact point (Z=0) in pixel coordinates for each player:
        - Standard standing/running player: bottom-center of bbox or VLM ground_point_norm.
        - Bottom-edge truncated player (legs cropped off by camera bottom edge y2 >= H - 8, e.g. #22):
          extrapolates y_floor below the bottom frame edge using VLM ground_point_norm and anthropometric stature ratio.
        - Airborne jumping player (action_posture == 'jumping_shoot' or ground_contact_status == 'airborne'):
          uses VLM vertical shadow ground_point_norm if available, else estimates ground shadow beneath torso.
        """
        if len(player_detections) == 0:
            return np.empty((0, 2), dtype=np.float32)

        fh, fw = frame_shape[:2]
        boxes = player_detections.xyxy
        ground_pts = np.zeros((len(boxes), 2), dtype=np.float32)

        data = player_detections.data or {}
        statuses = data.get("ground_contact_status", ["visible_on_floor"] * len(boxes))
        postures = data.get("action_posture", ["standing_running"] * len(boxes))
        gp_norms = data.get("ground_point_norm", [None] * len(boxes))

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            xc = 0.5 * (x1 + x2)
            yf = y2

            status = str(statuses[i]) if enable_vlm_priors and i < len(statuses) else "visible_on_floor"
            posture = str(postures[i]) if enable_vlm_priors and i < len(postures) else "standing_running"
            gpn = gp_norms[i] if enable_vlm_priors and i < len(gp_norms) else None

            vlm_y_px, vlm_x_px = None, None
            if isinstance(gpn, (list, tuple)) and len(gpn) == 2 and gpn[0] is not None and gpn[1] is not None:
                gy, gx = float(gpn[0]), float(gpn[1])
                vlm_y_px = (gy / 1000.0) * fh if max(abs(gy), abs(gx)) > 1.5 else gy * fh
                vlm_x_px = (gx / 1000.0) * fw if max(abs(gy), abs(gx)) > 1.5 else gx * fw

            # Detect bottom-edge leg truncation (either explicitly flagged by VLM or geometrically touching bottom border with cropped aspect ratio)
            is_bottom_truncated = (
                status == "truncated_bottom_edge"
                or (y2 >= fh - 8.0 and (bh / bw) < 2.65)
            )

            if is_bottom_truncated:
                # Full standing foreground player has height/width ratio ~ 3.15
                anat_y_floor = y1 + max(bh * 1.38, 3.15 * bw)
                if vlm_y_px is not None and vlm_y_px > y2:
                    yf = max(vlm_y_px, anat_y_floor)
                else:
                    yf = anat_y_floor
                if vlm_x_px is not None and x1 <= vlm_x_px <= x2:
                    xc = vlm_x_px
            elif status == "airborne" or posture in ("jumping_shoot", "jumping_block"):
                # Airborne player: use VLM floor shadow if reasonable, otherwise project slightly below feet
                if vlm_y_px is not None and y1 <= vlm_y_px <= y2 + 0.35 * bh:
                    yf = vlm_y_px
                if vlm_x_px is not None and x1 - 0.2 * bw <= vlm_x_px <= x2 + 0.2 * bw:
                    xc = vlm_x_px
            else:
                if vlm_y_px is not None and (y2 - 0.15 * bh) <= vlm_y_px <= (y2 + 0.15 * bh):
                    yf = vlm_y_px
                if vlm_x_px is not None and x1 <= vlm_x_px <= x2:
                    xc = vlm_x_px

            ground_pts[i, 0] = xc
            ground_pts[i, 1] = yf

        return ground_pts

    def regularize_projected_players(
        self,
        projected_pts: np.ndarray,
        player_detections: sv.Detections,
        ground_pts_px: np.ndarray,
        landmarks: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        visible_court_side: str = "full",
        enable_vlm_priors: bool = False
    ) -> np.ndarray:
        """
        Applies physics- and rule-informed topological regularization to 2D court coordinates:
        1. Handball IHF Rule 6 (Goal Area):
           - Non-airborne field players ('ground_contact_status' != 'airborne' and 'action_posture' != 'jumping_shoot')
             cannot stand inside the 6m goal area (radial distance r from goalposts < 600 cm).
             Players with 'on_6m_line' or 'defensive_stance' are snapped to the outer edge of the 6m arc (r in [615, 655] cm).
           - Airborne jump-shooters ('jumping_shoot' / 'airborne') legally flying over the 6m zone (IHF Rule 6:7a)
             preserve their in-zone air projection.
        2. Topological ordering relative to the 7m penalty mark and 9m free-throw arc:
           - If a defender's feet in the image are clearly on the camera/midcourt side of the 7m penalty mark,
             ensure their projected X coordinate stays on the midcourt side of the 7m mark (e.g. X <= 3285 cm on right side).
        3. Foreground bottom-truncated players (e.g. #22 at bottom frame edge):
           - Ensure lateral Y coordinate reflects their true near-sideline position derived from extrapolated ground contact.
        """
        if projected_pts is None or len(projected_pts) == 0:
            return projected_pts

        reg = np.array(projected_pts, dtype=np.float32, copy=True)
        fh, fw = frame_shape[:2]
        boxes = player_detections.xyxy
        data = player_detections.data or {}
        statuses = data.get("ground_contact_status", ["visible_on_floor"] * len(boxes))
        postures = data.get("action_posture", ["standing_running"] * len(boxes))
        zones = data.get("court_zone_hint", [""] * len(boxes))

        if self.sport != "handball":
            return reg

        # Determine active goal side (left X=0 or right X=4000)
        mean_x = float(np.mean(reg[:, 0]))
        is_right = (visible_court_side == "right") or (visible_court_side != "left" and mean_x >= 2000.0)
        goal_x = 4000.0 if is_right else 0.0

        # Find 7m penalty mark pixel if detected among landmarks
        pen7m_px = None
        for lm in (landmarks or []):
            lbl = str(lm.get("label", "")).lower()
            if "7m_penalty_mark" in lbl:
                pen7m_px = (float(lm["x"]), float(lm["y"]))
                break

        def _handball_goal_radial_vector(pt_x: float, pt_y: float) -> Tuple[float, float, float, float]:
            """
            Returns (anchor_x, anchor_y, radial_dist_cm, unit_dx, unit_dy) from the nearest goalpost/goal-segment
            following official IHF 6m/9m D-zone geometry (straight 3m segment Y in [850, 1150] + quarter-circles at Y=850 and Y=1150).
            """
            anchor_x = goal_x
            anchor_y = float(np.clip(pt_y, 850.0, 1150.0))
            dx = pt_x - anchor_x
            dy = pt_y - anchor_y
            # Ensure dx points towards midcourt
            if is_right and dx > -10.0:
                dx = -max(10.0, abs(dx))
            elif (not is_right) and dx < 10.0:
                dx = max(10.0, abs(dx))
            r = float(np.hypot(dx, dy))
            if r < 1e-3:
                ux = -1.0 if is_right else 1.0
                uy = 0.0
            else:
                ux = dx / r
                uy = dy / r
            return anchor_x, anchor_y, r, ux, uy

        inferred_statuses = []
        inferred_postures = []
        inferred_zones = []
        vlm_teams = data.get("vlm_team", [-1] * len(boxes))
        defending_team = 1 if is_right else 0

        for i in range(len(reg)):
            x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            foot_x, foot_y = float(ground_pts_px[i, 0]), float(ground_pts_px[i, 1])

            status = str(statuses[i]) if i < len(statuses) and statuses[i] else "visible_on_floor"
            posture = str(postures[i]) if i < len(postures) and postures[i] else "standing_running"
            zone = str(zones[i]) if i < len(zones) and zones[i] else ""
            p_team = int(vlm_teams[i]) if i < len(vlm_teams) else -1

            is_bottom_truncated = (status == "truncated_bottom_edge") or (y2 >= fh - 8.0 and (bh / bw) < 2.65)
            if is_bottom_truncated:
                status = "truncated_bottom_edge"

            is_airborne = (status == "airborne") or (posture in ("jumping_shoot", "jumping_block"))

            # 1. Bottom-truncated foreground player (e.g. #22 at bottom of image near sideline):
            # Because foot_y was extrapolated below the bottom edge (foot_y > fh), linear affine X = a*x - b*y + c
            # would artificially push X too far towards midcourt while underestimating Y near the bottom sideline.
            # Anchor X at the visible bottom frame edge (y = fh) and project Y to the near sideline (Y ~ 1885..1930 cm).
            if is_bottom_truncated and foot_y > fh:
                if self.homography_matrix is not None:
                    edge_pt = np.array([[[foot_x, float(fh)]]], dtype=np.float32)
                    edge_proj = cv2.perspectiveTransform(edge_pt, self.homography_matrix).reshape(2)
                    reg[i, 0] = float(edge_proj[0])
                extra_ratio = min(1.0, (foot_y - fh) / (0.22 * fh))
                target_near_y = 1865.0 + 75.0 * extra_ratio
                reg[i, 1] = max(float(reg[i, 1]), target_near_y)

            px_cm, py_cm = float(reg[i, 0]), float(reg[i, 1])
            ax, ay, r_dist, ux, uy = _handball_goal_radial_vector(px_cm, py_cm)

            # Infer posture and court_zone_hint when not explicitly provided by VLM
            if not zone:
                if py_cm < 320.0 or py_cm > 1680.0:
                    zone = "wing_corner" if abs(goal_x - px_cm) < 680.0 else "beyond_9m"
                elif r_dist < 680.0:
                    zone = "on_6m_line"
                elif r_dist < 890.0:
                    zone = "between_6m_and_9m"
                elif r_dist <= 940.0:
                    zone = "on_9m_line"
                else:
                    zone = "beyond_9m"

            if posture == "standing_running" and p_team == defending_team and r_dist < 860.0:
                posture = "defensive_stance"

            # 2. IHF Rule 6 Goal-Area & Zone Regularization (only for non-airborne players!)
            # Note: radar player circle has radius 12px (~24cm on court), so keeping r >= 655cm ensures the
            # drawn player disc never visually overlaps the white 6m arc or blue goalkeeper area.
            if not is_airborne:
                if zone == "on_6m_line" or (posture == "defensive_stance" and r_dist < 675.0):
                    target_r = float(np.clip(r_dist, 658.0, 682.0))
                    reg[i, 0] = ax + ux * target_r
                    reg[i, 1] = np.clip(ay + uy * target_r, 45.0, self.court_width - 45.0)
                elif zone == "between_6m_and_9m":
                    target_r = float(np.clip(r_dist, 675.0, 885.0))
                    reg[i, 0] = ax + ux * target_r
                    reg[i, 1] = np.clip(ay + uy * target_r, 45.0, self.court_width - 45.0)
                elif zone == "on_9m_line":
                    target_r = float(np.clip(r_dist, 890.0, 935.0))
                    reg[i, 0] = ax + ux * target_r
                    reg[i, 1] = np.clip(ay + uy * target_r, 45.0, self.court_width - 45.0)
                elif zone == "beyond_9m":
                    if r_dist < 935.0:
                        target_r = 955.0
                        reg[i, 0] = ax + ux * target_r
                        reg[i, 1] = np.clip(ay + uy * target_r, 45.0, self.court_width - 45.0)
                else:
                    # Default physical rule for any standing court player (including wing_corner):
                    # radial distance from nearest goalpost must be >= 658cm
                    if r_dist < 658.0:
                        target_r = 660.0
                        reg[i, 0] = ax + ux * target_r
                        reg[i, 1] = np.clip(ay + uy * target_r, 45.0, self.court_width - 45.0)

                # 3. Topological consistency with the 7m penalty mark in the central corridor (Y in [780, 1180] cm):
                # If a central defender's shoes in the image are lower/more foreground than the 7m mark (y_foot > y_7m + 20px),
                # they are physically on the midcourt/camera side of the 7m mark (X <= 3255 cm on right court so the disc is clearly left of the 7m line).
                if pen7m_px is not None and 780.0 <= float(reg[i, 1]) <= 1180.0:
                    if abs(foot_x - pen7m_px[0]) <= 180.0 and (foot_y - pen7m_px[1]) >= 20.0:
                        if is_right and float(reg[i, 0]) > 3255.0:
                            reg[i, 0] = 3255.0
                            zone = "between_6m_and_9m"
                        elif (not is_right) and float(reg[i, 0]) < 745.0:
                            reg[i, 0] = 745.0
                            zone = "between_6m_and_9m"

            # Keep inside court boundaries [45, L-45] x [45, W-45]
            reg[i, 0] = float(np.clip(reg[i, 0], 45.0, self.court_length - 45.0))
            reg[i, 1] = float(np.clip(reg[i, 1], 45.0, self.court_width - 45.0))

            inferred_statuses.append(status)
            inferred_postures.append(posture)
            inferred_zones.append(zone)

        if player_detections.data is not None:
            player_detections.data["ground_contact_status"] = np.array(inferred_statuses, dtype=object)
            player_detections.data["action_posture"] = np.array(inferred_postures, dtype=object)
            player_detections.data["court_zone_hint"] = np.array(inferred_zones, dtype=object)

        return reg

    def transform_points(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Transforms pixel coordinates to 2D court metric coordinates using homography matrix."""
        if self.homography_matrix is None or len(points) == 0:
            return None
            
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.homography_matrix)
        return transformed.reshape(-1, 2)

    def estimate_perspective_projection(self, points: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Robust perspective projection mapping player foot positions to standard 2D court coordinates
        when explicit keypoints are unavailable or insufficient.
        """
        if points is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
            
        fh, fw = frame_shape[:2]
        pts = np.array(points, dtype=np.float32)
        
        # Longitudinal position (X axis along court length)
        norm_x = np.clip(pts[:, 0] / float(fw), 0.05, 0.95)
        court_x = norm_x * self.court_length
        
        # Lateral perspective position (Y axis along court width)
        # Elevated sports cameras view the pitch from the sideline:
        # bottom of the frame is near sideline (Y near court_width), top of frame is far sideline (Y near 0)
        norm_y = np.clip((pts[:, 1] - 0.15 * fh) / (0.75 * fh), 0.05, 0.95)
        # Apply gentle perspective power to account for ground plane foreshortening
        persp_y = np.power(norm_y, 1.2)
        court_y = persp_y * self.court_width
        
        return np.column_stack([court_x, court_y]).astype(np.float32)

    def is_on_court(self, transformed_points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Returns boolean mask indicating whether points are inside court bounds."""
        if transformed_points is None or len(transformed_points) == 0:
            return np.array([], dtype=bool)
            
        in_x = (transformed_points[:, 0] >= -margin) & (transformed_points[:, 0] <= self.court_length + margin)
        in_y = (transformed_points[:, 1] >= -margin) & (transformed_points[:, 1] <= self.court_width + margin)
        return in_x & in_y

