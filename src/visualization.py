import supervision as sv
import cv2
import numpy as np
from typing import Optional, List, Union

from sports.basketball import (
    CourtConfiguration,
    League,
    draw_court,
    draw_points_on_court
)
from sports.soccer.annotators import (
    SoccerPitchConfiguration,
    draw_pitch,
    draw_points_on_pitch
)
from src.config import PLAYER_COLOR_PALETTE, KEYPOINT_COLOR


def draw_handball_court(length_cm: float = 4000.0, width_cm: float = 2000.0, scale: float = 0.2, padding: int = 40) -> np.ndarray:
    """Draws a standard 40m x 20m 2D handball court."""
    cw = int(length_cm * scale)
    ch = int(width_cm * scale)
    img_w = cw + 2 * padding
    img_h = ch + 2 * padding
    
    # Outer surround (golden wood)
    court = np.full((img_h, img_w, 3), (40, 140, 220), dtype=np.uint8)
    # Playing area (blue)
    court[padding:padding+ch, padding:padding+cw] = (180, 105, 30)
    
    # Boundary line
    cv2.rectangle(court, (padding, padding), (padding+cw, padding+ch), (255, 255, 255), 3)
    
    # Center line & circle
    mid_x = padding + cw // 2
    cv2.line(court, (mid_x, padding), (mid_x, padding+ch), (255, 255, 255), 3)
    cv2.circle(court, (mid_x, padding + ch // 2), int(300 * scale), (255, 255, 255), 3)
    
    # Goal areas (6m solid and 9m dashed)
    r6 = int(600 * scale)
    r9 = int(900 * scale)
    mid_y = padding + ch // 2
    goal_w = int(300 * scale) // 2
    
    for side in [0, 1]:
        gx = padding if side == 0 else padding + cw
        direction = 1 if side == 0 else -1
        
        # 6m solid arc (top quarter-circle: -90..0 on left, 180..270 on right; bottom quarter-circle: 0..90 on left, 90..180 on right)
        cv2.ellipse(court, (gx, mid_y - goal_w), (r6, r6), 0, -90 if side == 0 else 180, 0 if side == 0 else 270, (255, 255, 255), 3)
        cv2.ellipse(court, (gx, mid_y + goal_w), (r6, r6), 0, 0 if side == 0 else 90, 90 if side == 0 else 180, (255, 255, 255), 3)
        cv2.line(court, (gx + direction * r6, mid_y - goal_w), (gx + direction * r6, mid_y + goal_w), (255, 255, 255), 3)
        
        # 9m dashed free-throw arc (clipped to court sidelines [padding, padding + ch])
        for deg in range(-80, 0, 8):
            a1 = np.deg2rad(deg) if side == 0 else np.deg2rad(180 - deg)
            a2 = np.deg2rad(deg + 4) if side == 0 else np.deg2rad(180 - (deg + 4))
            x1 = int(round(gx + r9 * np.cos(a1)))
            y1 = int(round((mid_y - goal_w) + r9 * np.sin(a1)))
            x2 = int(round(gx + r9 * np.cos(a2)))
            y2 = int(round((mid_y - goal_w) + r9 * np.sin(a2)))
            if y1 >= padding and y2 >= padding:
                cv2.line(court, (x1, y1), (x2, y2), (255, 255, 255), 2)
        for deg in range(0, 80, 8):
            a1 = np.deg2rad(deg) if side == 0 else np.deg2rad(180 - deg)
            a2 = np.deg2rad(deg + 4) if side == 0 else np.deg2rad(180 - (deg + 4))
            x1 = int(round(gx + r9 * np.cos(a1)))
            y1 = int(round((mid_y + goal_w) + r9 * np.sin(a1)))
            x2 = int(round(gx + r9 * np.cos(a2)))
            y2 = int(round((mid_y + goal_w) + r9 * np.sin(a2)))
            if y1 <= padding + ch and y2 <= padding + ch:
                cv2.line(court, (x1, y1), (x2, y2), (255, 255, 255), 2)
        x9 = gx + direction * r9
        for yy in range(mid_y - goal_w, mid_y + goal_w, 12):
            cv2.line(court, (x9, yy), (x9, min(mid_y + goal_w, yy + 6)), (255, 255, 255), 2)
        
        # 7m penalty mark (1m line)
        p7_x = gx + direction * int(700 * scale)
        cv2.line(court, (p7_x, mid_y - int(50*scale)), (p7_x, mid_y + int(50*scale)), (255, 255, 255), 3)
        
        # 4m goalkeeper limit mark
        p4_x = gx + direction * int(400 * scale)
        cv2.line(court, (p4_x, mid_y - int(25*scale)), (p4_x, mid_y + int(25*scale)), (255, 255, 255), 2)
        
        # Goalposts (red/white)
        cv2.rectangle(court, (gx - (direction*10), mid_y - goal_w), (gx, mid_y + goal_w), (0, 0, 255), -1)

    return court


def draw_football_field(length_yards: float = 120.0, width_yards: float = 53.3, scale: float = 7.5, padding: int = 40) -> np.ndarray:
    """Draws a standard 120-yard x 53.3-yard 2D American football field."""
    cw = int(length_yards * scale)
    ch = int(width_yards * scale)
    img_w = cw + 2 * padding
    img_h = ch + 2 * padding
    
    # Turf green
    field = np.full((img_h, img_w, 3), (35, 115, 45), dtype=np.uint8)
    
    # Outer sideline
    cv2.rectangle(field, (padding, padding), (padding+cw, padding+ch), (255, 255, 255), 3)
    
    # End zones (10 yards each)
    ez_w = int(10 * scale)
    field[padding:padding+ch, padding:padding+ez_w] = (25, 80, 30)
    cv2.line(field, (padding+ez_w, padding), (padding+ez_w, padding+ch), (255, 255, 255), 3)
    field[padding:padding+ch, padding+cw-ez_w:padding+cw] = (25, 80, 30)
    cv2.line(field, (padding+cw-ez_w, padding), (padding+cw-ez_w, padding+ch), (255, 255, 255), 3)
    
    # Yard lines every 5 yards between endzones (15 to 105)
    for yd in range(15, 110, 5):
        lx = padding + int(yd * scale)
        is_ten = ((yd - 10) % 10 == 0)
        thickness = 2 if is_ten else 1
        cv2.line(field, (lx, padding), (lx, padding+ch), (255, 255, 255), thickness)
        
        # Numbers on 10-yard lines
        if is_ten:
            val = (yd - 10) if yd <= 60 else (110 - yd)
            txt = str(val)
            cv2.putText(field, txt, (lx - 12, padding + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(field, txt, (lx - 12, padding + ch - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return field


class SportsAnnotator:
    """
    Unified multi-sport annotator supporting 2D court/pitch minimap radar rendering,
    bounding box detections, keypoints, and side-by-side composite visualization.
    """
    def __init__(self, sport: str = "basket"):
        self.sport = sport.lower()
        self.box_annotator = sv.BoxAnnotator(color=PLAYER_COLOR_PALETTE, thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            color=PLAYER_COLOR_PALETTE, 
            text_color=sv.Color.BLACK
        )
        self.mask_annotator = sv.MaskAnnotator(color=PLAYER_COLOR_PALETTE, opacity=0.5)
        self.vertex_annotator = sv.VertexAnnotator(color=KEYPOINT_COLOR, radius=8)
        
        # Initialize sport-specific court/pitch template
        if self.sport == "basket":
            self.court_config = CourtConfiguration(league=League.NBA)
            self.court_image = draw_court(
                config=self.court_config, 
                scale=1.0, 
                line_thickness=5,
                padding=50
            )
            self.court_dims = (2865.0, 1524.0)
            self.scale = 1.0
            self.padding = 50
        elif self.sport == "foot":
            self.court_config = SoccerPitchConfiguration()
            self.court_image = draw_pitch(
                config=self.court_config,
                scale=0.1,
                padding=50
            )
            self.court_dims = (12000.0, 7000.0)
            self.scale = 0.1
            self.padding = 50
        elif self.sport == "handball":
            self.court_config = None
            self.scale = 0.2
            self.padding = 40
            self.court_dims = (4000.0, 2000.0)
            self.court_image = draw_handball_court(4000.0, 2000.0, scale=self.scale, padding=self.padding)
        elif self.sport == "usfootball":
            self.court_config = None
            self.scale = 7.5 / 91.44  # yards to cm conversion factor
            self.padding = 40
            self.court_dims = (10973.0, 4877.0)
            self.court_image = draw_football_field(120.0, 53.3, scale=7.5, padding=self.padding)
        else:
            self.court_config = CourtConfiguration(league=League.NBA)
            self.court_image = draw_court(config=self.court_config, scale=1.0, line_thickness=5, padding=50)
            self.court_dims = (2865.0, 1524.0)
            self.scale = 1.0
            self.padding = 50

    def annotate_frame(self, frame, detections, labels):
        annotated_frame = frame.copy()
        if detections.mask is not None:
            annotated_frame = self.mask_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, 
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
            key_points=keypoints
        )

    def draw_court_overlay(self, detections_xy: np.ndarray, colors: Optional[List[sv.Color]] = None) -> np.ndarray:
        court_image = self.court_image.copy()
        if detections_xy is None or len(detections_xy) == 0:
            return court_image
            
        pts = np.array(detections_xy, dtype=np.float32)
        n_pts = len(pts)
        if colors is None:
            colors = [sv.Color.BLACK] * n_pts

        if self.sport == "basket":
            unique_colors = set(colors)
            for color in unique_colors:
                mask = [c == color for c in colors]
                points_subset = pts[mask]
                court_image = draw_points_on_court(
                    config=self.court_config,
                    xy=points_subset,
                    court=court_image,
                    scale=1.0,
                    size=40,
                    padding=50,
                    fill_color=color
                )
            return court_image

        elif self.sport == "foot":
            unique_colors = set(colors)
            for color in unique_colors:
                mask = [c == color for c in colors]
                points_subset = pts[mask]
                court_image = draw_points_on_pitch(
                    config=self.court_config,
                    xy=points_subset,
                    pitch=court_image,
                    scale=self.scale,
                    padding=self.padding,
                    face_color=color,
                    edge_color=sv.Color.BLACK,
                    radius=12,
                    thickness=2
                )
            return court_image

        elif self.sport in ["handball", "usfootball"]:
            # Custom dot plotting on 2D court image
            for i, pt in enumerate(pts):
                color = colors[i] if i < len(colors) else sv.Color.BLACK
                bgr = color.as_bgr() if hasattr(color, "as_bgr") else (0, 0, 0)
                px = int(self.padding + pt[0] * self.scale)
                py = int(self.padding + pt[1] * self.scale)
                # Keep within interior court lines so active players never spill into the outer margin
                px = max(self.padding + 6, min(court_image.shape[1] - self.padding - 6, px))
                py = max(self.padding + 6, min(court_image.shape[0] - self.padding - 6, py))
                cv2.circle(court_image, (px, py), 12, bgr, -1)
                cv2.circle(court_image, (px, py), 12, (0, 0, 0), 2)
            return court_image

        return court_image

    def overlay_court(self, frame: np.ndarray, court_image: np.ndarray) -> np.ndarray:
        fh, fw = frame.shape[:2]
        ch, cw = court_image.shape[:2]
        
        # Scaling the court image to roughly 1/3.5 of the frame height
        scale_factor = (fh / 3.5) / ch
        new_cw = int(cw * scale_factor)
        new_ch = int(ch * scale_factor)
        resized_court = cv2.resize(court_image, (new_cw, new_ch))
        
        # Place in top-right corner with 20px padding
        padding = 20
        frame[padding:padding+new_ch, fw-new_cw-padding:fw-padding] = resized_court
        return frame


class BasketballAnnotator(SportsAnnotator):
    """Backwards-compatible subclass defaulting to basketball configuration."""
    def __init__(self):
        super().__init__(sport="basket")

