from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

@dataclass
class SportConfig:
    name: str
    display_name: str
    aliases: List[str]
    classes: List[str]
    prompt_description: str
    ball_class_name: str
    goal_class_name: str
    has_goalkeeper: bool = False
    max_players_on_court_per_team: int = 5
    default_court_dims: Optional[Tuple[float, float]] = None # (length, width) in cm
    keypoint_model_id: Optional[str] = None
    radar_type: Optional[str] = None
    vlm_landmarks_prompt: str = ""
    court_zones_prompt: str = "'inside_paint_or_goal_area', 'mid_range', 'beyond_arc_or_9m', 'wing_corner', 'midcourt'"
    canonical_landmarks_cm: Dict[str, Tuple[float, float]] = field(default_factory=dict)

SPORTS_CONFIGS: Dict[str, SportConfig] = {
    "basket": SportConfig(
        name="basket",
        display_name="Basketball",
        aliases=["basket", "basketball", "nba", "bball"],
        classes=["players", "referees", "ball", "basket", "field_marks"],
        ball_class_name="ball",
        goal_class_name="basket",
        has_goalkeeper=False,
        max_players_on_court_per_team=5,
        default_court_dims=(2865.0, 1524.0),
        keypoint_model_id="basketball-court-detection-2/14",
        radar_type="basketball",
        prompt_description=(
            "Detect all objects belonging to these classes in the basketball image:\n"
            "- 'players': active basketball players wearing team uniforms (jerseys and shorts) on the court.\n"
            "- 'referees': match referees / officials (wearing striped or grey referee shirts and black pants, including referees on the sidelines).\n"
            "- 'ball': the basketball.\n"
            "- 'basket': the rim/hoop/net of the basketball goal.\n"
            "- 'field_marks': tactical court lines (the key / paint boundary, free-throw circle, 3-point line, center circle)."
        ),
        vlm_landmarks_prompt=(
            "Identify visible basketball court FLOOR line intersections from this canonical list "
            "(all points MUST be on the wooden/painted court floor Z=0, never up in the air on the hoop; "
            "'far' = top of image / far sideline Y=0, 'near' = bottom of image / near sideline Y=1524):\n"
            "  Left half-court: 'left_corner_far', 'left_corner_near', 'left_paint_far_baseline', 'left_paint_near_baseline', "
            "'left_ft_line_far', 'left_ft_line_near', 'left_ft_circle_apex', 'left_3pt_top_of_arc', 'left_restricted_arc_apex'\n"
            "  Center court: 'center_line_far', 'center_circle_center', 'center_line_near'\n"
            "  Right half-court: 'right_corner_far', 'right_corner_near', 'right_paint_far_baseline', 'right_paint_near_baseline', "
            "'right_ft_line_far', 'right_ft_line_near', 'right_ft_circle_apex', 'right_3pt_top_of_arc', 'right_restricted_arc_apex'"
        ),
        canonical_landmarks_cm={
            "left_corner_far": (0.0, 0.0),
            "left_corner_near": (0.0, 1524.0),
            "left_paint_far_baseline": (0.0, 518.0),
            "left_paint_near_baseline": (0.0, 1006.0),
            "left_ft_line_far": (579.0, 518.0),
            "left_ft_line_near": (579.0, 1006.0),
            "left_ft_circle_apex": (759.0, 762.0),
            "left_3pt_top_of_arc": (884.0, 762.0),
            "left_restricted_arc_apex": (282.0, 762.0),
            "center_line_far": (1432.5, 0.0),
            "center_circle_center": (1432.5, 762.0),
            "center_line_near": (1432.5, 1524.0),
            "right_corner_far": (2865.0, 0.0),
            "right_corner_near": (2865.0, 1524.0),
            "right_paint_far_baseline": (2865.0, 518.0),
            "right_paint_near_baseline": (2865.0, 1006.0),
            "right_ft_line_far": (2286.0, 518.0),
            "right_ft_line_near": (2286.0, 1006.0),
            "right_ft_circle_apex": (2106.0, 762.0),
            "right_3pt_top_of_arc": (1981.0, 762.0),
            "right_restricted_arc_apex": (2583.0, 762.0),
        }
    ),
    "foot": SportConfig(
        name="foot",
        display_name="Football / Soccer",
        aliases=["foot", "soccer", "football"],
        classes=["players", "goalkeepers", "referees", "ball", "goal", "field_marks"],
        ball_class_name="ball",
        goal_class_name="goal",
        has_goalkeeper=True,
        max_players_on_court_per_team=11,
        default_court_dims=(12000.0, 7000.0),
        keypoint_model_id="football-field-detection-f07vi/14",
        radar_type="soccer",
        prompt_description=(
            "Detect all objects belonging to these classes in the soccer/football image:\n"
            "- 'players': outfield soccer players wearing team jerseys, shorts, and socks on the pitch.\n"
            "- 'goalkeepers': goalkeepers wearing distinct goalie kit/jersey and gloves.\n"
            "- 'referees': match referees and assistant referees/linesmen (wearing official referee kit, typically neon, yellow, or black).\n"
            "- 'ball': the soccer ball.\n"
            "- 'goal': the soccer goalposts and net (cages de but).\n"
            "- 'field_marks': tactical pitch markings (penalty box, penalty spot, 6-yard goal area, center circle, touchlines, corner arc)."
        ),
        vlm_landmarks_prompt=(
            "Identify visible soccer pitch line intersections from this canonical list "
            "('far' = top of image / far touchline Y=0, 'near' = bottom of image / near touchline Y=7000):\n"
            "  Left side: 'left_corner_far', 'left_corner_near', 'left_penalty_box_far_goal', 'left_penalty_box_near_goal', "
            "'left_penalty_box_far_corner', 'left_penalty_box_near_corner', 'left_6yd_far_corner', 'left_6yd_near_corner', 'left_penalty_spot'\n"
            "  Center: 'center_line_far', 'center_spot', 'center_line_near', 'center_circle_far', 'center_circle_near'\n"
            "  Right side: 'right_corner_far', 'right_corner_near', 'right_penalty_box_far_goal', 'right_penalty_box_near_goal', "
            "'right_penalty_box_far_corner', 'right_penalty_box_near_corner', 'right_6yd_far_corner', 'right_6yd_near_corner', 'right_penalty_spot'"
        ),
        canonical_landmarks_cm={
            "left_corner_far": (0.0, 0.0),
            "left_corner_near": (0.0, 7000.0),
            "left_penalty_box_far_goal": (0.0, 1484.0),
            "left_penalty_box_near_goal": (0.0, 5516.0),
            "left_penalty_box_far_corner": (2015.0, 1484.0),
            "left_penalty_box_near_corner": (2015.0, 5516.0),
            "left_6yd_far_corner": (550.0, 2584.0),
            "left_6yd_near_corner": (550.0, 4416.0),
            "left_penalty_spot": (1100.0, 3500.0),
            "center_line_far": (6000.0, 0.0),
            "center_spot": (6000.0, 3500.0),
            "center_line_near": (6000.0, 7000.0),
            "center_circle_far": (6000.0, 2585.0),
            "center_circle_near": (6000.0, 4415.0),
            "right_corner_far": (12000.0, 0.0),
            "right_corner_near": (12000.0, 7000.0),
            "right_penalty_box_far_goal": (12000.0, 1484.0),
            "right_penalty_box_near_goal": (12000.0, 5516.0),
            "right_penalty_box_far_corner": (9985.0, 1484.0),
            "right_penalty_box_near_corner": (9985.0, 5516.0),
            "right_6yd_far_corner": (11450.0, 2584.0),
            "right_6yd_near_corner": (11450.0, 4416.0),
            "right_penalty_spot": (10900.0, 3500.0),
        }
    ),
    "handball": SportConfig(
        name="handball",
        display_name="Handball",
        aliases=["handball", "hand"],
        classes=["players", "goalkeepers", "referees", "ball", "goal", "field_marks"],
        ball_class_name="ball",
        goal_class_name="goal",
        has_goalkeeper=True,
        max_players_on_court_per_team=7,
        default_court_dims=(4000.0, 2000.0),
        radar_type="handball",
        prompt_description=(
            "Detect all objects belonging to these classes in the handball image:\n"
            "- 'players': active handball field players wearing sleeveless or short-sleeved team jerseys and shorts.\n"
            "- 'goalkeepers': handball goalkeepers wearing long sweatpants and distinct colored long-sleeved jerseys.\n"
            "- 'referees': match referees wearing official referee kit.\n"
            "- 'ball': the handball.\n"
            "- 'goal': the handball goalposts (often striped red/white or blue/white) and net.\n"
            "- 'field_marks': tactical court markings (6m solid goal area arc, 9m dashed free-throw line, 7m penalty line, center line)."
        ),
        vlm_landmarks_prompt=(
            "Identify 6 to 12 visible handball court landmarks on the floor from this canonical IHF list. "
            "CRITICAL FOR 2D HOMOGRAPHY: You MUST select landmarks spread across BOTH court depth (goal line 0m -> 4m -> 6m -> 7m -> 9m -> center 20m) "
            "AND court width ('far' top/background side Y=0..850, 'center' Y=1000, 'near' bottom/foreground side Y=1150..2000). "
            "Do NOT return only center-axis points! Always include visible goalposts, 6m goal-line intersections, 6m/9m wings, corners, or sidelines:\n"
            "  - Goal & Goal Line (X=0 left / X=4000 right): "
            "'<side>_corner_far' (Y=0), '<side>_6m_goalline_far' (where 6m arc meets goal line, Y=250), "
            "'<side>_goalpost_far' (base of far goalpost, Y=850), '<side>_goal_center' (Y=1000), '<side>_goalpost_near' (base of near goalpost, Y=1150), "
            "'<side>_6m_goalline_near' (where 6m arc meets goal line, Y=1750), '<side>_corner_near' (Y=2000)\n"
            "  - 4m Goalkeeper & 7m Penalty Marks: "
            "'<side>_4m_mark' (4m line in front of goal, Y=1000), '<side>_7m_far_end' (Y=950), '<side>_7m_penalty_mark' (7m line center, Y=1000), '<side>_7m_near_end' (Y=1050)\n"
            "  - 6m Solid Goal Area Line: "
            "'<side>_6m_arc_far' (45-deg far wing of 6m arc), '<side>_6m_straight_far' (6m line in front of far post, Y=850), "
            "'<side>_6m_arc_center' (6m line center, Y=1000), '<side>_6m_straight_near' (6m line in front of near post, Y=1150), '<side>_6m_arc_near' (45-deg near wing of 6m arc)\n"
            "  - 9m Dashed Free-Throw Line: "
            "'<side>_9m_sideline_far' (where 9m dashed arc meets far sideline, Y=0), '<side>_9m_arc_far' (45-deg far wing of 9m arc), "
            "'<side>_9m_straight_far' (9m line in front of far post, Y=850), '<side>_9m_arc_center' (9m line center, Y=1000), "
            "'<side>_9m_straight_near' (9m line in front of near post, Y=1150), '<side>_9m_arc_near' (45-deg near wing of 9m arc), "
            "'<side>_9m_sideline_near' (where 9m dashed arc meets near sideline, Y=2000)\n"
            "  - Midcourt & Substitution Lines: "
            "'<side>_substitution_line_far' (Y=0), '<side>_substitution_line_near' (Y=2000), 'center_line_far' (Y=0), 'center_spot' (Y=1000), 'center_line_near' (Y=2000)\n"
            "  (Replace <side> with 'left' if the goal is on the left side of the image, or 'right' if the goal is on the right side)."
        ),
        court_zones_prompt=(
            "'inside_6m' (feet/shadow inside the blue 6m goal-area D-zone), "
            "'on_6m_line' (defending or pivoting right on/just outside the white 6m goal-area line, 6.0m to 6.5m from goal), "
            "'between_6m_and_9m' (standing between the 6m solid line and the 9m dashed free-throw line, ~6.5m to 8.8m from goal), "
            "'on_9m_line' (standing on or right next to the 9m dashed line), "
            "'beyond_9m' (backcourt player outside the 9m dashed line towards midcourt), "
            "or 'wing_corner' (wing player near the corner between the 6m goal-line intersection and the sideline corner)"
        ),
        canonical_landmarks_cm={
            # Left Half-Court (Goal at X = 0.0)
            "left_corner_far": (0.0, 0.0),
            "left_6m_goalline_far": (0.0, 250.0),
            "left_6m_sideline_far": (0.0, 250.0),
            "left_goalpost_far": (0.0, 850.0),
            "left_goal_center": (0.0, 1000.0),
            "left_goalpost_near": (0.0, 1150.0),
            "left_6m_goalline_near": (0.0, 1750.0),
            "left_6m_sideline_near": (0.0, 1750.0),
            "left_corner_near": (0.0, 2000.0),
            "left_4m_mark": (400.0, 1000.0),
            "left_6m_arc_far": (490.0, 560.0),
            "left_6m_straight_far": (600.0, 850.0),
            "left_6m_arc_center": (600.0, 1000.0),
            "left_6m_straight_near": (600.0, 1150.0),
            "left_6m_arc_near": (490.0, 1440.0),
            "left_7m_far_end": (700.0, 950.0),
            "left_7m_penalty_mark": (700.0, 1000.0),
            "left_7m_near_end": (700.0, 1050.0),
            "left_9m_sideline_far": (296.0, 0.0),
            "left_9m_arc_far": (760.0, 440.0),
            "left_9m_straight_far": (900.0, 850.0),
            "left_9m_arc_center": (900.0, 1000.0),
            "left_9m_straight_near": (900.0, 1150.0),
            "left_9m_arc_near": (760.0, 1560.0),
            "left_9m_sideline_near": (296.0, 2000.0),
            "left_substitution_line_far": (1550.0, 0.0),
            "left_substitution_line_near": (1550.0, 2000.0),
            # Center Line (X = 2000.0)
            "center_line_far": (2000.0, 0.0),
            "center_spot": (2000.0, 1000.0),
            "center_line_near": (2000.0, 2000.0),
            # Right Half-Court (Goal at X = 4000.0)
            "right_substitution_line_far": (2450.0, 0.0),
            "right_substitution_line_near": (2450.0, 2000.0),
            "right_9m_sideline_far": (3704.0, 0.0),
            "right_9m_arc_far": (3240.0, 440.0),
            "right_9m_straight_far": (3100.0, 850.0),
            "right_9m_arc_center": (3100.0, 1000.0),
            "right_9m_straight_near": (3100.0, 1150.0),
            "right_9m_arc_near": (3240.0, 1560.0),
            "right_9m_sideline_near": (3704.0, 2000.0),
            "right_7m_far_end": (3300.0, 950.0),
            "right_7m_penalty_mark": (3300.0, 1000.0),
            "right_7m_near_end": (3300.0, 1050.0),
            "right_6m_arc_far": (3510.0, 560.0),
            "right_6m_straight_far": (3400.0, 850.0),
            "right_6m_arc_center": (3400.0, 1000.0),
            "right_6m_straight_near": (3400.0, 1150.0),
            "right_6m_arc_near": (3510.0, 1440.0),
            "right_4m_mark": (3600.0, 1000.0),
            "right_corner_far": (4000.0, 0.0),
            "right_6m_goalline_far": (4000.0, 250.0),
            "right_6m_sideline_far": (4000.0, 250.0),
            "right_goalpost_far": (4000.0, 850.0),
            "right_goal_center": (4000.0, 1000.0),
            "right_goalpost_near": (4000.0, 1150.0),
            "right_6m_goalline_near": (4000.0, 1750.0),
            "right_6m_sideline_near": (4000.0, 1750.0),
            "right_corner_near": (4000.0, 2000.0),
        }
    ),
    "usfootball": SportConfig(
        name="usfootball",
        display_name="American Football",
        aliases=["usfootball", "american_football", "football_us", "nfl", "football-us"],
        classes=["players", "referees", "ball", "goalpost", "field_marks"],
        ball_class_name="ball",
        goal_class_name="goalpost",
        has_goalkeeper=False,
        max_players_on_court_per_team=11,
        default_court_dims=(10973.0, 4877.0),
        radar_type="usfootball",
        prompt_description=(
            "Detect all objects belonging to these classes in the American football image:\n"
            "- 'players': football players wearing helmets, shoulder pads, and team uniforms.\n"
            "- 'referees': officials wearing black-and-white vertical striped shirts, black/white caps, and black pants.\n"
            "- 'ball': the football (brown leather pigskin).\n"
            "- 'goalpost': the field goal uprights / goalposts.\n"
            "- 'field_marks': yard lines (yard line numbers, 5-yard and 10-yard lines), hash marks, end zone boundaries, sidelines."
        ),
        vlm_landmarks_prompt=(
            "Identify at least 4 to 8 visible American football field landmarks on the turf by combining the yard line "
            "and the lateral zone in the format 'yard_<YARDS>_<SIDE>_<ZONE>' or 'yard_50_<ZONE>':\n"
            "  - <YARDS>: 'goal' (0 yd goal line), '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'\n"
            "  - <SIDE>: 'left' (left half of field) or 'right' (right half of field) — omit for 'yard_50_<ZONE>'\n"
            "  - <ZONE>: 'far_sideline' (top sideline), 'far_numbers', 'far_hash', 'center', 'near_hash', 'near_numbers', 'near_sideline' (bottom sideline).\n"
            "  Examples: 'yard_20_left_near_numbers', 'yard_20_left_far_hash', 'yard_30_left_near_hash', 'yard_50_center'."
        ),
        canonical_landmarks_cm={
            "goal_left_far_sideline": (914.4, 0.0),
            "goal_left_near_sideline": (914.4, 4877.0),
            "yard_50_far_sideline": (5486.5, 0.0),
            "yard_50_far_hash": (5486.5, 1829.0),
            "yard_50_center": (5486.5, 2438.5),
            "yard_50_near_hash": (5486.5, 3048.0),
            "yard_50_near_sideline": (5486.5, 4877.0),
            "goal_right_far_sideline": (10058.4, 0.0),
            "goal_right_near_sideline": (10058.4, 4877.0),
        }
    )
}

def resolve_sport(sport_identifier: Optional[str]) -> SportConfig:
    """
    Resolves a sport identifier or file path to a known SportConfig.
    Prioritizes exact matches, then specific compound names (usfootball) before shorter ones (foot).
    Defaults to 'basket' if unrecognized.
    """
    if not sport_identifier:
        return SPORTS_CONFIGS["basket"]
        
    s_lower = sport_identifier.lower().replace("-", "").replace("_", "")
    
    # 1. Exact match on dictionary keys
    if s_lower in SPORTS_CONFIGS:
        return SPORTS_CONFIGS[s_lower]
        
    # 2. Exact match on aliases
    for sport_key, cfg in SPORTS_CONFIGS.items():
        clean_aliases = [a.replace("-", "").replace("_", "") for a in cfg.aliases]
        if s_lower in clean_aliases:
            return cfg

    # 3. Specific substring priority order: usfootball first, then handball, basket, foot
    for key in ["usfootball", "handball", "basket", "foot"]:
        cfg = SPORTS_CONFIGS[key]
        if key in s_lower:
            return cfg
        for alias in cfg.aliases:
            clean_alias = alias.replace("-", "").replace("_", "")
            if clean_alias in s_lower:
                return cfg
                
    return SPORTS_CONFIGS["basket"]
