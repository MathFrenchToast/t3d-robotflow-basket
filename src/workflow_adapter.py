import os
import time
import json
import base64
import requests
import numpy as np
import cv2
import supervision as sv
from typing import Dict, Any, List, Optional, Tuple, Union
from dotenv import load_dotenv

try:
    from src.sports_config import resolve_sport, SportConfig
except ImportError:
    from sports_config import resolve_sport, SportConfig

load_dotenv()

def read_image_robust(image_input: Union[str, np.ndarray]) -> np.ndarray:
    """
    Decodes an image from a numpy array, path, or binary stream.
    Falls back to ffmpeg pipe decoding if standard cv2.imread fails (e.g. for .avif, .heic).
    """
    if isinstance(image_input, np.ndarray):
        return image_input
    if isinstance(image_input, str) and os.path.exists(image_input):
        frame = cv2.imread(image_input)
        if frame is not None:
            return frame
        # Fallback via ffmpeg pipe for AVIF/HEIC/WEBP
        try:
            import subprocess
            cmd = ["ffmpeg", "-i", image_input, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if proc.returncode == 0 and len(proc.stdout) > 0:
                arr = np.frombuffer(proc.stdout, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
        except Exception:
            pass
        raise ValueError(f"Failed to decode image at {image_input}")
    raise ValueError(f"Invalid image_input: {image_input}")

class WorkflowAdapter:
    """
    Adapter for Roboflow Workflows (VLM / Foundation Models).
    Handles querying the serverless workflow, robust JSON parsing of bounding boxes
    and open-vocabulary classes, separating referees from players, and team classification.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        workspace_name: str = "wpt3dbasket",
        workflow_id: str = "gemini-3-8-flash-object-detection",
        api_url: str = "https://serverless.roboflow.com"
    ):
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        self.api_url = api_url.rstrip("/")

    def call_workflow(
        self,
        image_input: Union[str, np.ndarray],
        classes: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes the Roboflow Workflow via serverless API.
        
        Args:
            image_input: Path to image file, URL, or numpy BGR frame.
            classes: Open-vocabulary classes to detect.
            parameters: Additional workflow parameters.
        """
        if classes is None:
            classes = ["players", "referrees", "ball", "paint", "basket"]

        # If image is a local path or numpy array, encode to base64
        if isinstance(image_input, np.ndarray):
            _, buffer = cv2.imencode('.jpg', image_input)
            img_payload = {"type": "base64", "value": base64.b64encode(buffer).decode("utf-8")}
        elif isinstance(image_input, str) and os.path.exists(image_input):
            with open(image_input, "rb") as f:
                img_payload = {"type": "base64", "value": base64.b64encode(f.read()).decode("utf-8")}
        elif isinstance(image_input, str) and image_input.startswith(("http://", "https://")):
            img_payload = {"type": "url", "value": image_input}
        else:
            raise ValueError(f"Invalid image_input: {image_input}")

        url = f"{self.api_url}/{self.workspace_name}/workflows/{self.workflow_id}"
        payload = {
            "api_key": self.api_key,
            "inputs": {
                "image": img_payload,
                "classes": classes
            }
        }
        if self.google_api_key:
            payload["inputs"]["google_api_key"] = self.google_api_key
            payload["inputs"]["api_key"] = self.google_api_key

        if parameters:
            payload["inputs"].update(parameters)

        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 403:
                error_data = {}
                try:
                    error_data = response.json()
                except Exception:
                    pass
                msg = error_data.get("message", response.text)
                raise PermissionError(
                    f"Roboflow Serverless returned 403: {msg}\n"
                    "Note: If using Foundation Models (Gemini/OpenAI), ensure provider credits or a provider API key "
                    "is configured in the Roboflow Workflow editor under the model block."
                )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Workflow execution failed: {e}") from e

    def call_gemini_direct(
        self,
        image_input: Union[str, np.ndarray],
        classes: Optional[List[str]] = None,
        model_name: str = "gemini-3.8-flash",
        sport: str = "basket",
        mode: str = "high",
        enable_vlm_priors: bool = False
    ) -> Dict[str, Any]:
        """
        Directly queries Google Gemini API for object detection using GOOGLE_API_KEY.
        Supports multi-sport configurations, high mode (mediaResolution + thinkingConfig),
        and optional hybrid VLM priors (--vlm-priors feature flag).
        """
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is not set.")

        sport_cfg = resolve_sport(sport)
        frame = read_image_robust(image_input)

        # In high mode, use 1600 max dim so Gemini captures fine details while respecting token quotas
        max_dim = 1600 if mode == "high" else 1280
        fh, fw = frame.shape[:2]
        if max(fh, fw) > max_dim:
            scale = max_dim / max(fh, fw)
            frame_to_send = cv2.resize(frame, (int(fw * scale), int(fh * scale)))
        else:
            frame_to_send = frame

        # Compress to high-quality JPEG
        quality = 90 if mode == "high" else 85
        _, buffer = cv2.imencode('.jpg', frame_to_send, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        b64_data = base64.b64encode(buffer).decode("utf-8")

        if enable_vlm_priors:
            prompt = (
                f"{sport_cfg.prompt_description}\n\n"
                "Return a single JSON object with two top-level keys: 'objects' and 'court_calibration'.\n"
                "1. 'objects': array of detected objects, where each object has:\n"
                "   - 'box_2d': [ymin, xmin, ymax, xmax] (normalized integers 0 to 1000)\n"
                "   - 'label': exact class name from the requested list\n"
                "   - 'team_id': for 'players' only, integer 0 (light/white/bright uniform) or 1 (dark/contrasting uniform); null for non-players\n"
                "   - 'role_status': for 'players' only, one of 'active_on_court' (playing on court), 'active_inbound' (player on/outside boundary line actively throwing/passing ball inbounds), or 'bench_off_court' (substitute/coach on sideline bench); null for non-players\n"
                "   - 'jersey_number': string if readable on jersey, else null\n"
                "   - 'action_posture': for 'players' only, one of 'standing_running', 'defensive_stance', 'jumping_shoot' (airborne jump shot towards goal), 'jumping_block', or 'diving_falling'\n"
                "   - 'ground_contact_status': for 'players' only, one of 'visible_on_floor' (shoes/feet visible touching the floor), 'airborne' (jumping in the air above the floor), 'truncated_bottom_edge' (lower legs/feet cropped off by the bottom edge of the image frame), or 'occluded_by_player'\n"
                "   - 'ground_point_norm': [y_ground, x_ground] where the player's feet actually contact the floor (normalized 0..1000; if 'truncated_bottom_edge', estimate y_ground > 1000 e.g. 1120..1200 where the unseen shoes would touch the floor below the frame; if 'airborne', point to the vertical shadow on the floor directly beneath the player's torso)\n"
                f"   - 'court_zone_hint': for 'players' only, one of {sport_cfg.court_zones_prompt}\n"
                "2. 'court_calibration': object containing:\n"
                "   - 'visible_court_side': 'left', 'right', 'center', or 'full'\n"
                f"   - 'keypoints': array of visible court floor landmarks ({sport_cfg.vlm_landmarks_prompt}), each with:\n"
                "     - 'name': exact canonical landmark name\n"
                "     - 'point_2d': [y, x] (normalized integers 0 to 1000 pointing to the exact intersection on the floor)\n"
                "Return only valid JSON without markdown backticks."
            )
        else:
            prompt = (
                f"{sport_cfg.prompt_description}\n"
                "For each detected object, return a JSON object with: "
                "'box_2d': [ymin, xmin, ymax, xmax] (normalized integers between 0 and 1000), "
                "'label': the exact class name from the requested list. "
                "Return only a valid JSON array of objects without markdown backticks."
            )

        clean_model = model_name.replace("models/", "")
        canonical_model = f"models/{clean_model}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{canonical_model}:generateContent?key={self.google_api_key}"
        
        gen_config = {
            "response_mime_type": "application/json"
        }
        if mode == "high":
            gen_config["mediaResolution"] = "MEDIA_RESOLUTION_HIGH"
            gen_config["thinkingConfig"] = {
                "thinkingBudget": 1024
            }

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_data
                            }
                        }
                    ]
                }
            ],
            "generationConfig": gen_config
        }

        max_retries = 6
        data = None
        for attempt in range(max_retries):
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code == 429 and ("free_tier_requests" in resp.text or "limit: 20" in resp.text):
                cascade = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
                if clean_model in cascade:
                    idx = cascade.index(clean_model)
                    if idx + 1 < len(cascade):
                        next_model = cascade[idx + 1]
                        print(f"Notice: Daily free tier quota exhausted for {clean_model}. Automatically cascading to {next_model}...")
                        return self.call_gemini_direct(
                            image_input, classes=classes, model_name=next_model,
                            sport=sport, mode=mode, enable_vlm_priors=enable_vlm_priors
                        )
                elif clean_model not in cascade:
                    print(f"Notice: Daily free tier quota exhausted for {clean_model}. Cascading to gemini-3.6-flash...")
                    return self.call_gemini_direct(
                        image_input, classes=classes, model_name="gemini-3.6-flash",
                        sport=sport, mode=mode, enable_vlm_priors=enable_vlm_priors
                    )

            if resp.status_code in [503, 429] and attempt < max_retries - 1:
                retry_header = resp.headers.get("Retry-After")
                if retry_header and retry_header.isdigit():
                    wait_s = int(retry_header) + 1
                elif resp.status_code == 429:
                    wait_s = min(60, 15 * (attempt + 1))  # 15s, 30s, 45s, 60s, 60s
                else:
                    wait_s = 2 ** attempt + 2
                print(f"Notice: Gemini API returned {resp.status_code} ({resp.reason}). Retrying in {wait_s}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            data = resp.json()
            break

        try:
            candidates = data.get("candidates", [])
            text = candidates[0]["content"]["parts"][0]["text"]
            parsed_json = json.loads(text)
            court_calibration = {}
            if isinstance(parsed_json, dict):
                items = parsed_json.get("objects") or parsed_json.get("predictions") or []
                raw_cal = parsed_json.get("court_calibration") or {}
                if isinstance(raw_cal, dict):
                    norm_kps = []
                    for kp in raw_cal.get("keypoints") or []:
                        pt = kp.get("point_2d") or kp.get("point") or []
                        name = kp.get("name") or kp.get("label")
                        if name and isinstance(pt, list) and len(pt) == 2:
                            py, px = float(pt[0]) / 1000.0, float(pt[1]) / 1000.0
                            norm_kps.append({
                                "name": str(name),
                                "x_norm": px,
                                "y_norm": py,
                                "confidence": float(kp.get("confidence", 0.85))
                            })
                    court_calibration = {
                        "visible_court_side": raw_cal.get("visible_court_side", "full"),
                        "keypoints": norm_kps
                    }
            elif isinstance(parsed_json, list):
                items = parsed_json
            else:
                items = []

            predictions = []
            for item in items:
                box = item.get("box_2d", [])
                label = item.get("label", "players")
                if len(box) == 4:
                    ymin, xmin, ymax, xmax = [float(v) / 1000.0 for v in box]
                    w = xmax - xmin
                    h = ymax - ymin
                    x = xmin + w / 2.0
                    y = ymin + h / 2.0
                    pred_obj = {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "confidence": 0.95,
                        "class": label
                    }
                    if "team_id" in item and item["team_id"] is not None:
                        try:
                            pred_obj["vlm_team"] = int(item["team_id"])
                        except (ValueError, TypeError):
                            pass
                    if "role_status" in item and item["role_status"] is not None:
                        pred_obj["vlm_role_status"] = str(item["role_status"])
                    if "jersey_number" in item and item["jersey_number"] is not None:
                        pred_obj["jersey_number"] = str(item["jersey_number"])
                    if "action_posture" in item and item["action_posture"] is not None:
                        pred_obj["action_posture"] = str(item["action_posture"])
                    if "ground_contact_status" in item and item["ground_contact_status"] is not None:
                        pred_obj["ground_contact_status"] = str(item["ground_contact_status"])
                    if "ground_point_norm" in item and isinstance(item["ground_point_norm"], (list, tuple)) and len(item["ground_point_norm"]) == 2:
                        pred_obj["ground_point_norm"] = [float(item["ground_point_norm"][0]), float(item["ground_point_norm"][1])]
                    if "court_zone_hint" in item and item["court_zone_hint"] is not None:
                        pred_obj["court_zone_hint"] = str(item["court_zone_hint"])
                    predictions.append(pred_obj)

            usage_metadata = data.get("usageMetadata", {})
            usage = {
                "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                "candidates_tokens": usage_metadata.get("candidatesTokenCount", 0),
                "thoughts_tokens": usage_metadata.get("thoughtsTokenCount", 0),
                "total_tokens": usage_metadata.get("totalTokenCount", 0),
                "model": clean_model
            }
            res_dict = {"model": {"predictions": predictions}, "usage": usage}
            if court_calibration:
                res_dict["court_calibration"] = court_calibration
            return res_dict
        except Exception as e:
            raise ValueError(f"Failed to parse Gemini direct response: {e}, raw: {data}")

    def call_gpt61_workflow(
        self,
        image_input: Union[str, np.ndarray],
        classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes Roboflow Serverless with GPT-6.1 Sol block, binding OPENAI_API_KEY.
        """
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required to run GPT-6.1 Sol on Roboflow Serverless "
                "(workspace credits are 0, so provider billing requires OPENAI_API_KEY in .env)."
            )

        from inference_sdk import InferenceHTTPClient
        client = InferenceHTTPClient(
            api_url=self.api_url,
            api_key=self.api_key
        )
        spec = {
            "version": "1.0",
            "inputs": [
                {"type": "InferenceImage", "name": "image"},
                {"type": "WorkflowParameter", "name": "classes"},
                {"type": "WorkflowParameter", "name": "openai_api_key"}
            ],
            "steps": [
                {
                    "type": "roboflow_core/open_ai@v7",
                    "name": "model",
                    "images": "$inputs.image",
                    "classes": "$inputs.classes",
                    "api_key": "$inputs.openai_api_key",
                    "task_type": "object-detection",
                    "max_tokens": 16384,
                    "model_version": "gpt-6.1-sol"
                }
            ],
            "outputs": [
                {"type": "JsonField", "name": "model", "coordinates_system": "own", "selector": "$steps.model.*"}
            ]
        }
        if classes is None:
            classes = ["players", "referees", "ball", "basket"]

        res = client.run_workflow(
            specification=spec,
            images={"image": image_input},
            parameters={
                "classes": classes,
                "openai_api_key": self.openai_api_key
            }
        )
        if isinstance(res, list) and len(res) > 0:
            return res[0]
        return res

    def call_openai_direct(
        self,
        image_input: Union[str, np.ndarray],
        classes: Optional[List[str]] = None,
        model_name: str = "gpt-6.1-sol"
    ) -> Dict[str, Any]:
        """
        Directly queries OpenAI API for object detection using OPENAI_API_KEY.
        Matches Gemini direct output schema for unified parsing.
        """
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        if classes is None:
            classes = ["players", "referees", "ball", "basket"]

        if isinstance(image_input, np.ndarray):
            frame = image_input
        elif isinstance(image_input, str) and os.path.exists(image_input):
            frame = cv2.imread(image_input)
            if frame is None:
                raise ValueError(f"Failed to read image at {image_input}")
        else:
            raise ValueError(f"Invalid image_input for OpenAI direct: {image_input}")

        max_dim = 1280
        fh, fw = frame.shape[:2]
        if max(fh, fw) > max_dim:
            scale = max_dim / max(fh, fw)
            frame_to_send = cv2.resize(frame, (int(fw * scale), int(fh * scale)))
        else:
            frame_to_send = frame

        _, buffer = cv2.imencode('.jpg', frame_to_send, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        b64_data = base64.b64encode(buffer).decode("utf-8")

        prompt = (
            "Detect all objects belonging to these classes in the basketball image:\n"
            "- 'players': active basketball players wearing team uniforms (jerseys and shorts) on the court.\n"
            "- 'referees': match referees / officials (wearing striped or grey referee shirts and black pants, including referees on the sidelines).\n"
            "- 'ball': the basketball.\n"
            "- 'basket': the rim/hoop/net of the basketball goal.\n"
            "For each detected object, return a JSON array under the key 'predictions' with objects containing: "
            "'box_2d': [ymin, xmin, ymax, xmax] (normalized integers between 0 and 1000), "
            "'label': the exact class name from the requested list. "
            "Return only valid JSON matching this schema."
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_data}"
                            }
                        }
                    ]
                }
            ],
            "response_format": {"type": "json_object"}
        }

        url = "https://api.openai.com/v1/chat/completions"
        max_retries = 3
        data = None
        for attempt in range(max_retries):
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            if resp.status_code in [503, 429] and attempt < max_retries - 1:
                wait_s = 2 ** attempt + 1
                print(f"Notice: OpenAI API returned {resp.status_code}. Retrying in {wait_s}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            data = resp.json()
            break

        try:
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            items = parsed.get("predictions", parsed if isinstance(parsed, list) else [])
            predictions = []
            for item in items:
                box = item.get("box_2d", item.get("bbox", []))
                label = item.get("label", item.get("class", "players"))
                if len(box) == 4:
                    ymin, xmin, ymax, xmax = [float(v) / 1000.0 if max(box) > 1.0 else float(v) for v in box]
                    w = xmax - xmin
                    h = ymax - ymin
                    x = xmin + w / 2.0
                    y = ymin + h / 2.0
                    predictions.append({
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "confidence": 0.95,
                        "class": label
                    })
            return {"model": {"predictions": predictions}}
        except Exception as e:
            raise ValueError(f"Failed to parse OpenAI direct response: {e}, raw content: {data}")

    def call_openrouter(
        self,
        image_input: Union[str, np.ndarray],
        classes: Optional[List[str]] = None,
        model_name: str = "google/gemini-3.8-flash",
        sport: str = "basket",
        mode: str = "high",
        enable_vlm_priors: bool = False
    ) -> Dict[str, Any]:
        """
        Queries OpenRouter unified API (https://openrouter.ai/api/v1/chat/completions)
        using OPENROUTER_API_KEY. Supports any vision-capable model from Google, OpenAI,
        Anthropic, Qwen, Mistral, or xAI, in both standard and --vlm-priors modes.
        """
        if not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set in environment or .env file. "
                "Please set OPENROUTER_API_KEY=sk-or-v1-... in .env."
            )

        sport_cfg = resolve_sport(sport)
        if classes is None:
            classes = sport_cfg.classes

        frame = read_image_robust(image_input)
        fh, fw = frame.shape[:2]

        # Normalize model_name if passed without provider prefix
        or_model = model_name.strip()
        if "/" not in or_model:
            low = or_model.lower()
            if low.startswith("gemini"):
                or_model = f"google/{or_model}"
            elif low.startswith(("gpt-", "o1", "o3", "o4")):
                or_model = f"openai/{or_model}"
            elif low.startswith("claude"):
                or_model = f"anthropic/{or_model}"
            elif low.startswith("pixtral") or low.startswith("mistral"):
                or_model = f"mistralai/{or_model}"
            elif low.startswith("qwen"):
                or_model = f"qwen/{or_model}"
            elif low.startswith("deepseek"):
                or_model = f"deepseek/{or_model}"
            elif low.startswith("grok"):
                or_model = f"x-ai/{or_model}"

        is_openai = or_model.startswith("openai/")
        # OpenAI vision models natively tile 16:9 images onto a 1536x864 canvas (48x27 patches of 32px)
        if is_openai:
            max_dim = 1536 if mode == "high" else 1280
        else:
            max_dim = 1920 if mode == "high" else 1280

        if max(fh, fw) > max_dim:
            scale = max_dim / max(fh, fw)
            frame_to_send = cv2.resize(frame, (int(round(fw * scale)), int(round(fh * scale))))
        else:
            frame_to_send = frame

        send_h, send_w = frame_to_send.shape[:2]
        jpeg_quality = 92 if mode == "high" else 85
        _, buffer = cv2.imencode('.jpg', frame_to_send, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        b64_data = base64.b64encode(buffer).decode("utf-8")

        # When --vlm-priors is active, court keypoints handle field geometry so we omit redundant field_marks boxes
        base_prompt_desc = sport_cfg.prompt_description
        if enable_vlm_priors:
            filtered_lines = [
                line for line in base_prompt_desc.splitlines()
                if "'field_marks'" not in line
            ]
            base_prompt_desc = "\n".join(filtered_lines)

        if enable_vlm_priors:
            if is_openai:
                coord_instructions = (
                    f"CRITICAL Coordinates System (Image size is {send_w}x{send_h} pixels):\n"
                    f"- All bounding boxes 'box_2d' MUST use integer pixel coordinates [xmin, ymin, xmax, ymax] "
                    f"within the {send_w}x{send_h} image grid (where [0, 0] is top-left and [{send_w}, {send_h}] is bottom-right).\n"
                    f"- All court keypoints 'point_2d' MUST use integer pixel coordinates [x, y] "
                    f"within the {send_w}x{send_h} image grid (where x is horizontal 0..{send_w} and y is vertical 0..{send_h}).\n\n"
                )
                box_example = "[xmin, ymin, xmax, ymax]"
                pt_example = "[x, y]"
            else:
                coord_instructions = (
                    "CRITICAL Coordinates System:\n"
                    "- All bounding boxes 'box_2d' MUST use normalized integer coordinates [ymin, xmin, ymax, xmax] scaled from 0 to 1000 "
                    "(where [0, 0] is top-left and [1000, 1000] is bottom-right of the image).\n"
                    "- All court keypoints 'point_2d' MUST use normalized integer coordinates [y, x] scaled from 0 to 1000 "
                    "(where y is vertical top-to-bottom 0..1000 and x is horizontal left-to-right 0..1000).\n\n"
                )
                box_example = "[ymin, xmin, ymax, xmax]"
                pt_example = "[y, x]"

            gk_instruction = ""
            gk_example = ""
            if sport_cfg.has_goalkeeper:
                gk_instruction = (
                    "   - For goalkeepers, use label 'goalkeepers' and set 'team_id' (0 or 1) to match the team defending that goal "
                    "(determined by court side / defending teammates, since goalkeepers wear a distinct jersey color).\n"
                )
                gk_example = f'    {{"box_2d": {box_example}, "label": "goalkeepers", "team_id": 0, "role_status": "active_on_court", "jersey_number": "1", "action_posture": "defensive_stance"}},\n'

            prompt = (
                f"{base_prompt_desc}\n\n"
                f"{coord_instructions}"
                "HYBRID VLM PROBABILISTIC PRIORS (Semantic & Biomechanical Enrichment):\n"
                "1. Exhaustively detect EVERY visible player on the field/court (both teams, including closely packed players on the line of scrimmage/clusters, corners, and edges) with label 'players', plus all visible 'goalkeepers' (if applicable) and 'referees', and provide for each player/goalkeeper:\n"
                "   - 'team_id': integer 0 (light/white/home kit) or 1 (dark/colored/away kit).\n"
                f"{gk_instruction}"
                "   - 'role_status': 'active_on_court', 'active_inbound', or 'bench_off_court'.\n"
                "   - 'jersey_number': string with visible digits (e.g. '22') or null.\n"
                "   - 'action_posture': optional, 'standing_running', 'defensive_stance', or 'jumping_shoot' (if jumping in air).\n"
                f"2. Court Calibration Landmarks:\n{sport_cfg.vlm_landmarks_prompt}\n\n"
                "Return ONLY a valid JSON object matching this exact structure (no markdown fences):\n"
                "{\n"
                '  "objects": [\n'
                f'    {{"box_2d": {box_example}, "label": "players", "team_id": 0, "role_status": "active_on_court", "jersey_number": "10", "action_posture": "standing_running"}},\n'
                f"{gk_example}"
                f'    {{"box_2d": {box_example}, "label": "referees"}}\n'
                "  ],\n"
                '  "court_calibration": {\n'
                '    "visible_court_side": "left|right|center|full",\n'
                '    "keypoints": [\n'
                f'      {{"name": "<canonical_landmark_name>", "point_2d": {pt_example}}}\n'
                "    ]\n"
                "  }\n"
                "}"
            )
        else:
            if is_openai:
                prompt = (
                    f"{sport_cfg.prompt_description}\n"
                    f"For each detected object in this {send_w}x{send_h} image, return a JSON object with a key 'objects' containing a JSON array of objects with: "
                    f"'box_2d': [xmin, ymin, xmax, ymax] (integer pixel coordinates within 0..{send_w} horizontally and 0..{send_h} vertically), "
                    "'label': the exact class name from the requested list. "
                    "Return only valid JSON."
                )
            else:
                prompt = (
                    f"{sport_cfg.prompt_description}\n"
                    "For each detected object, return a JSON object with a key 'objects' containing a JSON array of objects with: "
                    "'box_2d': [ymin, xmin, ymax, xmax] (normalized integers between 0 and 1000), "
                    "'label': the exact class name from the requested list. "
                    "Return only valid JSON."
                )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/t3d-sports-analytics",
            "X-Title": "T3D Multi-Sport 2D Homography Pipeline"
        }
        payload: Dict[str, Any] = {
            "model": or_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_data}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 16384,
            "response_format": {"type": "json_object"}
        }

        url = "https://openrouter.ai/api/v1/chat/completions"
        max_retries = 3
        data = None
        for attempt in range(max_retries):
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 400 and "response_format" in payload:
                # Some providers on OpenRouter do not accept response_format={"type": "json_object"}
                payload.pop("response_format", None)
                continue
            if resp.status_code in [503, 502, 429] and attempt < max_retries - 1:
                wait_s = 3 * (attempt + 1)
                print(f"Notice: OpenRouter API ({or_model}) returned {resp.status_code}. Retrying in {wait_s}s...")
                time.sleep(wait_s)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter API error ({resp.status_code}) for model '{or_model}': {resp.text[:500]}")
            data = resp.json()
            break

        try:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, str):
                cleaned = content.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip().startswith("```"):
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()
                # Extract first JSON object or array if surrounded by commentary
                if not cleaned.startswith(("{", "[")):
                    start_obj = cleaned.find("{")
                    start_arr = cleaned.find("[")
                    starts = [i for i in (start_obj, start_arr) if i != -1]
                    if starts:
                        cleaned = cleaned[min(starts):]
                parsed = json.loads(cleaned)
            else:
                parsed = content

            court_calibration = None
            if isinstance(parsed, dict):
                court_calibration = parsed.get("court_calibration")
                items = parsed.get("objects", parsed.get("detections", parsed.get("predictions", [])))
            elif isinstance(parsed, list):
                items = parsed
            else:
                items = []

            if is_openai and isinstance(court_calibration, dict):
                for kp in court_calibration.get("keypoints") or []:
                    if "point_2d" in kp and isinstance(kp["point_2d"], (list, tuple)) and len(kp["point_2d"]) == 2:
                        px_val, py_val = float(kp["point_2d"][0]), float(kp["point_2d"][1])
                        if max(abs(px_val), abs(py_val)) > 1.0:
                            kp["x_norm"] = px_val / float(send_w)
                            kp["y_norm"] = py_val / float(send_h)
                        else:
                            kp["x_norm"] = px_val
                            kp["y_norm"] = py_val

            predictions = []
            for item in items:
                box = item.get("box_2d", item.get("bbox", []))
                label = item.get("label", item.get("class", "players"))
                if len(box) == 4:
                    if is_openai:
                        raw_x1, raw_y1, raw_x2, raw_y2 = [float(v) for v in box]
                        if max(abs(raw_x1), abs(raw_y1), abs(raw_x2), abs(raw_y2)) <= 1.0:
                            xmin, ymin, xmax, ymax = raw_x1, raw_y1, raw_x2, raw_y2
                        else:
                            xmin = raw_x1 / float(send_w)
                            ymin = raw_y1 / float(send_h)
                            xmax = raw_x2 / float(send_w)
                            ymax = raw_y2 / float(send_h)
                    else:
                        ymin, xmin, ymax, xmax = [float(v) / 1000.0 if max(box) > 1.0 else float(v) for v in box]
                    w = xmax - xmin
                    h = ymax - ymin
                    x = xmin + w / 2.0
                    y = ymin + h / 2.0
                    pred_obj: Dict[str, Any] = {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "confidence": 0.95,
                        "class": label
                    }
                    if "team_id" in item and item["team_id"] is not None:
                        try:
                            pred_obj["vlm_team"] = int(item["team_id"])
                        except (ValueError, TypeError):
                            pass
                    if "role_status" in item and item["role_status"] is not None:
                        pred_obj["vlm_role_status"] = str(item["role_status"])
                    if "jersey_number" in item and item["jersey_number"] is not None:
                        pred_obj["jersey_number"] = str(item["jersey_number"])
                    if "action_posture" in item and item["action_posture"] is not None:
                        pred_obj["action_posture"] = str(item["action_posture"])
                    if "ground_contact_status" in item and item["ground_contact_status"] is not None:
                        pred_obj["ground_contact_status"] = str(item["ground_contact_status"])
                    if "ground_point_norm" in item and isinstance(item["ground_point_norm"], (list, tuple)) and len(item["ground_point_norm"]) == 2:
                        pred_obj["ground_point_norm"] = [float(item["ground_point_norm"][0]), float(item["ground_point_norm"][1])]
                    if "court_zone_hint" in item and item["court_zone_hint"] is not None:
                        pred_obj["court_zone_hint"] = str(item["court_zone_hint"])
                    predictions.append(pred_obj)

            usage_raw = data.get("usage", {})
            details = usage_raw.get("completion_tokens_details") or {}
            reasoning_tok = int(details.get("reasoning_tokens", 0) or 0)
            comp_tok = int(usage_raw.get("completion_tokens", 0) or 0)
            cand_tok = max(0, comp_tok - reasoning_tok)
            usage = {
                "prompt_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
                "candidates_tokens": cand_tok,
                "thoughts_tokens": reasoning_tok,
                "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
                "cost_usd": usage_raw.get("cost"),
                "model": data.get("model", or_model)
            }
            res_dict: Dict[str, Any] = {"model": {"predictions": predictions}, "usage": usage}
            if court_calibration:
                res_dict["court_calibration"] = court_calibration
            return res_dict
        except Exception as e:
            raise ValueError(f"Failed to parse OpenRouter response for '{or_model}': {e}, raw: {data}")

    def parse_court_calibration(
        self,
        workflow_output: Any,
        image_shape: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Extracts and denormalizes VLM court calibration priors (visible_court_side and keypoints)
        into pixel coordinates matching image_shape (height, width).
        """
        if isinstance(workflow_output, list) and len(workflow_output) > 0:
            workflow_output = workflow_output[0]
        if not isinstance(workflow_output, dict):
            return {"visible_court_side": "full", "keypoints": []}

        cal = workflow_output.get("court_calibration")
        img_h, img_w = image_shape if image_shape is not None else (1, 1)
        px_keypoints = []

        if isinstance(cal, dict):
            raw_kps = cal.get("keypoints") or []
            vis_side = cal.get("visible_court_side", "full")
        elif "landmarks" in workflow_output and isinstance(workflow_output["landmarks"], list):
            raw_kps = [lm for lm in workflow_output["landmarks"] if lm.get("source") in ("vlm_prior", None)]
            vis_side = workflow_output.get("homography", {}).get("visible_court_side", "full")
        else:
            return {"visible_court_side": "full", "keypoints": []}

        for kp in raw_kps:
            name = kp.get("name") or kp.get("label")
            if not name:
                continue
            if "x_norm" in kp and "y_norm" in kp:
                x_px = float(kp["x_norm"]) * img_w
                y_px = float(kp["y_norm"]) * img_h
            elif "point_2d" in kp and isinstance(kp["point_2d"], (list, tuple)) and len(kp["point_2d"]) == 2:
                py, px = float(kp["point_2d"][0]), float(kp["point_2d"][1])
                y_norm = py / 1000.0 if max(abs(py), abs(px)) > 1.0 else py
                x_norm = px / 1000.0 if max(abs(py), abs(px)) > 1.0 else px
                x_px = x_norm * img_w
                y_px = y_norm * img_h
            elif "x" in kp and "y" in kp:
                x_px = float(kp["x"])
                y_px = float(kp["y"])
            else:
                continue
            px_keypoints.append({
                "name": str(name),
                "x": x_px,
                "y": y_px,
                "confidence": float(kp.get("confidence", 0.85))
            })

        return {
            "visible_court_side": vis_side,
            "keypoints": px_keypoints
        }

    def parse_predictions(
        self,
        workflow_output: Any,
        image_shape: Optional[Tuple[int, int]] = None
    ) -> Dict[str, sv.Detections]:
        """
        Parses Roboflow Workflow output JSON into standardized supervision.Detections by category.
        Preserves optional VLM priors (vlm_team, vlm_role_status, jersey_number, action_posture,
        ground_contact_status, ground_point_norm, court_zone_hint) in Detections.data.
        """
        # Unpack nested workflow response structure
        raw_preds = []
        if isinstance(workflow_output, list) and len(workflow_output) > 0:
            workflow_output = workflow_output[0]

        if isinstance(workflow_output, dict):
            if "model" in workflow_output and isinstance(workflow_output["model"], dict):
                raw_preds = workflow_output["model"].get("predictions", [])
            elif "predictions" in workflow_output:
                raw_preds = workflow_output["predictions"]
            elif "result" in workflow_output and isinstance(workflow_output["result"], dict):
                raw_preds = workflow_output["result"].get("predictions", [])
            elif "detections" in workflow_output and isinstance(workflow_output["detections"], dict):
                for cat in ["players", "goalkeepers", "referees", "ball", "basket", "goal_or_basket", "field_marks"]:
                    c_name = "basket" if cat == "goal_or_basket" else cat
                    for item in workflow_output["detections"].get(cat, []):
                        raw_preds.append({
                            "bbox": item.get("bbox"),
                            "confidence": item.get("confidence", 1.0),
                            "class": c_name,
                            "vlm_team": item.get("vlm_team", item.get("vlm_team_prior", item.get("team"))),
                            "vlm_role_status": item.get("vlm_role_status"),
                            "jersey_number": item.get("jersey_number"),
                            "action_posture": item.get("action_posture"),
                            "ground_contact_status": item.get("ground_contact_status"),
                            "ground_point_norm": item.get("ground_point_norm"),
                            "court_zone_hint": item.get("court_zone_hint")
                        })

        # Categorized lists
        boxes_by_cat = {
            "players": [],
            "goalkeepers": [],
            "referees": [],
            "ball": [],
            "basket": [],
            "field_marks": []
        }
        conf_by_cat = {k: [] for k in boxes_by_cat}
        meta_keys = [
            "vlm_team",
            "vlm_role_status",
            "jersey_number",
            "action_posture",
            "ground_contact_status",
            "ground_point_norm",
            "court_zone_hint"
        ]
        cat_meta = {
            "players": {k: [] for k in meta_keys},
            "goalkeepers": {k: [] for k in meta_keys}
        }

        img_h, img_w = image_shape if image_shape is not None else (1, 1)

        for pred in raw_preds:
            box = None
            conf = float(pred.get("confidence", 1.0))
            class_name = str(pred.get("class") or pred.get("label") or "").strip().lower()

            if "x" in pred and "y" in pred and "width" in pred and "height" in pred:
                x, y, w, h = float(pred["x"]), float(pred["y"]), float(pred["width"]), float(pred["height"])
                if image_shape is not None and x <= 1.0 and y <= 1.0 and w <= 1.0 and h <= 1.0:
                    x *= img_w
                    y *= img_h
                    w *= img_w
                    h *= img_h
                x1 = x - w / 2.0
                y1 = y - h / 2.0
                x2 = x + w / 2.0
                y2 = y + h / 2.0
                box = [x1, y1, x2, y2]
            elif "box" in pred or "bbox" in pred:
                b = pred.get("box") or pred.get("bbox")
                if len(b) == 4:
                    box = [float(coord) for coord in b]

            if box is None:
                continue

            target_cat = "players"
            if any(term in class_name for term in ["goalkeeper", "gardien", "goalie", "keeper"]):
                target_cat = "goalkeepers"
            elif any(term in class_name for term in ["referee", "referree", "arbitre", "linesman", "judge", "official"]):
                target_cat = "referees"
            elif any(term in class_name for term in ["player", "joueur", "person", "attacker", "defender", "wing"]):
                target_cat = "players"
            elif any(term in class_name for term in ["ball", "balloon", "ballon"]):
                target_cat = "ball"
            elif any(term in class_name for term in ["basket", "rim", "panier", "hoop", "goal", "cage", "but", "goalpost", "poteau"]):
                target_cat = "basket"
            elif any(term in class_name for term in ["paint", "mark", "field", "court", "raquette", "line", "ligne", "yard"]):
                target_cat = "field_marks"

            boxes_by_cat[target_cat].append(box)
            conf_by_cat[target_cat].append(conf)
            if target_cat in ("players", "goalkeepers"):
                m_dict = cat_meta[target_cat]
                vt = pred.get("vlm_team")
                m_dict["vlm_team"].append(int(vt) if vt in (0, 1, "0", "1") else -1)
                m_dict["vlm_role_status"].append(str(pred.get("vlm_role_status") or "active_on_court"))
                m_dict["jersey_number"].append(str(pred.get("jersey_number") or ""))
                m_dict["action_posture"].append(str(pred.get("action_posture") or "standing_running"))
                m_dict["ground_contact_status"].append(str(pred.get("ground_contact_status") or "visible_on_floor"))
                gpn = pred.get("ground_point_norm")
                if isinstance(gpn, (list, tuple, np.ndarray)) and len(gpn) == 2:
                    m_dict["ground_point_norm"].append((float(gpn[0]), float(gpn[1])))
                else:
                    m_dict["ground_point_norm"].append(None)
                m_dict["court_zone_hint"].append(str(pred.get("court_zone_hint") or ""))

        # Filter out degenerate/tiny noise boxes for players while preserving upright profile players in wide shots
        if len(boxes_by_cat["players"]) > 0:
            min_bw = max(14.0, img_w * 0.010) if image_shape is not None else 14.0
            min_bh = max(22.0, img_h * 0.020) if image_shape is not None else 22.0
            valid_p_indices = []
            for i, box in enumerate(boxes_by_cat["players"]):
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                if bw >= min_bw and bh >= min_bh:
                    valid_p_indices.append(i)
            boxes_by_cat["players"] = [boxes_by_cat["players"][i] for i in valid_p_indices]
            conf_by_cat["players"] = [conf_by_cat["players"][i] for i in valid_p_indices]
            for k in meta_keys:
                cat_meta["players"][k] = [cat_meta["players"][k][i] for i in valid_p_indices]

        # Convert each category to sv.Detections
        detections_by_cat = {}
        for cat, boxes in boxes_by_cat.items():
            if len(boxes) > 0:
                xyxy = np.array(boxes, dtype=np.float32)
                confidence = np.array(conf_by_cat[cat], dtype=np.float32)
                data_dict = {}
                if cat in ("players", "goalkeepers"):
                    m_dict = cat_meta[cat]
                    gpn_arr = np.empty(len(xyxy), dtype=object)
                    for idx_g, g_val in enumerate(m_dict["ground_point_norm"]):
                        gpn_arr[idx_g] = list(g_val) if g_val is not None else None
                    data_dict = {
                        "vlm_team": np.array(m_dict["vlm_team"], dtype=int),
                        "vlm_role_status": np.array(m_dict["vlm_role_status"], dtype=object),
                        "jersey_number": np.array(m_dict["jersey_number"], dtype=object),
                        "action_posture": np.array(m_dict["action_posture"], dtype=object),
                        "ground_contact_status": np.array(m_dict["ground_contact_status"], dtype=object),
                        "ground_point_norm": gpn_arr,
                        "court_zone_hint": np.array(m_dict["court_zone_hint"], dtype=object)
                    }
                detections_by_cat[cat] = sv.Detections(
                    xyxy=xyxy,
                    confidence=confidence,
                    class_id=np.zeros(len(xyxy), dtype=int),
                    data=data_dict
                )
            else:
                detections_by_cat[cat] = sv.Detections.empty()

        # Apply NMS on players (0.75 preserves closely packed linemen/scrimmage while removing duplicate boxes)
        if len(detections_by_cat["players"]) > 0:
            detections_by_cat["players"] = detections_by_cat["players"].with_nms(threshold=0.75)

        # Cross-class suppression: if a player overlaps heavily with a referee, referee label wins
        if len(detections_by_cat["referees"]) > 0 and len(detections_by_cat["players"]) > 0:
            ious = sv.box_iou_batch(detections_by_cat["players"].xyxy, detections_by_cat["referees"].xyxy)
            keep_non_ref = ious.max(axis=1) < 0.5
            detections_by_cat["players"] = detections_by_cat["players"][keep_non_ref]

        # Cross-class suppression: if a player overlaps heavily with a goalkeeper, goalkeeper label wins
        if len(detections_by_cat.get("goalkeepers", [])) > 0 and len(detections_by_cat["players"]) > 0:
            ious_gk = sv.box_iou_batch(detections_by_cat["players"].xyxy, detections_by_cat["goalkeepers"].xyxy)
            keep_non_gk = ious_gk.max(axis=1) < 0.5
            detections_by_cat["players"] = detections_by_cat["players"][keep_non_gk]

        return detections_by_cat

    def _extract_dominant_jersey_color(self, torso_crop: np.ndarray) -> np.ndarray:
        """
        Extracts dominant jersey color from torso crop using pixel clustering,
        selecting the majority cluster representing the jersey fabric over background.
        """
        if torso_crop is None or torso_crop.size == 0:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        small = cv2.resize(torso_crop, (30, 40))
        pixels = small.reshape(-1, 3).astype(np.float32)

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, n_init=5, random_state=42).fit(pixels)
        counts = np.bincount(km.labels_)
        best_idx = int(np.argmax(counts))
        return km.cluster_centers_[best_idx]

    def classify_player_teams(
        self,
        frame: np.ndarray,
        player_detections: sv.Detections,
        court_mask: Optional[np.ndarray] = None,
        max_players_per_team: int = 5,
        enable_vlm_priors: bool = False
    ) -> List[Optional[int]]:
        """
        Classifies detected players into two teams using dominant jersey color extraction.
        When enable_vlm_priors=True, performs Bayesian consensus fusing VLM semantic team priors
        with local K-Means/HSV color clustering, weighted by player occlusion IoU.
        """
        num_players = len(player_detections)
        if num_players == 0:
            return []

        frame_h, frame_w, _ = frame.shape
        jersey_colors = []

        # Compute pairwise overlap to identify occluded players
        boxes = player_detections.xyxy
        ious = sv.box_iou_batch(boxes, boxes) if len(boxes) > 1 else np.zeros((1, 1))
        np.fill_diagonal(ious, 0.0)
        max_ious = ious.max(axis=1) if len(boxes) > 1 else np.zeros(num_players)

        # Pass 1: Extract candidate clusters for each player
        player_candidates = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                player_candidates.append(None)
                continue

            h, w, _ = crop.shape
            torso = crop[int(h * 0.18):int(h * 0.55), int(w * 0.20):int(w * 0.80)]
            if torso.size == 0:
                torso = crop

            small = cv2.resize(torso, (30, 40))
            pixels = small.reshape(-1, 3).astype(np.float32)

            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=2, n_init=5, random_state=42).fit(pixels)
            counts = np.bincount(km.labels_)
            weights = counts / len(pixels)
            player_candidates.append((km.cluster_centers_[0], km.cluster_centers_[1], weights))

        # Detect if match features a white/light jersey team
        light_majority_players = sum(
            1 for item in player_candidates
            if item is not None and (
                (item[0].mean() >= 155 and item[2][0] >= 0.50) or
                (item[1].mean() >= 155 and item[2][1] >= 0.50)
            )
        )
        has_white_team = (light_majority_players >= 2)

        def is_white_jersey(c, weight):
            bgr_u8 = np.uint8([[c.round()]])
            hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)[0, 0]
            b, g, r = float(c[0]), float(c[1]), float(c[2])
            return (hsv[1] <= 28 and abs(r - b) <= 15 and hsv[2] > 150 and weight >= 0.25)

        for i, item in enumerate(player_candidates):
            if item is None:
                jersey_colors.append(np.array([128.0, 128.0, 128.0], dtype=np.float32))
                continue
            c0, c1, w = item
            if has_white_team:
                c0_w = is_white_jersey(c0, w[0])
                c1_w = is_white_jersey(c1, w[1])
                if c0_w and not c1_w:
                    chosen = c0
                elif c1_w and not c0_w:
                    chosen = c1
                elif c0_w and c1_w:
                    chosen = c0 if np.mean(c0) > np.mean(c1) else c1
                else:
                    chosen = c0 if w[0] >= w[1] else c1
            else:
                chosen = c0 if w[0] >= w[1] else c1
            jersey_colors.append(chosen)

        if num_players < 2:
            return [0] * num_players

        from sklearn.cluster import KMeans
        km_teams = KMeans(n_clusters=2, n_init=10, random_state=42).fit(jersey_colors)
        centers = km_teams.cluster_centers_

        c0, c1 = centers[0], centers[1]
        score_0 = float(np.mean(c0)) + float(c0[0] - c0[2])
        score_1 = float(np.mean(c1)) + float(c1[0] - c1[2])

        if score_0 > score_1:
            team0_center, team1_center = c0, c1
        else:
            team0_center, team1_center = c1, c0

        axis = team0_center - team1_center
        axis_dist = float(np.linalg.norm(axis))
        axis_norm = axis / (axis_dist + 1e-6)
        scores = np.array([float(np.dot(c, axis_norm)) for c in jersey_colors])
        midpoint = float(np.dot(team0_center, axis_norm) + np.dot(team1_center, axis_norm)) / 2.0

        kmeans_assignments = [0 if s >= midpoint else 1 for s in scores]

        # Hybrid Bayesian Consensus when --vlm-priors is enabled
        if enable_vlm_priors and player_detections.data and "vlm_team" in player_detections.data:
            vlm_teams = np.array(player_detections.data["vlm_team"], dtype=int)
            valid_vlm = (vlm_teams == 0) | (vlm_teams == 1)
            if np.any(valid_vlm):
                # Align VLM polarity (0/1) with K-Means polarity (0/1) on unoccluded players (IoU < 0.15)
                unoccluded = valid_vlm & (max_ious < 0.15)
                ref_mask = unoccluded if np.sum(unoccluded) >= 2 else valid_vlm
                same_polarity = sum(1 for idx in np.where(ref_mask)[0] if vlm_teams[idx] == kmeans_assignments[idx])
                flip_polarity = (same_polarity < (np.sum(ref_mask) / 2.0))

                tau = max(8.0, 0.25 * axis_dist)
                final_assignments = []
                for i in range(num_players):
                    p_kmeans_t0 = float(1.0 / (1.0 + np.exp(-np.clip((scores[i] - midpoint) / tau, -10.0, 10.0))))
                    if valid_vlm[i]:
                        aligned_vt = (1 - vlm_teams[i]) if flip_polarity else vlm_teams[i]
                        p_vlm_t0 = 0.92 if aligned_vt == 0 else 0.08
                        # Strong semantic weight on polarity-aligned VLM team prior (handles large back jersey numbers & occlusions)
                        alpha = 0.85 if max_ious[i] >= 0.15 else 0.65
                        p_final_t0 = alpha * p_vlm_t0 + (1.0 - alpha) * p_kmeans_t0
                        final_assignments.append(0 if p_final_t0 >= 0.50 else 1)
                    else:
                        final_assignments.append(kmeans_assignments[i])
                return final_assignments

        return kmeans_assignments


