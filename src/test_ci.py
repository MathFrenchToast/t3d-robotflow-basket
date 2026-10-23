"""Deterministic CI & local test suite for T3D Multi-Sport Analyzer & FastAPI server."""

import json
import subprocess
import sys
import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.court_homography import CourtHomographyCalculator
from src.sports_config import SPORTS_CONFIGS, resolve_sport
from src.web.app import app


def test_cpu_only_import_without_torch() -> None:
    """Ensure FastAPI app, visualization, and homography import cleanly when torch is absent (Docker slim / CI)."""
    code = (
        "import importlib.abc, sys\n"
        "class BlockHeavyLibs(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path, target=None):\n"
        "        if fullname.split('.')[0] in ('torch', 'umap', 'transformers'):\n"
        "            raise ModuleNotFoundError(f'Blocked {fullname}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, BlockHeavyLibs())\n"
        "from src.web.app import app\n"
        "from src.visualization import SportsAnnotator\n"
        "annot = SportsAnnotator(sport='basket')\n"
        "assert annot.court_config is not None\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Import without torch failed:\n{proc.stderr}"



def test_sports_configs_floor_landmarks() -> None:
    """Verify all supported sports and ensure basketball landmarks are strictly on the floor plane (Z=0)."""
    for sport_key, expected_name in (("basket", "basket"), ("soccer", "foot"), ("handball", "handball")):
        cfg = resolve_sport(sport_key)
        assert cfg.name == expected_name
        assert cfg.default_court_dims is not None
        assert cfg.default_court_dims[0] > 0
        assert cfg.default_court_dims[1] > 0
        assert len(cfg.canonical_landmarks_cm) >= 8

    basket_landmarks = SPORTS_CONFIGS["basket"].canonical_landmarks_cm
    assert "left_basket_ring" not in basket_landmarks
    assert "right_basket_ring" not in basket_landmarks
    assert "left_restricted_arc_apex" in basket_landmarks
    assert "right_restricted_arc_apex" in basket_landmarks


def test_homography_calculator_from_landmarks() -> None:
    """Verify planar homography computation and projection using 4 floor landmarks."""
    calc = CourtHomographyCalculator(sport="basket")
    landmarks = [
        {"name": "right_corner_far", "x": 1250.0, "y": 380.0, "confidence": 0.95},
        {"name": "right_corner_near", "x": 1820.0, "y": 920.0, "confidence": 0.95},
        {"name": "right_ft_line_far", "x": 770.0, "y": 520.0, "confidence": 0.95},
        {"name": "right_ft_line_near", "x": 1050.0, "y": 730.0, "confidence": 0.95},
    ]
    H, inliers_count, matched = calc.compute_from_vlm_landmarks(landmarks, image_size=(1920, 1080))
    assert H is not None
    assert inliers_count >= 4
    assert len(matched) == 4
    assert calc.homography_matrix is not None

    pts_cm = calc.transform_points(np.array([[1350.0, 680.0]], dtype=np.float32))
    assert pts_cm.shape == (1, 2)
    assert 2000.0 <= float(pts_cm[0, 0]) <= 2900.0
    assert 0.0 <= float(pts_cm[0, 1]) <= 1524.0


def test_fastapi_endpoints_and_visualize() -> None:
    """Verify FastAPI health, readiness, UI, documentation pages, OpenAPI, and /api/visualize."""
    with TestClient(app) as client:
        r_health = client.get("/healthz")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "ok"

        r_ready = client.get("/readyz")
        assert r_ready.status_code == 200
        assert r_ready.json()["status"] == "ready"

        r_ui = client.get("/")
        assert r_ui.status_code == 200
        assert "T3D Multi-Sport Image Analyzer" in r_ui.text

        r_doc_hub = client.get("/documentation")
        assert r_doc_hub.status_code == 200
        assert "Documentation Hub" in r_doc_hub.text

        r_doc1 = client.get("/docs/vlm-priors")
        assert r_doc1.status_code == 200
        assert "VLM Probabilistic Priors" in r_doc1.text

        r_doc2 = client.get("/docs/output-schema")
        assert r_doc2.status_code == 200
        assert "X-Processing-Time-Seconds" in r_doc2.text

        r_openapi = client.get("/openapi.json")
        assert r_openapi.status_code == 200
        assert "/api/analyze" in r_openapi.json()["paths"]

        dummy_img = np.zeros((360, 640, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", dummy_img)
        assert ok

        analysis_payload = {
            "sport": "basket",
            "homography": {"computed": True, "visible_court_side": "right"},
            "detections": {
                "players": [
                    {
                        "bbox": [100, 100, 150, 220],
                        "team": 0,
                        "role": "player",
                        "jersey_number": "24",
                        "is_on_court": True,
                        "court_position_cm": [2663.8, 939.2],
                    }
                ],
                "referees": [],
                "ball": [{"bbox": [120, 130, 135, 145], "court_position_cm": [2650.0, 920.0]}],
            },
        }

        r_vis = client.post(
            "/api/visualize",
            files={"image": ("test.jpg", buf.tobytes(), "image/jpeg")},
            data={"analysis_json": json.dumps(analysis_payload), "view_mode": "side_by_side"},
        )
        assert r_vis.status_code == 200
        assert r_vis.headers["content-type"] == "image/jpeg"
        assert len(r_vis.content) > 1000
