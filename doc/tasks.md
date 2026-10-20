# Tasks

- [x] Project initialization with `uv`
- [x] Dependency configuration
- [x] Implement `src/config.py`
- [x] Implement `src/models.py`
- [x] Implement `src/tracking.py`
- [x] Implement `src/visualization.py`
- [x] Implement `src/pipeline.py`
- [x] Implement `src/main.py`
- [x] Create `README.md` and `.env.example`
- [x] Verification and testing

## Roboflow VLM Workflow & Partial Court Homography
- [x] Implement `src/workflow_adapter.py` (workflow client & parser for predictions, boxes, classes)
- [x] Implement `src/court_homography.py` (robust homography on partial field marks, paint, basket, free-throw)
- [x] Integrate team classification filtering out referees
- [x] Fix team color classification (chromaticity B-R separation, deterministic T0 Blue / T1 Dark)
- [x] Correct misclassified T1 player running across blue paint floor via majority pixel clustering
- [x] Recover occluded T0 player hidden behind T1 player in the paint (establishing exact 5v5 on court)
- [x] Enforce strict on-court vs bench separation (`is_on_court`) and 5-player active limit per team on radar
- [x] Update `src/analyze_image.py` to support workflow-driven single image analysis
- [x] Validate and test end-to-end with `test/basket3.png`
- [x] Validate and test end-to-end with `test/basketball-frame.jpg` (left half-court framing, Warriors vs Clippers)
- [x] Support arbitrary player counts ($N \ne 10$) without forcing 5v5 quotas
- [x] Enhance VLM prompt to distinguish sideline referees (striped/grey shirt, black pants) from players
- [x] Resolve inbound pass duel on sideline: correctly classify defender as T0 (on-court) and inbounder as T1 (exact 5v5)
- [x] Investigate Gemini 4 Argon availability and API quota limits

## GPT-6.1 Sol & Warm Latency Benchmarking (FastAPI Lifespan Integration)
- [x] Investigate Roboflow Serverless execution for GPT-6.1 Sol (`gpt-6-1-sol-object-detection`)
- [x] Diagnose 403 Forbidden error on step `model` (missing provider API key / 0 Roboflow credits)
- [x] Validate inline workflow specification binding `openai_api_key` to `roboflow_core/open_ai@v7`
- [x] Isolate and measure warm request execution time (excluding FastAPI lifespan startup):
  - Initialization (Court model + annotator): ~1.88s (mutualized in FastAPI startup)
  - Gemini 3.8 Flash VLM inference: ~11.4s - 12.8s
  - Local Court Homography: ~16ms - 100ms
  - Local Team Classification: ~32ms - 60ms
  - Total warm latency: ~11.4s - 12.9s
- [x] Add exponential backoff retry mechanism for transient 503/429 errors from VLM providers
- [x] Implement both `call_gpt61_workflow` and `call_openai_direct` in `src/workflow_adapter.py`
- [x] Update `src/analyze_image.py` with `--backend openai-direct` and `--backend gpt61-workflow`
- [x] Document `OPENAI_API_KEY` in `.env.example`
- [ ] Run and compare GPT-6.1 Sol benchmark upon receipt of `OPENAI_API_KEY` in `.env`

## Multi-Sport Batch Campaign & 2D Tactical Minimaps
- [x] Ingest multi-sport dataset from `/home/mathieu/dev/projects/T3D/t3d-test-data` (`basket`, `foot`, `handball`, `usfootball`)
- [x] Implement video frame extraction (frame 0 with fallback to frame 30 if low quality/sharpness)
- [x] Build unified multi-sport configuration in `src/sports_config.py`
- [x] Implement 2D pitch/court minimap rendering for all sports in `src/visualization.py`:
  - Basketball: NBA court radar with `sports.basketball`
  - Football / Soccer: FIFA pitch radar with `sports.soccer` and `football-field-detection-f07vi/14` keypoints
  - Handball: 40m x 20m indoor court radar with 6m, 7m, 9m arcs and goals
  - American Football: 120-yard NFL/NCAA field radar with end zones and yard lines
- [x] Overlay 2D minimap in PiP top-right corner and generate side-by-side composite images (`_side_by_side.jpg`) for all sports
- [x] Implement `thinkingConfig: {"thinkingBudget": 1024}` reducing reasoning latency from 120s to ~5s per image in High Mode
- [x] Implement seamless daily quota cascading across Gemini Flash models (`gemini-3.6-flash` -> `gemini-3.5-flash` -> `gemini-3.1-flash-lite`)
- [x] Implement token consumption tracking (prompt, thoughts, candidates, total)
- [x] Implement batch runner `src/run_batch_sports.py` with `--limit 15`, `--sport all`, and cache reuse
- [x] Generate comprehensive model-specific benchmark reports (`out/report_<model>.md`, `out/batch_results_<model>.json`) and index `out/README.md` with coherent warm latencies (`342.22s` cumulative / `9.51s` mean across 36 items)

## Pipeline Architecture Documentation & Hybrid VLM Probabilistic Priors (`--vlm-priors`)
- [x] Create reusable end-to-end technical architecture specification in `doc/pipeline_2d_homography.md` (macro-steps, detailed steps, Input/Output contracts, ML/DL vs VLM vs mathematical operations, Cloud vs Local execution, and % latency breakdown)
- [x] Document ADR 6 in `doc/adr.md` ("Hybrid VLM Probabilistic Priors (`--vlm-priors`) for Zero-Cost Semantic Enrichment of Steps 4 & 5")
- [x] Enrich `src/sports_config.py` with canonical court landmark dictionaries (`canonical_landmarks_cm`) and prompt specifications (`vlm_landmarks_prompt`) across all 4 sports
- [x] Implement `compute_from_vlm_landmarks()` in `src/court_homography.py` (RANSAC homography estimation on VLM 2D keypoints with non-collinearity, determinant, and player-bounds validation + dynamic OCR yard-line resolver for American Football)
- [x] Update `src/workflow_adapter.py` behind `enable_vlm_priors` flag:
  - Enriched VLM JSON schema (`objects` with `team_id`, `role_status`, `jersey_number` + `court_calibration.keypoints`)
  - Polarity-aligned adaptive Bayesian fusion ($\alpha = 0.75$ under occlusion $\text{IoU} \ge 0.15$, $\alpha = 0.35$ when isolated) in `classify_player_teams()`
- [x] Wire `--vlm-priors` (and `T3D_ENABLE_VLM_PRIORS=1`) in `src/analyze_image.py` and `src/run_batch_sports.py`
- [x] Validate `--vlm-priors` end-to-end on representative test images


