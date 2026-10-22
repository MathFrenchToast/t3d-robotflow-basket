# Architecture Decision Records

## ADR 1: Project Structure and Dependency Management

**Date:** 2026-04-19

**Status:** Accepted

**Context:**
The goal is to convert a complex Colab notebook into a maintainable Python project. The project involves heavy dependencies like `inference-gpu` and `supervision`.

**Decision:**
- Use `uv` for lightning-fast dependency management and project isolation.
- Separate logic into modules (`config`, `models`, `tracking`, `visualization`, `pipeline`) to ensure clean separation of concerns and ease of testing.
- Use a `src/` layout for the package.

**Consequences:**
- Easier to maintain and extend compared to a single script.
- Consistent environment setup using `uv.lock`.
- Clear boundaries between model inference and domain logic (basketball tracking).

## ADR 2: Roboflow Workflows (VLM) & Partial-Court Homography

**Date:** 2026-10-17

**Status:** Accepted

**Context:**
Modern foundation vision models (VLMs such as Gemini Flash, GPT-6.1 Sol / GPT-4o) and Roboflow Workflows enable serverless object detection with open classes (`players`, `referees`, `ball`, `basket`, `paint`, `field marks`).
Furthermore, broadcast and tactical images rarely show the full court or all 4 corners; camera framing is often focused on the half-court or key/paint area.

**Decision:**
1. Implement a `WorkflowAdapter` to query Roboflow Serverless Workflows and parse their JSON predictions into standard `supervision.Detections`.
2. Compute homography based on any $\ge 4$ non-collinear field marks (such as the paint/key corners, free-throw line, 3-point line intersections, or rim anchor) using `cv2.findHomography` with RANSAC and `supervision.ViewTransformer`.
3. Separate referees from players using VLM predictions with cross-class IoU suppression, and classify team jerseys using chromaticity clustering ($B - R$ dominance) without forced arbitrary quota reassignments.
4. Support both serverless workflow execution and local fallback.
5. Strictly delineate on-court players vs bench/off-court players using homography court boundaries ($0 \le X \le 2865$, $0 \le Y \le 1524$), enforcing at most 5 active on-court players per team on the tactical radar.

**Consequences:**
- Resilient camera calibration even when corners are off-screen.
- Clean JSON ingestion compatible with Roboflow Workflows output.
- Serverless flexibility without mandatory heavy local CUDA drivers.
- Reliable team assignment invariant to shadows or bench players.

## ADR 3: Handling Arbitrary Player Counts and Partial-Court Views

**Date:** 2026-10-17

**Status:** Accepted

**Context:**
Single-image basketball analysis frequently encounters camera framings where fewer than 10 players are visible (e.g. fast breaks, corner isolation, 3v3 half-court plays, or bench occlusions). Enforcing a rigid 10-player or 5-per-team quota causes false reassignments, arbitrary demotions, and pipeline crashes.

**Decision:**
1. Detection models (both local YOLO/SAM and VLM Workflows) operate in open-count mode, capturing whatever entities are visible without count constraints.
2. Team clustering uses unsupervised $K$-Means ($K=2$) to establish the two team appearance centroids. Players are assigned strictly according to proximity to centroids along the discriminant axis, with no balancing quota.
3. The active on-court mask (`active_court_mask`) only demotes surplus players if a single team strictly exceeds 5 on-court players (`len(t_indices) > 5`), retaining the 5 players deepest inside the playing bounds and demoting sideline bleed. For any team count $\le 5$ (e.g. 1, 2, 3, 4, or 5 players), all visible players remain active on court.
4. Partial-court homography is computed with a minimal keypoint threshold ($\ge 4$), allowing calibration on left half-court, right half-court, or painted area alone.

**Consequences:**
- Full flexibility across arbitrary camera shots (1 player, 6 players, or 10 players).
- No artificial re-balancing of teams when a team has fewer visible players.
- Reliable tactical radar representation matching reality.

## ADR 4: Multi-Provider VLM Integration (Gemini 3.8 Flash & GPT-6.1 Sol) and Warm Latency Isolation

**Date:** 2026-10-18

**Status:** Accepted

**Context:**
1. The user wishes to benchmark and compare object detection and team classification between Google Gemini 3.8 Flash and OpenAI GPT-6.1 Sol.
2. Roboflow Serverless workflows with foundation models (`roboflow_core/open_ai@v7` or `roboflow_core/google_gemini@v6`) default to Roboflow managed credits. On free-tier workspaces with 0 credits, this triggers a `403 Forbidden` error. Supplying provider keys (`google_api_key`, `openai_api_key`) directly routes requests through provider billing.
3. Execution latency is critical for production deployment in FastAPI. One-time model and annotator loading is amortized during the FastAPI lifespan (`@asynccontextmanager`), so benchmarks must isolate **warm request processing time**.

**Decision:**
1. Support dual execution paths for both Gemini 3.8 Flash and GPT-6.1 Sol:
   - Direct Provider API (`call_gemini_direct`, `call_openai_direct`) with normalized bounding box ingestion.
   - Serverless Workflow execution with inline specification parameterizing the provider API key.
2. Embed exponential backoff retry for transient network and server unavailability errors (HTTP 503 / 429).
3. Performance metrics in output JSON report discrete stages:
   - `vlm_inference_seconds` (network roundtrip + VLM reasoning)
   - `court_homography_seconds` (YOLO keypoint inference + RANSAC homography)
   - `team_classification_seconds` (jersey crop extraction + chromaticity clustering)
   - `warm_pipeline_seconds` (total per-request warm latency)

**Consequences:**
- Clear performance baseline established: local court homography (~16-100ms) and team clustering (~32-60ms) constitute < 2% of pipeline latency, while VLM inference (~11.4-12.8s for Gemini 3.8 Flash) constitutes ~98%.
- Complete independence from Roboflow credit exhaustion via direct provider key binding.
- Readiness to benchmark GPT-6.1 Sol as soon as `OPENAI_API_KEY` is supplied in `.env`.

## ADR 5: Multi-Sport Generalization, 2D Tactical Minimaps, Token Budgeting, and Daily Quota Cascading

**Date:** 2026-10-18

**Status:** Accepted

**Context:**
1. Test data encompasses 4 distinct sports (`basket`, `foot`, `handball`, `usfootball`) with both static images and video clips.
2. Unconstrained visual reasoning in Gemini Flash (`thinkingLevel: high`) produced 120s timeouts and consumed excessive reasoning tokens.
3. Google AI Studio Free Tier limits daily requests to 20 requests per day per model version.
4. Users require visual verification through 2D tactical court/pitch minimaps (radar top-down projection) across **all 4 sports**, not just basketball.

**Decision:**
1. **Multi-Sport 2D Radar Architectures:**
   - **Basketball:** NBA court model with `sports.basketball` + `basketball-court-detection-2/14` keypoints.
   - **Football / Soccer:** FIFA pitch model with `sports.soccer` + `football-field-detection-f07vi/14` keypoints mapped 1-to-1 to pitch vertices.
   - **Handball:** Standard 40m x 20m 2D court graphic with 6m, 7m, 9m arcs and perspective player projection.
   - **American Football:** 120-yard NFL/NCAA field graphic with endzones and 5/10-yard lines with perspective player projection.
2. **Dual Visual Artifacts:**
   - Picture-in-Picture (PiP) minimap overlay in the top-right corner of `<stem>_workflow_annotated.jpg`.
   - Side-by-side composite `<stem>_side_by_side.jpg` pairing detections on the left with the full-resolution 2D tactical radar on the right.
3. **Reasoning Token Budgeting:**
   - Grounded in official Google documentation, configure `mediaResolution: MEDIA_RESOLUTION_HIGH` with `thinkingConfig: {"thinkingBudget": 1024}`.
   - Reduces latency from 120s down to **~4.6s** per image while preserving fine multimodal detail.
4. **Seamless Quota Cascading:**
   - Detect HTTP 429 quota exhaustion (`generativelanguage.googleapis.com/generate_content_free_tier_requests`, limit: 20) and cascade seamlessly through `gemini-3.6-flash` -> `gemini-3.5-flash` -> `gemini-3.1-flash-lite`.
5. **Statistical Reporting:**
   - Track prompt, thoughts, and output tokens per item.
   - Compute total, mean, and 95th percentile (P95) metrics globally and per sport in `out/README.md` and `out/batch_results.json`.

**Consequences:**
- Universal tactical radar visualization available across all sports.
- Predictable and fast inference latency (~4-5s per request).
- Autonomous completion of multi-sport campaigns despite free-tier rate limits.

## ADR 6: Hybrid VLM Probabilistic Priors & Multi-Stage Consensus (`--vlm-priors` Feature Flag)

**Date:** 2026-10-18

**Status:** Accepted

**Context:**
1. In the baseline pipeline (ADR 2–5), Step 2 (VLM inference) consumes **~96.9%** of total execution time to encode the high-resolution image (1,120 tokens) and run visual reasoning (`thinkingBudget: 1024`), but only outputs coarse 2D bounding boxes (`box_2d`) and class labels.
2. Meanwhile, downstream local stages face two structural limitations when relying strictly on local pixels or dedicated ONNX keypoint models:
   - **Step 4 (Court Homography):** Handball and American Football lack dedicated Roboflow ONNX keypoint models, forcing a fallback to generic perspective projection. Conversely, frontier VLMs can natively point to canonical landmarks (6m/7m/goalposts in Handball) and read yard line numbers via OCR (`10`, `20`, `30`, `40`, `50` and hash marks in American Football).
   - **Step 5 (Team & Active-Court Classification):** Local `K-Means` HSV clustering on torso crops degrades under heavy player-to-player occlusion ($\text{IoU} \ge 0.15$), and purely geometric boundary checks can misclassify an active inbound passer stepping outside the sideline as a `Bench` player.
3. Official documentation from Google AI ([Gemini Spatial Pointing & Vision](https://ai.google.dev/gemini-api/docs/vision)) and Roboflow ([Frontier VLM Segmentation & Multi-Model Consensus Workflows](https://blog.roboflow.com/how-to-use-florence-2-for-instance-segmentation/)) demonstrates that pairing VLM semantic/spatial priors with local geometric/ML validation achieves superior accuracy with negligible latency overhead (< 150ms for ~100 extra output tokens).

**Decision:**
1. **Introduce a Feature Flag (`--vlm-priors` / `enable_vlm_priors: bool = False` / `T3D_ENABLE_VLM_PRIORS`):**
   - Preserves 100% backward compatibility with the baseline pipeline when disabled (`False`).
   - Allows side-by-side A/B benchmarking of accuracy, homography inliers, and latency overhead when enabled (`True`).
2. **Enrich Step 2 Output Schema (under `--vlm-priors`):**
   - **Per-Player Semantic, Biomechanical & Topological Priors:**
     - `team_id` (`0` or `1`), `role_status` (`"active_on_court"`, `"active_inbound"`, `"bench_off_court"`), and `jersey_number`.
     - `action_posture` (`"standing_running"`, `"defensive_stance"`, `"jumping_shoot"`, `"jumping_block"`, `"diving_falling"`).
     - `ground_contact_status` (`"visible_on_floor"`, `"airborne"`, `"truncated_bottom_edge"`, `"occluded_by_player"`).
     - `ground_point_norm` (`[y_ground, x_ground]` in 0..1000 scale, allowing `y_ground > 1000` when legs are cropped by the bottom edge of the camera frame, or pointing to the vertical floor shadow under an airborne player).
     - `court_zone_hint` (e.g., `"inside_6m"`, `"on_6m_line"`, `"between_6m_and_9m"`, `"on_9m_line"`, `"beyond_9m"` in Handball) providing direct topological localization relative to painted court markings.
   - **Canonical Court Calibration Landmarks (`court_calibration`):** `visible_court_side` (`"left" | "right" | "center" | "full"`) and named `keypoints` (`{"name": "<canonical_landmark>", "point_2d": [y, x]}`) mapped to exact metric coordinates in `src/sports_config.py`.
3. **Hybrid Homography, Ground-Contact Extrapolation & Physics-Informed Regularization in Step 4 (`CourtHomographyCalculator`):**
   - **Ground-Contact Point ($Z=0$) Correction:** Instead of naively projecting the bottom of the 2D bounding box (`y2`), compute the true floor contact point using `ground_point_norm` and anthropometric stature extrapolation ($y_{\text{floor}} = \max(y_{\text{VLM}}, y_1 + 3.15 \times (x_2 - x_1))$) for players truncated at the bottom frame edge (`truncated_bottom_edge`, e.g., foreground player `#22`), and vertical floor projection for airborne jump shooters (`airborne`).
   - **Posture- & Zone-Aware Regularization (IHF Rule 6 Compliance):** In Handball, non-airborne field players (`ground_contact_status != "airborne"`) with `court_zone_hint` outside the 6m goal area (`"on_6m_line"`, `"between_6m_and_9m"`, `"beyond_9m"`) are radially regularized so their projected $(X, Y)$ coordinates strictly respect the 6m arc, 7m penalty mark, and 9m free-throw arc, whereas airborne jump-shooters (`action_posture == "jumping_shoot"` or `ground_contact_status == "airborne"`) legally flying over the 6m zone (per [IHF Rule 6:7a](https://www.ihf.info/sites/default/files/2019-07/New%20Playing%20Rules_E_0.pdf)) preserve their in-zone air projection.
4. **Probabilistic Bayesian Consensus in Step 5 (`classify_player_teams` & `active_court_mask`):**
   - Align VLM team polarity (`0`/`1`) with local `K-Means/HSV` polarity via majority vote on unoccluded players ($\text{IoU}_{\max} < 0.15$).
   - Fuse team probabilities per player $i$ weighted by occlusion level ($\alpha_i = 0.65$ for unoccluded players and $\alpha_i = 0.85$ for occluded players).
   - Fuse geometric court bounds with VLM posture (`active_inbound` / `active_on_court` vs `bench_off_court`).

**Consequences:**
- Enables true RANSAC homography on Handball and American Football without requiring custom-trained ONNX keypoint networks.
- Resolves foreground bottom-edge truncation (`#22`), eliminates false 6m goal-area intrusions for standing defenders while preserving airborne jump-shots, and enforces topological ordering relative to the 6m/7m/9m markings.
- Controlled rollout and measurable A/B evaluation thanks to the `--vlm-priors` feature flag.


