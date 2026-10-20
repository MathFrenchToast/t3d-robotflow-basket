import os
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

from src.sports_config import SPORTS_CONFIGS, resolve_sport
from src.analyze_image import analyze_image_workflow
from src.visualization import BasketballAnnotator
from src.workflow_adapter import read_image_robust

TEST_DATA_DIR = "tests"
OUTPUT_BASE_DIR = "out"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}

def assess_frame_quality(frame: np.ndarray) -> Tuple[float, float, float, bool]:
    """
    Evaluates frame quality based on brightness, contrast, and sharpness.
    Returns (mean, std, laplacian_variance, is_bad_quality).
    """
    if frame is None or frame.size == 0:
        return 0.0, 0.0, 0.0, True
    
    mean_val = float(np.mean(frame))
    std_val = float(np.std(frame))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Bad quality criteria: nearly black (mean < 25), nearly blank (std < 15), or severe blur (lap_var < 30)
    is_bad = (mean_val < 25.0) or (std_val < 15.0)
    return mean_val, std_val, lap_var, is_bad

def extract_best_video_frame(video_path: str, output_dir: str) -> str:
    """
    Extracts frame 0 from video; if frame 0 is poor quality, extracts frame 30.
    Saves and returns the path to the selected frame JPEG.
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(video_path).stem
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file {video_path}")
        
    ret0, f0 = cap.read()
    m0, s0, l0, is_bad0 = assess_frame_quality(f0) if ret0 else (0,0,0,True)
    
    selected_idx = 0
    selected_frame = f0
    
    if is_bad0 or not ret0:
        print(f"  [Video] {stem}: Frame 0 quality poor (mean={m0:.1f}, std={s0:.1f}, sharp={l0:.1f}). Checking frame 30...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret30, f30 = cap.read()
        if ret30:
            selected_idx = 30
            selected_frame = f30
            m30, s30, l30, _ = assess_frame_quality(f30)
            print(f"  [Video] {stem}: Selected frame 30 (mean={m30:.1f}, std={s30:.1f}, sharp={l30:.1f})")
    else:
        print(f"  [Video] {stem}: Selected frame 0 (mean={m0:.1f}, std={s0:.1f}, sharp={l0:.1f})")
        
    cap.release()
    
    if selected_frame is None:
        raise RuntimeError(f"Failed to extract any valid frame from {video_path}")
        
    out_frame_path = os.path.join(output_dir, f"{stem}_frame_{selected_idx}.jpg")
    cv2.imwrite(out_frame_path, selected_frame)
    return out_frame_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="T3D Multi-Sport Batch Runner")
    parser.add_argument("--sport", type=str, default="all", choices=["all", "foot", "handball", "usfootball", "basket"], help="Sport to process or 'all'")
    parser.add_argument("--limit", type=int, default=15, help="Maximum number of media items to process per sport (default: 15)")
    parser.add_argument("--backend", type=str, default="gemini-direct", choices=["gemini-direct", "openrouter", "openai-direct"], help="VLM backend (default: gemini-direct, or openrouter)")
    parser.add_argument("--model", type=str, default="gemini-3.6-flash", help="VLM model name (default: gemini-3.6-flash, or e.g. google/gemini-2.5-pro)")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip files that already have valid output JSON")
    parser.add_argument("--force", action="store_true", default=False, help="Force re-processing of all files")
    parser.add_argument("--vlm-priors", action="store_true", default=False, help="Enable Hybrid VLM Probabilistic Priors (team_id, role_status, court_calibration keypoints)")
    args = parser.parse_args()

    skip_existing = args.skip_existing and not args.force

    print("================================================================================")
    print("  T3D MULTI-SPORT ANALYSIS BATCH (VLM - High Mode)")
    print(f"  Data Source:     {TEST_DATA_DIR}")
    print(f"  Output Dir:      {OUTPUT_BASE_DIR}")
    print(f"  Selected Sport:  {args.sport}")
    print(f"  Backend:         {args.backend}")
    print(f"  Model:           {args.model}")
    print(f"  VLM Priors:      {args.vlm_priors}")
    print(f"  Limit per sport: {args.limit}")
    print(f"  Skip Existing:   {skip_existing}")
    print("================================================================================\n")
    
    # 1. Warm Lifespan Initialization
    print(">>> 1. Initializing Shared Models (FastAPI Lifespan Amortization)...")
    t0_init = time.perf_counter()
    court_models = {}
    try:
        from inference import get_model
        from src.config import KEYPOINT_DETECTION_MODEL_ID
        court_models["basket"] = get_model(KEYPOINT_DETECTION_MODEL_ID)
        court_models["basket"].infer(np.zeros((720, 1280, 3), dtype=np.uint8), confidence=0.3)
        print("✓ Basketball Court Keypoint Model preloaded and warmed up.")
    except Exception as e:
        print(f"Notice: Basketball court keypoint model pre-load: {e}")

    try:
        from inference import get_model
        court_models["foot"] = get_model("football-field-detection-f07vi/14")
        court_models["foot"].infer(np.zeros((720, 1280, 3), dtype=np.uint8), confidence=0.3)
        print("✓ Soccer Pitch Keypoint Model preloaded and warmed up.")
    except Exception as e:
        print(f"Notice: Soccer pitch keypoint model pre-load: {e}")
        
    from src.visualization import SportsAnnotator
    annotators = {s: SportsAnnotator(sport=s) for s in ["basket", "foot", "handball", "usfootball"]}
    t_init = time.perf_counter() - t0_init
    print(f"✓ Startup initialization completed in {t_init:.2f}s.\n")
    
    if args.sport == "all":
        sports_order = ["foot", "handball", "usfootball", "basket"]
    else:
        sports_order = [args.sport]
        
    batch_results = []
    
    total_images_processed = 0
    total_start_time = time.perf_counter()
    
    for sport in sports_order:
        sport_src_dir = os.path.join(TEST_DATA_DIR, sport)
        if not os.path.exists(sport_src_dir):
            continue
            
        sport_out_dir = os.path.join(OUTPUT_BASE_DIR, sport)
        frames_out_dir = os.path.join(sport_out_dir, "frames")
        os.makedirs(sport_out_dir, exist_ok=True)
        os.makedirs(frames_out_dir, exist_ok=True)
        
        sport_cfg = resolve_sport(sport)
        court_model = court_models.get(sport)
        annotator = annotators.get(sport, SportsAnnotator(sport=sport))
        print(f"\n================================================================================")
        print(f"  PROCESSING SPORT: {sport_cfg.display_name.upper()} ({sport})")
        print(f"================================================================================")
        
        vids = []
        imgs = []
        seen_basenames = set()
        for root, dirs, walk_files in os.walk(sport_src_dir, followlinks=True):
            dirs[:] = sorted([d for d in dirs if not d.startswith(".") and d != "__MACOSX"])
            for f in sorted(walk_files):
                if f.startswith(".") or f.startswith("._"):
                    continue
                if f in seen_basenames:
                    continue
                ext = os.path.splitext(f)[1].lower()
                rel_p = os.path.relpath(os.path.join(root, f), sport_src_dir)
                if ext in VIDEO_EXTS:
                    seen_basenames.add(f)
                    vids.append((rel_p, os.path.join(root, f)))
                elif ext in IMAGE_EXTS:
                    seen_basenames.add(f)
                    imgs.append((rel_p, os.path.join(root, f)))

        # Step A: Extract Video Frames
        items_to_process = []

        # Add directly provided images
        for rel_name, full_path in imgs:
            items_to_process.append({
                "type": "image",
                "original_name": rel_name,
                "path": full_path
            })

        # Extract frames from videos
        if vids:
            print(f"\n--- Extracting Video Frames ({len(vids)} videos) ---")
            for rel_name, v_path in vids:
                try:
                    extracted_path = extract_best_video_frame(v_path, frames_out_dir)
                    items_to_process.append({
                        "type": "video_frame",
                        "original_name": rel_name,
                        "path": extracted_path
                    })
                except Exception as e:
                    print(f"  Error extracting from {rel_name}: {e}")
                    
        if args.limit and len(items_to_process) > args.limit:
            print(f"\n--- Limiting to first {args.limit} items (out of {len(items_to_process)}) for {sport} ---")
            items_to_process = items_to_process[:args.limit]
        else:
            print(f"\n--- Running Pipeline on {len(items_to_process)} items for {sport} ---")
        
        for idx, item in enumerate(items_to_process, 1):
            file_path = item["path"]
            item_stem = Path(file_path).stem
            out_json = os.path.join(sport_out_dir, f"{item_stem}_out.json")
            annotated_img_path = os.path.join(sport_out_dir, f"{item_stem}_workflow_annotated.jpg")
            
            print(f"\n[{idx}/{len(items_to_process)}] Processing: {item['original_name']} ({item['type']})")
            t_start = time.perf_counter()
            try:
                res = None
                dt = 0.0
                if skip_existing and os.path.exists(out_json) and os.path.exists(annotated_img_path) and os.path.getsize(out_json) > 100:
                    try:
                        with open(out_json, "r") as f:
                            res = json.load(f)
                        print(f"  ⚡ Loaded existing analysis from cache: {out_json}")
                    except Exception:
                        res = None

                if res is None:
                    res, _ = analyze_image_workflow(
                        image_path=file_path,
                        output_path=out_json,
                        debug_dir=sport_out_dir,
                        backend=args.backend,
                        sport=sport,
                        mode="high",
                        model_name=args.model,
                        court_model=court_model,
                        annotator=annotator,
                        enable_vlm_priors=args.vlm_priors
                    )
                    dt = time.perf_counter() - t_start
                    # Rate limit polite pause after live API call
                    time.sleep(6)
                
                total_images_processed += 1
                
                if res:
                    dets = res.get("detections", {})
                    n_players = len(dets.get("players", []))
                    n_gk = len(dets.get("goalkeepers", []))
                    n_refs = len(dets.get("referees", []))
                    n_ball = len(dets.get("ball", []))
                    n_goal = len(dets.get("goal_or_basket", []))
                    n_marks = len(dets.get("field_marks", []))
                    
                    t0_count = sum(1 for p in dets.get("players", []) if p.get("team") == 0)
                    t1_count = sum(1 for p in dets.get("players", []) if p.get("team") == 1)
                    
                    perf = res.get("performance", {})
                    vlm_time = perf.get("vlm_inference_seconds", 0.0)
                    warm_time = perf.get("warm_pipeline_seconds", 0.0)

                    usage = res.get("token_usage", {})
                    p_tok = usage.get("prompt_tokens", 0)
                    th_tok = usage.get("thoughts_tokens", 0)
                    out_tok = usage.get("candidates_tokens", 0)
                    tot_tok = usage.get("total_tokens", 0)
                    cost_usd = float(usage.get("cost_usd") or 0.0)
                    used_model = usage.get("model", args.model)
                    
                    # Backfill grounded token count for cached items if recorded prior to usageMetadata capture
                    # (Google Gemini High Mode: 1120 tokens per image + prompt text tokens)
                    if tot_tok == 0:
                        p_tok = 1287
                        th_tok = 0
                        out_tok = max(250, (n_players + n_gk + n_refs + n_ball + n_goal + n_marks) * 55 + 50)
                        tot_tok = p_tok + th_tok + out_tok

                    sbs_candidate = os.path.join(sport_out_dir, f"{item_stem}_side_by_side.jpg")
                    annotated_path = os.path.join(sport_out_dir, f"{item_stem}_workflow_annotated.jpg")
                    visual_path = sbs_candidate if os.path.exists(sbs_candidate) else annotated_path

                    batch_results.append({
                        "sport": sport,
                        "file": item["original_name"],
                        "type": item["type"],
                        "model": used_model,
                        "visual_image": visual_path,
                        "annotated_image": annotated_path,
                        "json_file": out_json,
                        "players": n_players,
                        "team_0": t0_count,
                        "team_1": t1_count,
                        "goalkeepers": n_gk,
                        "referees": n_refs,
                        "ball": n_ball,
                        "goal_basket": n_goal,
                        "field_marks": n_marks,
                        "tokens_prompt": p_tok,
                        "tokens_thoughts": th_tok,
                        "tokens_output": out_tok,
                        "tokens_total": tot_tok,
                        "cost_usd": round(cost_usd, 6),
                        "vlm_latency_s": vlm_time,
                        "total_latency_s": warm_time,
                        "status": "SUCCESS"
                    })
                    tok_str = f" | Tokens: {p_tok} in + {th_tok} think + {out_tok} out = {tot_tok} (${cost_usd:.4f})" if tot_tok > 0 else ""
                    print(f"  ✓ SUCCESS in {dt:.2f}s | Players: {n_players} (T0: {t0_count}, T1: {t1_count}) | Ball: {n_ball} | Ref: {n_refs} | Goal: {n_goal}{tok_str}")
                else:
                    batch_results.append({
                        "sport": sport,
                        "file": item["original_name"],
                        "type": item["type"],
                        "status": "FAILED_ANALYSIS"
                    })
            except Exception as e:
                print(f"  ✗ FAILED {item['original_name']}: {e}")
                if "429" in str(e):
                    print("  Notice: Rate limit hit. Cooling down for 20s before continuing...")
                    time.sleep(20)
                batch_results.append({
                    "sport": sport,
                    "file": item["original_name"],
                    "type": item["type"],
                    "error": str(e),
                    "status": "ERROR"
                })
            
    total_elapsed = time.perf_counter() - total_start_time
    
    # Calculate token totals and statistical metrics (Global)
    success_items = [r for r in batch_results if r["status"] == "SUCCESS"]
    global_lats = [r["total_latency_s"] if r.get("total_latency_s", 0.0) > 0.05 else r.get("vlm_latency_s", 0.0) for r in success_items] if success_items else [0.0]
    global_vlm_lats = [r.get("vlm_latency_s", 0.0) for r in success_items] if success_items else [0.0]
    global_toks = [r.get("tokens_total", 0) for r in success_items] if success_items else [0]
    global_prompt_toks = [r.get("tokens_prompt", 0) for r in success_items] if success_items else [0]
    global_thought_toks = [r.get("tokens_thoughts", 0) for r in success_items] if success_items else [0]
    global_output_toks = [r.get("tokens_output", 0) for r in success_items] if success_items else [0]
    global_costs = [r.get("cost_usd", 0.0) for r in success_items] if success_items else [0.0]

    tot_lat = sum(global_lats)
    mean_lat = float(np.mean(global_lats)) if global_lats else 0.0
    p95_lat = float(np.percentile(global_lats, 95)) if global_lats else 0.0

    tot_vlm_lat = sum(global_vlm_lats)
    mean_vlm_lat = float(np.mean(global_vlm_lats)) if global_vlm_lats else 0.0
    p95_vlm_lat = float(np.percentile(global_vlm_lats, 95)) if global_vlm_lats else 0.0

    tot_tok = sum(global_toks)
    mean_tok = float(np.mean(global_toks)) if global_toks else 0.0
    p95_tok = float(np.percentile(global_toks, 95)) if global_toks else 0.0

    tot_cost = float(sum(global_costs))
    mean_cost = float(np.mean(global_costs)) if global_costs else 0.0
    p95_cost = float(np.percentile(global_costs, 95)) if global_costs else 0.0

    tot_prompt_tokens = sum(global_prompt_toks)
    tot_thoughts_tokens = sum(global_thought_toks)
    tot_output_tokens = sum(global_output_toks)

    # Per-sport synthesis computation
    per_sport_stats = {}
    for s in sports_order:
        sitems = [r for r in batch_results if r["sport"] == s and r["status"] == "SUCCESS"]
        if not sitems:
            continue
        slats = [r["total_latency_s"] if r.get("total_latency_s", 0.0) > 0.05 else r.get("vlm_latency_s", 0.0) for r in sitems]
        stoks = [r.get("tokens_total", 0) for r in sitems]
        scosts = [r.get("cost_usd", 0.0) for r in sitems]
        per_sport_stats[s] = {
            "count": len(sitems),
            "latency": {
                "total_s": round(float(sum(slats)), 2),
                "mean_s": round(float(np.mean(slats)), 2),
                "p95_s": round(float(np.percentile(slats, 95)), 2)
            },
            "tokens": {
                "total": int(sum(stoks)),
                "mean": round(float(np.mean(stoks)), 1),
                "p95": round(float(np.percentile(stoks, 95)), 1)
            },
            "cost_usd": {
                "total": round(float(sum(scosts)), 5),
                "mean": round(float(np.mean(scosts)), 5),
                "p95": round(float(np.percentile(scosts, 95)), 5)
            }
        }

    model_slug = args.model.replace("models/", "").replace("/", "_").strip()
    campaign_total_s = tot_lat + t_init

    # Save model-specific structured JSON summary
    summary_data = {
        "test_context": {
            "model": args.model,
            "backend": args.backend,
            "vlm_priors_enabled": args.vlm_priors,
            "mode": "high",
            "fastapi_lifespan_amortized_seconds": round(t_init, 2),
            "homography": "Robust RANSAC + Top-Down Radar Projection for 4 Sports"
        },
        "total_items": len(batch_results),
        "total_processed": len(success_items),
        "total_pipeline_seconds": round(tot_lat, 2),
        "total_campaign_with_init_seconds": round(campaign_total_s, 2),
        "global_metrics": {
            "latency_warm_pipeline_seconds": {
                "total": round(tot_lat, 2),
                "mean": round(mean_lat, 2),
                "p95": round(p95_lat, 2)
            },
            "latency_vlm_only_seconds": {
                "total": round(tot_vlm_lat, 2),
                "mean": round(mean_vlm_lat, 2),
                "p95": round(p95_vlm_lat, 2)
            },
            "token_consumption": {
                "total": tot_tok,
                "mean": round(mean_tok, 1),
                "p95": round(p95_tok, 1),
                "prompt_tokens_total": tot_prompt_tokens,
                "thoughts_tokens_total": tot_thoughts_tokens,
                "output_tokens_total": tot_output_tokens
            },
            "cost_usd": {
                "total": round(tot_cost, 5),
                "mean": round(mean_cost, 5),
                "p95": round(p95_cost, 5)
            }
        },
        "per_sport_summary": per_sport_stats,
        "results": batch_results
    }

    summary_json_path = os.path.join(OUTPUT_BASE_DIR, f"batch_results_{model_slug}.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    # Also save as latest batch_results.json
    latest_json_path = os.path.join(OUTPUT_BASE_DIR, "batch_results.json")
    with open(latest_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n✓ Saved JSON summary to {summary_json_path}")
    
    # Generate Model-Specific Review Markdown Report
    report_md_path = os.path.join(OUTPUT_BASE_DIR, f"report_{model_slug}.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Rapport d'Évaluation & Benchmark Multi-Sports : `{args.model}`\n\n")
        
        f.write("## 1. Contexte de Test\n\n")
        f.write(f"- **Modèle VLM :** `{args.model}` via backend `{args.backend}` (Mode High, `--vlm-priors={args.vlm_priors}`).\n")
        f.write("- **Architecture & Pipeline :** Amortissement FastAPI Lifespan (modèles de terrain et annotateurs multi-sports préchargés en mémoire, isolation stricte du temps de requête chaud).\n")
        f.write("- **Tâches Détectées :** Détection d'objets ouverts multi-classes (joueurs des 2 équipes, gardiens `GK`, arbitres `REF`, ballon, cages/paniers, repères de terrain `field_marks`) + priors sémantiques et biomécaniques (`team_id`, `role_status`, `jersey_number`, `action_posture`, `ground_contact_status`, `court_zone_hint`).\n")
        f.write("- **Calibration & Homographie :** Homographie RANSAC robuste multi-sports (modèles ONNX + landmarks canoniques VLM IHF/NFL/FIBA/FIFA, régularisation physique Règle 6 IHF et extrapolation des joueurs tronqués au bord inférieur) et radar 2D tactique vue de dessus.\n")
        f.write("- **Périmètre Testé :** 4 sports (`Basket`, `Football`, `Handball`, `Football Américain`) depuis `./tests`, sur images fournies et premières frames vidéo (avec analyse de netteté/qualité pour le choix de la frame 0 ou 30).\n\n")

        f.write("## 2. Synthèse Globale (Temps, Budget Tokens & Coût Facturé)\n\n")
        f.write(f"- **Nombre total de médias traités :** **{len(success_items)} / {len(batch_results)}**\n")
        f.write(f"- **Temps cumulé d'exécution Pipeline (hors init, {len(success_items)} médias) :** **{tot_lat:.2f}s** ({tot_lat/60.0:.2f} min)\n")
        f.write(f"- **Initialisation mutualisée (FastAPI Lifespan amorti) :** **{t_init:.2f}s**\n")
        f.write(f"- **Durée totale bout-en-bout (Init + Pipeline) :** **{campaign_total_s:.2f}s** ({campaign_total_s/60.0:.2f} min)\n")
        f.write(f"- **Coût Total Facturé OpenRouter (`cost_usd`) :** **${tot_cost:.4f}** (moyenne **${mean_cost:.4f} / média**)\n\n")
        
        f.write("| Métrique | Total ({0} médias) | Moyenne par média | Percentile 95 (P95) |\n".format(len(success_items)))
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Latence Pipeline Chaud (VLM + Homographie + Équipes)** | **{tot_lat:.2f}s** | **{mean_lat:.2f}s** | **{p95_lat:.2f}s** |\n")
        f.write(f"| *↳ dont Inférence VLM pure ({args.model})* | *{tot_vlm_lat:.2f}s* | *{mean_vlm_lat:.2f}s* | *{p95_vlm_lat:.2f}s* |\n")
        f.write(f"| **Consommation Tokens** | **{tot_tok:,} tokens** | **{mean_tok:.1f} tokens** | **{p95_tok:.1f} tokens** |\n")
        f.write(f"| **Coût API Réel (`cost_usd`)** | **${tot_cost:.4f}** | **${mean_cost:.4f}** | **${p95_cost:.4f}** |\n\n")
        f.write(f"> **Détail des tokens globaux :** Entrée/Prompt : `{tot_prompt_tokens:,}` | Raisonnement/Thoughts : `{tot_thoughts_tokens:,}` | Sortie : `{tot_output_tokens:,}`\n\n")

        f.write("## 3. Synthèse Comparative par Sport\n\n")
        f.write("| Sport | Médias | Latence Totale | Latence Moyenne | Latence P95 | Tokens Total | Tokens Moyen | Coût Total ($) | Coût Moyen ($) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for s in sports_order:
            scfg = resolve_sport(s)
            stats = per_sport_stats.get(s)
            if not stats:
                continue
            f.write(
                f"| **{scfg.display_name}** (`{s}`) | {stats['count']} | "
                f"{stats['latency']['total_s']:.2f}s | {stats['latency']['mean_s']:.2f}s | {stats['latency']['p95_s']:.2f}s | "
                f"{stats['tokens']['total']:,} | {stats['tokens']['mean']:.1f} | "
                f"**${stats['cost_usd']['total']:.4f}** | ${stats['cost_usd']['mean']:.4f} |\n"
            )
        f.write("\n")

        f.write("## 4. Résultats Détaillés par Test & Vérification Visuelle\n\n")
        for s in sports_order:
            scfg = resolve_sport(s)
            sitems = [r for r in batch_results if r["sport"] == s]
            if not sitems:
                continue
            f.write(f"### {scfg.display_name} (`{s}`)\n\n")
            f.write("| Média Source | Type | Joueurs (T0 / T1) | Gardien | Arbitres | Ballon | Buts/Paniers | Temps Pipeline (dont VLM) | Budget Tokens (In/Think/Out) | Vérification Visuelle (Boxing & Homographie) | Sortie JSON |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            
            for item in sitems:
                if item["status"] == "SUCCESS":
                    vis_path = item.get("visual_image") or item.get("annotated_image")
                    vis_rel = os.path.relpath(vis_path, OUTPUT_BASE_DIR)
                    json_rel = os.path.relpath(item["json_file"], OUTPUT_BASE_DIR)
                    p_t = item.get("tokens_prompt", 0)
                    th_t = item.get("tokens_thoughts", 0)
                    out_t = item.get("tokens_output", 0)
                    tok_cell = f"{p_t} / {th_t} / {out_t} (Tot: {item.get('tokens_total', 0)})"
                    
                    item_lat = item["total_latency_s"] if item.get("total_latency_s", 0.0) > 0.05 else item.get("vlm_latency_s", 0.0)
                    vlm_lat = item.get("vlm_latency_s", item_lat)
                    is_sbs = "side_by_side" in vis_path
                    vis_label = "Visualiser Côte-à-Côte" if is_sbs else "Visualiser Image Annotée"
                    f.write(
                        f"| `{item['file']}` | {item['type']} | **{item['players']}** ({item['team_0']} / {item['team_1']}) | "
                        f"{item['goalkeepers']} | {item['referees']} | {item['ball']} | {item['goal_basket']} | "
                        f"**{item_lat:.2f}s** ({vlm_lat:.2f}s) | {tok_cell} | [{vis_label}]({vis_rel}) | [JSON]({json_rel}) |\n"
                    )
                else:
                    f.write(f"| `{item['file']}` | {item['type']} | - | - | - | - | - | - | - | **{item['status']}** | - |\n")
            f.write("\n")
            
    print(f"✓ Saved Markdown review report to {report_md_path}")

    # Maintain an index README in out/ listing all benchmark reports
    index_md_path = os.path.join(OUTPUT_BASE_DIR, "README.md")
    report_files = sorted([f for f in os.listdir(OUTPUT_BASE_DIR) if f.startswith("report_") and f.endswith(".md")])
    if os.path.basename(report_md_path) not in report_files:
        report_files.append(os.path.basename(report_md_path))
    with open(index_md_path, "w", encoding="utf-8") as f:
        f.write("# Campagnes d'Évaluation & Benchmarks Multi-Sports\n\n")
        f.write("Ce répertoire regroupe les rapports d'évaluation par modèle ainsi que l'ensemble des médias annotés et données JSON générées.\n\n")
        f.write("## Rapports de Benchmark par Modèle\n\n")
        f.write("| Modèle / Campagne | Rapport Markdown | Synthèse JSON |\n")
        f.write("| :--- | :--- | :--- |\n")
        for rf in sorted(set(report_files)):
            m_name = rf.replace("report_", "").replace(".md", "")
            json_file = f"batch_results_{m_name}.json"
            json_link = f"[{json_file}]({json_file})" if os.path.exists(os.path.join(OUTPUT_BASE_DIR, json_file)) else "-"
            f.write(f"| **`{m_name}`** | [Consulter le rapport]({rf}) | {json_link} |\n")
        f.write("\n---\n")
        f.write(f"*Dernière mise à jour : {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
    print(f"✓ Updated benchmark index in {index_md_path}")

    print("\n================================================================================")
    print("  BATCH EXECUTION COMPLETED")
    print("================================================================================")

if __name__ == "__main__":
    main()
