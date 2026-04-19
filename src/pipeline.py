import supervision as sv
from tqdm import tqdm
from src.config import (
    PLAYER_DETECTION_MODEL_CONFIDENCE,
    PLAYER_DETECTION_MODEL_IOU_THRESHOLD,
    KEYPOINT_DETECTION_MODEL_CONFIDENCE
)
from src.models import BasketballModels
from src.tracking import initialize_trackers
from src.visualization import BasketballAnnotator
from sports.common import ViewTransformer

def run_pipeline(source_video_path: str, target_video_path: str):
    models = BasketballModels()
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    byte_tracker, shot_tracker = initialize_trackers(video_info.fps)
    annotator = BasketballAnnotator()
    
    frame_generator = sv.get_video_frames_generator(source_video_path)
    
    # Placeholder for ViewTransformer - in the notebook it's initialized after first frame court detection
    view_transformer = None

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            # 1. Detection
            results = models.player_model.infer(
                frame, 
                confidence=PLAYER_DETECTION_MODEL_CONFIDENCE, 
                iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD
            )[0]
            detections = sv.Detections.from_inference(results)
            
            # 2. Tracking
            detections = byte_tracker.update_with_detections(detections)
            
            # 3. Court Detection & Transformation (Simplified for this version)
            # In a full implementation, you'd detect keypoints here and update the view_transformer
            
            # 4. Annotation
            labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
            annotated_frame = annotator.annotate_frame(frame, detections, labels)
            
            sink.write_frame(annotated_frame)
