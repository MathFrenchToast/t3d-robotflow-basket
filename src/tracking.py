import supervision as sv
from sports.basketball import ShotEventTracker

def initialize_trackers(fps: float):
    byte_tracker = sv.ByteTrack()
    
    shot_event_tracker = ShotEventTracker(
        reset_time_frames=int(fps * 1.7),
        minimum_frames_between_starts=int(fps * 0.5),
        cooldown_frames_after_made=int(fps * 0.5),
    )
    
    return byte_tracker, shot_event_tracker
