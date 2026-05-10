import supervision as sv
from sports.basketball import ShotEventTracker
from sports import ConsecutiveValueTracker

def initialize_trackers(fps: float):
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.35, # Slightly higher to avoid tracking noise
        lost_track_buffer=int(fps * 2), 
        minimum_matching_threshold=0.7, # Slightly more conservative matching
        frame_rate=int(fps),
        minimum_consecutive_frames=5   # Require 5 frames of detection before ID is assigned
    )
    
    shot_event_tracker = ShotEventTracker(
        reset_time_frames=int(fps * 1.7),
        minimum_frames_between_starts=int(fps * 0.5),
        cooldown_frames_after_made=int(fps * 0.5),
    )
    
    # Trackers for jersey numbers and team assignments
    number_validator = ConsecutiveValueTracker(n_consecutive=3)
    team_validator = ConsecutiveValueTracker(n_consecutive=1)
    
    return byte_tracker, shot_event_tracker, number_validator, team_validator
