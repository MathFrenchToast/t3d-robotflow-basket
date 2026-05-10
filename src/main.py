import argparse
import sys
import os
from src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Basketball AI - Detect, Track and Identify Players")
    parser.add_argument("--source", type=str, required=True, help="Path to source video")
    parser.add_argument("--target", type=str, default="result.mp4", help="Path to target video")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to process")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source):
        print(f"Error: Source video {args.source} not found.")
        sys.exit(1)
        
    try:
        run_pipeline(args.source, args.target, max_frames=args.frames)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
