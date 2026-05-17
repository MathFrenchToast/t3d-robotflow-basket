import argparse
import sys
import os
from src.pipeline import run_pipeline
from src.analyze_image import analyze_image

def main():
    parser = argparse.ArgumentParser(description="Basketball AI - Detect, Track and Identify Players")
    parser.add_argument("--source", type=str, help="Path to source video")
    parser.add_argument("--image", type=str, help="Path to source image")
    parser.add_argument("--target", type=str, default="result.mp4", help="Path to target video (for --source)")
    parser.add_argument("--output", type=str, default="output.json", help="Path to output JSON (for --image)")
    parser.add_argument("--debug-dir", type=str, default="out", help="Directory for debug output (for --image)")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to process (for --source)")
    
    args = parser.parse_args()
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Source image {args.image} not found.")
            sys.exit(1)
        
        if not os.path.exists(args.debug_dir):
            os.makedirs(args.debug_dir)
            
        try:
            analyze_image(args.image, args.output, args.debug_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An error occurred during image analysis: {e}")
            sys.exit(1)
    elif args.source:
        if not os.path.exists(args.source):
            print(f"Error: Source video {args.source} not found.")
            sys.exit(1)
        try:
            run_pipeline(args.source, args.target, max_frames=args.frames)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An error occurred during video processing: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
