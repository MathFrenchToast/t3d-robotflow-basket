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
    parser.add_argument("--debug-dir", type=str, default=None, help="Directory for debug output (defaults to out/debug if --debug else out)")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to process (for --source)")
    parser.add_argument("--sam-version", type=str, choices=["sam2", "sam3"], help="SAM model version to use (overrides config)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug outputs (crops, masks)")
    
    args = parser.parse_args()
    
    # Set default debug directory dynamically
    if args.debug_dir is None:
        args.debug_dir = "out/debug" if args.debug else "out"
    
    # Update config if sam-version is provided
    if args.sam_version:
        from src import config
        config.SAM_VERSION = args.sam_version
        print(f"Forcing SAM version to: {args.sam_version}")
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Source image {args.image} not found.")
            sys.exit(1)
        
        if not os.path.exists(args.debug_dir):
            os.makedirs(args.debug_dir)
            
        try:
            analyze_image(args.image, args.output, args.debug_dir, debug=args.debug)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An error occurred during image analysis: {e}")
            sys.exit(1)
    elif args.source:
        if not os.path.exists(args.source):
            print(f"Error: Source video {args.source} not found.")
            sys.exit(1)
        
        if args.debug and not os.path.exists(args.debug_dir):
            os.makedirs(args.debug_dir)
            
        try:
            run_pipeline(args.source, args.target, max_frames=args.frames, debug=args.debug, debug_dir=args.debug_dir)
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
