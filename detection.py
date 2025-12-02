import cv2
import numpy as np
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
import multiprocessing
from functools import partial

def detect_static_background(video_path, threshold=10, learning_rate=0.01):
    """
    Detect static background using MOG2 background subtractor
    """
    cap = cv2.VideoCapture(video_path)
    
    # Create background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=500,  # Number of frames to consider
        varThreshold=threshold,
        detectShadows=False
    )
    back_sub.setBackgroundRatio(learning_rate)
    
    backgrounds = []
    frame_count = 0
    sample_rate = 10  # Sample every 10 frames
    
    print("Detecting background...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            # Update background model
            fg_mask = back_sub.apply(frame, learningRate=learning_rate)
            
            # Get current background image
            background = back_sub.getBackgroundImage()
            if background is not None:
                backgrounds.append(background)
        
        frame_count += 1
    
    cap.release()
    
    if not backgrounds:
        print("No background detected")
        return None
    
    # Use median of sampled backgrounds
    median_background = np.median(backgrounds, axis=0).astype(np.uint8)
    
    print(f"Background detected from {len(backgrounds)} samples")
    return median_background

def create_alpha_mask(frame, background, threshold=30, smooth=True):
    """
    Create alpha mask by comparing frame with background
    """
    # Convert to grayscale for comparison
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference
    diff = cv2.absdiff(frame_gray, bg_gray)
    
    # Apply threshold to create binary mask
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up mask
    if smooth:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask

def frame_to_rgba(frame, background, threshold=30, softness=0.5):
    """
    Convert BGR frame to RGBA with transparency
    """
    # Create alpha mask
    alpha_mask = create_alpha_mask(frame, background, threshold)
    
    # Apply softness to alpha channel
    if softness > 0:
        alpha_mask = cv2.GaussianBlur(alpha_mask, (0, 0), softness * 10)
    
    # Normalize alpha to 0-255
    alpha = alpha_mask.astype(np.float32) / 255.0
    
    # Create RGBA image
    b, g, r = cv2.split(frame)
    rgba = cv2.merge([b, g, r, alpha_mask])
    
    return rgba

def process_video_to_transparent(video_path, output_path, background_threshold=10, 
                                 mask_threshold=30, softness=0.5, codec='png', 
                                 fps=None, show_preview=False):
    """
    Main function to process video and create transparent version
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps is None:
        fps = orig_fps
    
    print(f"Video Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Original FPS: {orig_fps:.2f}")
    print(f"  Output FPS: {fps:.2f}")
    
    # Detect background
    background = detect_static_background(video_path, background_threshold)
    if background is None:
        print("Failed to detect background")
        return False
    
    # Save background for reference
    cv2.imwrite(str(Path(output_path).parent / "detected_background.jpg"), background)
    
    # Prepare video writer for RGBA output
    fourcc = None
    output_ext = Path(output_path).suffix.lower()
    
    if output_ext == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif output_ext == '.mov':
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    elif output_ext == '.webm':
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
    else:
        # Default to PNG sequence if codec not supported
        print("Note: For best transparency, using PNG image sequence")
        codec = 'png'
    
    writer = None
    frame_dir = None
    
    if codec == 'png':
        # Create directory for PNG sequence
        frame_dir = Path(output_path).with_suffix('')
        frame_dir.mkdir(exist_ok=True)
        print(f"Saving PNG sequence to: {frame_dir}")
    else:
        # Create video writer
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
        if not writer.isOpened():
            print(f"Warning: Could not create video writer for {output_path}")
            print("Falling back to PNG sequence")
            codec = 'png'
            frame_dir = Path(output_path).with_suffix('')
            frame_dir.mkdir(exist_ok=True)
    
    # Process frames
    print("\nProcessing frames...")
    frame_num = 0
    success_frames = 0
    
    with tqdm(total=total_frames, desc="Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGBA with transparency
            rgba = frame_to_rgba(frame, background, mask_threshold, softness)
            
            if show_preview and frame_num % 30 == 0:  # Show every 30th frame
                preview = cv2.resize(rgba, (width//2, height//2))
                cv2.imshow('Preview (Press Q to stop preview)', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    show_preview = False
                    cv2.destroyAllWindows()
            
            # Save frame
            if codec == 'png':
                frame_filename = frame_dir / f"frame_{frame_num:06d}.png"
                cv2.imwrite(str(frame_filename), rgba, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                writer.write(rgba)
            
            success_frames += 1
            frame_num += 1
            pbar.update(1)
    
    cap.release()
    if writer is not None:
        writer.release()
    
    if show_preview:
        cv2.destroyAllWindows()
    
    print(f"\nProcessing complete!")
    print(f"  Successfully processed: {success_frames}/{total_frames} frames")
    
    if codec == 'png':
        print(f"  PNG frames saved to: {frame_dir}")
        # Create a text file with instructions
        with open(frame_dir / "README.txt", 'w') as f:
            f.write(f"PNG sequence of transparent video\n")
            f.write(f"Total frames: {success_frames}\n")
            f.write(f"FPS: {fps}\n")
            f.write(f"Use video editing software to import as image sequence\n")
    
    return True

def process_frame_batch(frame_batch, background, mask_threshold, softness, output_dir, start_idx):
    """
    Process a batch of frames (for parallel processing)
    """
    results = []
    for i, frame in enumerate(frame_batch):
        rgba = frame_to_rgba(frame, background, mask_threshold, softness)
        frame_filename = output_dir / f"frame_{start_idx + i:06d}.png"
        cv2.imwrite(str(frame_filename), rgba, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        results.append(frame_filename)
    return results

def process_video_parallel(video_path, output_dir, background_threshold=10, 
                          mask_threshold=30, softness=0.5, num_processes=4):
    """
    Parallel version for faster processing
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Detect background
    background = detect_static_background(video_path, background_threshold)
    if background is None:
        return False
    
    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Read all frames
    print("Reading all frames...")
    frames = []
    with tqdm(total=total_frames, desc="Reading frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            pbar.update(1)
    
    cap.release()
    
    # Split frames into batches
    batch_size = len(frames) // num_processes
    frame_batches = []
    for i in range(0, len(frames), batch_size):
        frame_batches.append(frames[i:i + batch_size])
    
    print(f"Processing {len(frames)} frames with {num_processes} processes...")
    
    # Process in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        process_func = partial(process_frame_batch, 
                             background=background,
                             mask_threshold=mask_threshold,
                             softness=softness,
                             output_dir=output_path)
        
        results = list(tqdm(pool.imap(process_func, frame_batches, 
                                     chunksize=1),
                          total=len(frame_batches),
                          desc="Processing batches"))
    
    print(f"Processed {len(frames)} frames to {output_path}")
    return True

class VideoBackgroundRemover:
    """
    Class-based approach for more advanced background removal
    """
    def __init__(self, method='mog2'):
        self.method = method
        self.background = None
        self.cap = None
        
        if method == 'mog2':
            self.back_sub = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=False
            )
        elif method == 'knn':
            self.back_sub = cv2.createBackgroundSubtractorKNN(
                history=500,
                dist2Threshold=400,
                detectShadows=False
            )
    
    def load_video(self, video_path):
        """Load video file"""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        return self.cap
    
    def estimate_background(self, sample_rate=10, learning_rate=0.01):
        """Estimate background from video"""
        if self.cap is None:
            raise ValueError("No video loaded")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        backgrounds = []
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Apply background subtraction
                fg_mask = self.back_sub.apply(frame, learningRate=learning_rate)
                
                # Get background
                bg = self.back_sub.getBackgroundImage()
                if bg is not None:
                    backgrounds.append(bg)
            
            frame_count += 1
        
        if backgrounds:
            self.background = np.median(backgrounds, axis=0).astype(np.uint8)
        
        return self.background
    
    def remove_background_frame(self, frame, threshold=30, 
                                use_color_diff=False, blur_radius=5):
        """Remove background from a single frame"""
        if self.background is None:
            return frame
        
        if use_color_diff:
            # Color-based difference (more accurate but slower)
            diff = cv2.absdiff(frame, self.background)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
        else:
            # Grayscale difference (faster)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bg_gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(frame_gray, bg_gray)
            _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply blur for soft edges
        if blur_radius > 0:
            mask = cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), 0)
        
        # Create RGBA image
        b, g, r = cv2.split(frame)
        rgba = cv2.merge([b, g, r, mask])
        
        return rgba
    
    def process_video(self, output_path, threshold=30, progress_callback=None):
        """Process entire video"""
        if self.cap is None or self.background is None:
            raise ValueError("Video not loaded or background not estimated")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output directory for PNG sequence
        output_dir = Path(output_path).with_suffix('')
        output_dir.mkdir(exist_ok=True)
        
        frame_count = 0
        
        with tqdm(total=total_frames, desc="Processing") as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                rgba = self.remove_background_frame(frame, threshold)
                
                # Save frame
                frame_filename = output_dir / f"frame_{frame_count:06d}.png"
                cv2.imwrite(str(frame_filename), rgba)
                
                frame_count += 1
                pbar.update(1)
                
                if progress_callback:
                    progress_callback(frame_count, total_frames)
        
        self.cap.release()
        
        # Create video from PNG sequence
        self.create_video_from_sequence(output_dir, output_path, fps)
        
        return output_dir
    
    def create_video_from_sequence(self, input_dir, output_path, fps):
        """Create video from PNG sequence"""
        png_files = sorted(input_dir.glob("*.png"))
        if not png_files:
            return
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(png_files[0]), cv2.IMREAD_UNCHANGED)
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
        
        print("Creating video from PNG sequence...")
        for png_file in tqdm(png_files, desc="Creating video"):
            frame = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
            
            # Convert RGBA to BGR for video writer
            if frame.shape[2] == 4:
                # Split alpha channel
                b, g, r, a = cv2.split(frame)
                
                # Create white background
                white_bg = np.ones_like(b) * 255
                
                # Composite with white background for preview
                alpha = a.astype(float) / 255.0
                b = (b * alpha + white_bg * (1 - alpha)).astype(np.uint8)
                g = (g * alpha + white_bg * (1 - alpha)).astype(np.uint8)
                r = (r * alpha + white_bg * (1 - alpha)).astype(np.uint8)
                
                frame = cv2.merge([b, g, r])
            
            writer.write(frame)
        
        writer.release()
        print(f"Video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Remove static background from video and make it transparent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.mp4 output.mp4
  %(prog)s input.mp4 output --threshold 20 --softness 1.0
  %(prog)s input.mp4 output.png --preview
  %(prog)s input.mp4 output --parallel --processes 8
        """
    )
    
    parser.add_argument('input', help='Input video file')
    parser.add_argument('output', help='Output file or directory')
    
    parser.add_argument('--bg-threshold', type=int, default=10,
                       help='Background detection threshold (default: 10)')
    parser.add_argument('--mask-threshold', type=int, default=30,
                       help='Mask creation threshold (default: 30)')
    parser.add_argument('--softness', type=float, default=0.5,
                       help='Edge softness (0.0-2.0, default: 0.5)')
    parser.add_argument('--fps', type=float, 
                       help='Output FPS (default: same as input)')
    parser.add_argument('--preview', action='store_true',
                       help='Show preview during processing')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel processing (for PNG sequences)')
    parser.add_argument('--processes', type=int, default=4,
                       help='Number of parallel processes (default: 4)')
    parser.add_argument('--method', choices=['mog2', 'knn'], default='mog2',
                       help='Background subtraction method (default: mog2)')
    
    args = parser.parse_args()
    
    try:
        print("Starting video background removal...")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        
        if args.parallel:
            # Use parallel processing for PNG sequence
            output_dir = args.output if Path(args.output).suffix == '' else \
                        Path(args.output).with_suffix('')
            
            success = process_video_parallel(
                video_path=args.input,
                output_dir=output_dir,
                background_threshold=args.bg_threshold,
                mask_threshold=args.mask_threshold,
                softness=args.softness,
                num_processes=args.processes
            )
        else:
            # Use class-based approach
            remover = VideoBackgroundRemover(method=args.method)
            remover.load_video(args.input)
            
            print("Estimating background...")
            background = remover.estimate_background()
            if background is None:
                print("Failed to estimate background")
                sys.exit(1)
            
            # Save detected background
            bg_path = Path(args.output).parent / "detected_background.jpg"
            cv2.imwrite(str(bg_path), background)
            print(f"Detected background saved to: {bg_path}")
            
            print("Processing video...")
            output_dir = remover.process_video(
                output_path=args.output,
                threshold=args.mask_threshold
            )
            
            print(f"\nProcessing complete!")
            print(f"Transparent frames saved to: {output_dir}")
            print(f"Use these PNG files in your video editing software")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()