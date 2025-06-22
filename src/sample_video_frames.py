import cv2
import os
import argparse
import numpy as np


def preprocess_frame_lighting(frame, enable_clahe=False, clahe_clip_limit=2.0, clahe_tile_size=8,
                             gamma_correction=None, exposure_adjustment=None, 
                             brightness_adjustment=None, contrast_adjustment=None,
                             enhance_texture=False, texture_enhancement_strength=1.5,
                             reduce_bright_spots=False):
    """
    Apply various lighting corrections to improve frame quality for 3D reconstruction.
    
    Args:
        frame: Input frame (BGR)
        enable_clahe: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe_clip_limit: Clip limit for CLAHE (default: 2.0)
        clahe_tile_size: Tile grid size for CLAHE (default: 8x8)
        gamma_correction: Gamma value for correction (1.0 = no change, <1.0 = brighter, >1.0 = darker)
        exposure_adjustment: Exposure adjustment (-100 to +100, negative = darker, positive = brighter)
        brightness_adjustment: Brightness adjustment (-100 to +100)
        contrast_adjustment: Contrast adjustment (0.5 to 3.0, 1.0 = no change)
        enhance_texture: Apply texture enhancement for featureless surfaces
        texture_enhancement_strength: Strength of texture enhancement (1.0-3.0, default: 1.5)
        reduce_bright_spots: Apply preset parameters optimized for reducing specular reflections and bright spots
    
    Returns:
        Preprocessed frame
    """
    processed = frame.copy()
    
    # Apply bright spot reduction preset if enabled
    if reduce_bright_spots:
        # Override parameters with optimal settings for bright spot reduction
        enable_clahe = True
        clahe_clip_limit = 1.3
        gamma_correction = 1.5
        exposure_adjustment = -30
        contrast_adjustment = 0.8
        print("Applied bright spot reduction preset: CLAHE(1.3), gamma(1.5), exposure(-30), contrast(0.8)")
    
    # Exposure adjustment (simple linear scaling)
    if exposure_adjustment is not None and exposure_adjustment != 0:
        # Convert exposure adjustment to scaling factor
        # -100 to +100 maps to 0.1 to 2.0
        scale = 1.0 + (exposure_adjustment / 100.0)
        scale = max(0.1, min(2.0, scale))  # Clamp to reasonable range
        processed = np.clip(processed.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    
    # Brightness adjustment
    if brightness_adjustment is not None and brightness_adjustment != 0:
        processed = np.clip(processed.astype(np.int16) + brightness_adjustment, 0, 255).astype(np.uint8)
    
    # Contrast adjustment
    if contrast_adjustment is not None and contrast_adjustment != 1.0:
        processed = np.clip(processed.astype(np.float32) * contrast_adjustment, 0, 255).astype(np.uint8)
    
    # Gamma correction
    if gamma_correction is not None and gamma_correction != 1.0:
        # Build lookup table for gamma correction
        inv_gamma = 1.0 / gamma_correction
        lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        processed = cv2.LUT(processed, lookup_table)
    
    # Texture enhancement for featureless surfaces (like white objects)
    if enhance_texture:
        # Convert to LAB for better texture processing
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Method 1: Unsharp masking to enhance subtle details
        gaussian = cv2.GaussianBlur(l_channel, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(l_channel, 1.0 + texture_enhancement_strength, gaussian, -texture_enhancement_strength, 0)
        
        # Method 2: High-pass filter to enhance fine details
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * 0.15 * texture_enhancement_strength
        high_pass = cv2.filter2D(l_channel, -1, kernel)
        enhanced_l = cv2.addWeighted(unsharp_mask, 0.7, high_pass, 0.3, 0)
        
        # Clamp values
        enhanced_l = np.clip(enhanced_l, 0, 255).astype(np.uint8)
        
        # Merge back
        lab_enhanced = cv2.merge([enhanced_l, a_channel, b_channel])
        processed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # CLAHE (applied to luminance channel to preserve color)
    if enable_clahe:
        # Convert to LAB color space to work on luminance channel only
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to luminance channel
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size))
        l_channel = clahe.apply(l_channel)
        
        # Merge back and convert to BGR
        lab = cv2.merge([l_channel, a_channel, b_channel])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return processed


def calculate_laplacian_variance(image):
    """
    Calculate the Laplacian variance of an image to measure blur.
    Higher values indicate sharper images, lower values indicate blurry images.
    
    Args:
        image: Input image (BGR or grayscale)
    
    Returns:
        float: Laplacian variance (higher = sharper)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian and return variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def is_frame_sharp(frame, blur_threshold=100.0):
    """
    Determine if a frame is sharp enough based on Laplacian variance.
    
    Args:
        frame: Input frame
        blur_threshold: Minimum Laplacian variance for a frame to be considered sharp
    
    Returns:
        bool: True if frame is sharp, False if blurry
        float: The calculated Laplacian variance
    """
    variance = calculate_laplacian_variance(frame)
    return variance >= blur_threshold, variance


def sample_video_frames(video_path, n_images, output_dir, blur_threshold=100.0, max_attempts_multiplier=3,
                       enable_lighting_correction=False, **lighting_kwargs):
    """
    Sample frames from video, filtering out blurry ones using Laplacian variance.
    Uses a two-pass approach to maintain temporal uniformity:
    1. First pass: Sample candidate frames uniformly across the video
    2. Second pass: Select the sharpest frames while preserving temporal distribution
    
    Args:
        video_path: Path to input video
        n_images: Number of sharp images to extract
        output_dir: Output directory for images
        blur_threshold: Minimum Laplacian variance for sharp images
        max_attempts_multiplier: How many times n_images to attempt before giving up
        enable_lighting_correction: Apply lighting preprocessing to frames
        **lighting_kwargs: Parameters for lighting correction (see preprocess_frame_lighting)
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # We'll sample more frames than needed to account for blurry ones
    max_attempts = min(total_frames, n_images * max_attempts_multiplier)
    
    lighting_info = ""
    if enable_lighting_correction:
        corrections = []
        if lighting_kwargs.get('enable_clahe', False):
            corrections.append(f"CLAHE(clip={lighting_kwargs.get('clahe_clip_limit', 2.0)})")
        if lighting_kwargs.get('gamma_correction') is not None:
            corrections.append(f"gamma={lighting_kwargs.get('gamma_correction')}")
        if lighting_kwargs.get('exposure_adjustment') is not None:
            corrections.append(f"exposure={lighting_kwargs.get('exposure_adjustment')}")
        if lighting_kwargs.get('brightness_adjustment') is not None:
            corrections.append(f"brightness={lighting_kwargs.get('brightness_adjustment')}")
        if lighting_kwargs.get('contrast_adjustment') is not None:
            corrections.append(f"contrast={lighting_kwargs.get('contrast_adjustment')}")
        lighting_info = f" with lighting correction: {', '.join(corrections)}" if corrections else ""
    
    print(f"Sampling {n_images} sharp frames from {total_frames} total frames (blur threshold: {blur_threshold}){lighting_info}")
    
    # PHASE 1: Sample candidate frames uniformly and evaluate sharpness
    candidates = []  # List of (frame_index, variance, frame_data)

    # Compute frame indices to sample uniformly across the video
    indices = [int(i * total_frames / max_attempts) for i in range(max_attempts)]
    indices = list(sorted(set(indices)))  # Remove duplicates if any

    frame_id = 0
    next_idx = indices.pop(0) if indices else None
    
    while cap.isOpened() and next_idx is not None:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_id == next_idx:
            # Apply lighting correction if enabled
            if enable_lighting_correction:
                frame = preprocess_frame_lighting(frame, **lighting_kwargs)
            
            # Evaluate sharpness
            is_sharp, variance = is_frame_sharp(frame, blur_threshold)
            candidates.append((frame_id, variance, frame.copy()))
            
            # Progress indicator every 25 candidates
            if len(candidates) % 25 == 0:
                print(f"Evaluated {len(candidates)} candidate frames...")
            
            # Move to next frame index
            if indices:
                next_idx = indices.pop(0)
            else:
                break
                
        frame_id += 1
    
    cap.release()
    
    # PHASE 2: Select best frames while maintaining temporal distribution
    if len(candidates) == 0:
        print("No candidate frames found!")
        return
    
    # Sort candidates by sharpness (variance) in descending order
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Filter out frames below blur threshold first
    sharp_candidates = [c for c in candidates if c[1] >= blur_threshold]
    
    if len(sharp_candidates) >= n_images:
        # We have enough sharp frames, now select for temporal uniformity
        selected_frames = select_temporally_uniform_frames(sharp_candidates, n_images)
    else:
        # Not enough sharp frames, use all sharp ones plus best blurry ones
        remaining_needed = n_images - len(sharp_candidates)
        blurry_candidates = [c for c in candidates if c[1] < blur_threshold]
        blurry_candidates.sort(key=lambda x: x[1], reverse=True)  # Best blurry frames first
        
        selected_frames = sharp_candidates + blurry_candidates[:remaining_needed]
    
    # Sort selected frames by time for consistent output naming
    selected_frames.sort(key=lambda x: x[0])
    
    # PHASE 3: Save selected frames
    saved = 0
    for i, (frame_idx, variance, frame_data) in enumerate(selected_frames):
        out_path = os.path.join(output_dir, f"frame_{i+1:04d}.jpg")
        cv2.imwrite(out_path, frame_data)
        saved += 1
    
    # Summary
    sharp_count = sum(1 for _, variance, _ in selected_frames if variance >= blur_threshold)
    blurry_count = len(selected_frames) - sharp_count
    
    print(f"Saved {saved} frames: {sharp_count} sharp, {blurry_count} blurry")
    if blurry_count > 0:
        print(f"Note: {blurry_count} frames were below blur threshold (consider lowering threshold or improving video quality)")


def select_temporally_uniform_frames(candidates, n_images):
    """
    Select n_images from candidates to maximize temporal uniformity.
    Uses a greedy approach to select frames that are well-distributed in time.
    
    Args:
        candidates: List of (frame_index, variance, frame_data) sorted by variance (best first)
        n_images: Number of frames to select
    
    Returns:
        List of selected (frame_index, variance, frame_data) tuples
    """
    if len(candidates) <= n_images:
        return candidates
    
    # Start with the sharpest frame
    selected = [candidates[0]]
    remaining = candidates[1:]
    
    # Greedily select frames that maximize minimum temporal distance
    for _ in range(n_images - 1):
        if not remaining:
            break
            
        best_candidate = None
        best_min_distance = -1
        best_idx = -1
        
        for i, candidate in enumerate(remaining):
            candidate_time = candidate[0]
            
            # Calculate minimum temporal distance to already selected frames
            min_distance = min(abs(candidate_time - selected_frame[0]) for selected_frame in selected)
            
            # Prefer candidates with larger minimum distance (better temporal separation)
            # If tied, prefer sharper frames (they're already sorted by variance)
            if min_distance > best_min_distance:
                best_min_distance = min_distance
                best_candidate = candidate
                best_idx = i
        
        if best_candidate:
            selected.append(best_candidate)
            remaining.pop(best_idx)
    
    return selected


def main():
    parser = argparse.ArgumentParser(description="Sample N sharp frames uniformly from a video, filtering out blurry ones.")
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--n_images', type=int, required=True, help='Number of sharp images to sample')
    parser.add_argument('--output', type=str, required=True, help='Output directory for images')
    parser.add_argument('--blur-threshold', type=float, default=100.0,
                       help='Laplacian variance threshold for blur detection (default: 100.0, lower = more permissive)')
    parser.add_argument('--max-attempts-multiplier', type=int, default=3,
                       help='How many times n_images to sample before giving up (default: 3)')
    parser.add_argument('--enable-lighting-correction', action='store_true',
                       help='Apply lighting preprocessing to frames')
    parser.add_argument('--enable-clahe', action='store_true',
                       help='Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)')
    parser.add_argument('--clahe-clip-limit', type=float, default=2.0,
                       help='Clip limit for CLAHE (default: 2.0)')
    parser.add_argument('--clahe-tile-size', type=int, default=8,
                       help='Tile grid size for CLAHE (default: 8x8)')
    parser.add_argument('--gamma-correction', type=float,
                       help='Gamma value for correction (1.0 = no change, <1.0 = brighter, >1.0 = darker)')
    parser.add_argument('--exposure-adjustment', type=int,
                       help='Exposure adjustment (-100 to +100, negative = darker, positive = brighter)')
    parser.add_argument('--brightness-adjustment', type=int,
                       help='Brightness adjustment (-100 to +100)')
    parser.add_argument('--contrast-adjustment', type=float,
                       help='Contrast adjustment (0.5 to 3.0, 1.0 = no change)')
    parser.add_argument('--enhance-texture', action='store_true',
                       help='Apply texture enhancement for featureless surfaces')
    parser.add_argument('--texture-enhancement-strength', type=float, default=1.5,
                       help='Strength of texture enhancement (1.0-3.0, default: 1.5)')
    parser.add_argument('--reduce-bright-spots', action='store_true',
                       help='Apply preset parameters optimized for reducing specular reflections and bright spots')
    
    args = parser.parse_args()
    sample_video_frames(args.video, args.n_images, args.output, 
                       args.blur_threshold, args.max_attempts_multiplier,
                       args.enable_lighting_correction,
                       enable_clahe=args.enable_clahe,
                       clahe_clip_limit=args.clahe_clip_limit,
                       clahe_tile_size=args.clahe_tile_size,
                       gamma_correction=args.gamma_correction,
                       exposure_adjustment=args.exposure_adjustment,
                       brightness_adjustment=args.brightness_adjustment,
                       contrast_adjustment=args.contrast_adjustment,
                       enhance_texture=args.enhance_texture,
                       texture_enhancement_strength=args.texture_enhancement_strength,
                       reduce_bright_spots=args.reduce_bright_spots)

if __name__ == '__main__':
    main()
