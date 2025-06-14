import cv2
import os
import argparse


def sample_video_frames(video_path, n_images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_images > total_frames:
        print(f"Warning: Requested {n_images} images, but video only has {total_frames} frames. Sampling all frames.")
        n_images = total_frames

    # Compute frame indices to sample uniformly
    indices = [int(i * total_frames / n_images) for i in range(n_images)]
    indices = list(sorted(set(indices)))  # Remove duplicates if any

    frame_id = 0
    saved = 0
    next_idx = indices.pop(0) if indices else None
    while cap.isOpened() and next_idx is not None:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id == next_idx:
            out_path = os.path.join(output_dir, f"frame_{saved+1:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
            if indices:
                next_idx = indices.pop(0)
            else:
                break
        frame_id += 1
    cap.release()
    print(f"Saved {saved} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sample N frames uniformly from a video.")
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--n_images', type=int, required=True, help='Number of images to sample')
    parser.add_argument('--output', type=str, required=True, help='Output directory for images')
    args = parser.parse_args()
    sample_video_frames(args.video, args.n_images, args.output)

if __name__ == '__main__':
    main()
