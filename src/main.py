import cv2
import yaml
import time
import torch
from pathlib import Path
from ultralytics import YOLO
import csv
from collections import defaultdict
import pandas as pd

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    """Load project settings from config.yaml."""
    config_path = PROJECT_ROOT / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def initialize_model(config):
    """Load the YOLO model."""
    model_name = config["model"]["name"]
    model_path = PROJECT_ROOT / "models" / model_name

    print(f"Loading YOLO model: {model_name}")

    model = YOLO(str(model_path) if model_path.exists() else model_name)

    print("YOLO model loaded successfully!")

    return model
def track_video(model, config):
    """Detect and track multiple objects in a video."""

    input_video = PROJECT_ROOT / config["video"]["input"]
    output_video = PROJECT_ROOT / config["video"]["output"]

    # Check whether input video exists
    if not input_video.exists():
        print(f"Error: Video not found at {input_video}")
        print("Please place your test video inside data/videos/")
        return

    # Create output directory
    output_video.parent.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(input_video))

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video resolution: {width}x{height}")
    print(f"Original FPS: {fps:.2f}")

    # Prepare output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps if fps > 0 else 25,
        (width, height)
    )

    frame_count = 0
    start_time = time.time()

    # Store frame-by-frame tracking metrics
    frame_metrics = []
    all_tracking_ids = set()

    print("Starting YOLO + ByteTrack...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

                # Start frame processing timer
        frame_start = time.time()

        # Run object detection and tracking
        results = model.track(
            frame,
            persist=True,
            tracker=config["tracker"]["name"],
            conf=config["model"]["confidence"],
            imgsz=config["model"]["image_size"],
            device=config["device"]["type"],
            verbose=False
        )

        # Collect tracking metrics
        boxes = results[0].boxes
        tracking_ids = []

        if boxes is not None and boxes.id is not None:
            tracking_ids = boxes.id.int().cpu().tolist()
            all_tracking_ids.update(tracking_ids)

        confidences = (
            boxes.conf.cpu().tolist()
            if boxes is not None and boxes.conf is not None
            else []
        )

        frame_metrics.append({
            "frame": frame_count + 1,
            "detection_count": len(boxes) if boxes is not None else 0,
            "tracked_object_count": len(tracking_ids),
            "tracking_ids": ";".join(map(str, tracking_ids)),
            "average_confidence": (
                sum(confidences) / len(confidences) if confidences else 0
            ),
            "processing_time_sec": time.time() - frame_start
        })

        # Draw detection boxes and tracking IDs
        annotated_frame = results[0].plot()

        # Save processed frame
        out.write(annotated_frame)

        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    # Release resources
    cap.release()
    out.release()

    elapsed_time = time.time() - start_time
    processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    print("\nTracking completed!")
    print(f"Total frames processed: {frame_count}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Average processing FPS: {processing_fps:.2f}")
    print(f"Output saved to: {output_video}")
        # Save frame-by-frame tracking metrics
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    frame_metrics_df = pd.DataFrame(frame_metrics)
    frame_metrics_df.to_csv(
        metrics_dir / "frame_metrics.csv",
        index=False
    )

    print(f"Frame metrics saved to: {metrics_dir / 'frame_metrics.csv'}")
        # Save summary metrics
    summary_metrics = {
        "total_frames": frame_count,
        "unique_tracking_ids": len(all_tracking_ids),
        "processing_time_sec": elapsed_time,
        "average_fps": processing_fps,
        "average_confidence": (
            frame_metrics_df["average_confidence"].mean()
            if not frame_metrics_df.empty else 0
        )
    }

    summary_df = pd.DataFrame([summary_metrics])
    summary_df.to_csv(
        metrics_dir / "summary_metrics.csv",
        index=False
    )

    print(f"Summary metrics saved to: {metrics_dir / 'summary_metrics.csv'}")
def main():
    """Main function to run the project."""

    print("=" * 55)
    print(" REAL-TIME OBJECT DETECTION AND MULTI-OBJECT TRACKING")
    print("              YOLO + ByteTrack")
    print("=" * 55)

    config = load_config()

    print(f"Using device: {config['device']['type']}")
    print(f"PyTorch version: {torch.__version__}")

    model = initialize_model(config)

    track_video(model, config)


if __name__ == "__main__":
    main()
