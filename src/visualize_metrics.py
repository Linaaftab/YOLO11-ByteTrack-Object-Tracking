import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load frame-by-frame metrics
data = pd.read_csv(METRICS_DIR / "frame_metrics.csv")

# Plot 1: Detection count per frame
plt.figure(figsize=(10, 5))
plt.plot(data["frame"], data["detection_count"])
plt.title("Object Detection Count per Frame")
plt.xlabel("Frame Number")
plt.ylabel("Number of Detections")
plt.grid(True)
plt.savefig(PLOTS_DIR / "detection_count.png", dpi=300)
plt.close()

# Plot 2: Average confidence per frame
plt.figure(figsize=(10, 5))
plt.plot(data["frame"], data["average_confidence"])
plt.title("Average Detection Confidence per Frame")
plt.xlabel("Frame Number")
plt.ylabel("Average Confidence")
plt.grid(True)
plt.savefig(PLOTS_DIR / "average_confidence.png", dpi=300)
plt.close()

# Plot 3: Processing time per frame
plt.figure(figsize=(10, 5))
plt.plot(data["frame"], data["processing_time_sec"])
plt.title("Processing Time per Frame")
plt.xlabel("Frame Number")
plt.ylabel("Processing Time (seconds)")
plt.grid(True)
plt.savefig(PLOTS_DIR / "processing_time.png", dpi=300)
plt.close()

print("All graphs generated successfully!")
print(f"Graphs saved in: {PLOTS_DIR}")