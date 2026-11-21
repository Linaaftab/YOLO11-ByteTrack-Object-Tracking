
# Experimental Report

## Real-Time Object Detection and Multi-Object Tracking Using YOLO11 and ByteTrack

### 1. Objective

The objective of this experiment is to implement and evaluate a video-based object detection and multi-object tracking pipeline using YOLO11 and ByteTrack.

The system detects objects in video frames, associates detections across consecutive frames, assigns tracking IDs, and records performance metrics.

### 2. Experimental Environment

| Component | Configuration |
|---|---|
| Operating System | Windows |
| Programming Language | Python 3.11.9 |
| Deep Learning Framework | PyTorch |
| Object Detection Model | YOLO11n |
| Tracking Algorithm | ByteTrack |
| Computer Vision Library | OpenCV |
| Data Analysis | Pandas |
| Visualization | Matplotlib |
| Inference Device | CPU |
| GPU Acceleration | Not used |

The system was tested without NVIDIA GPU acceleration. Therefore, the measured inference speed reflects CPU-based execution.

### 3. Dataset and Input Video

A sample video was used to test the detection and tracking pipeline.

| Property | Value |
|---|---|
| Video Resolution | 640 × 360 |
| Frame Rate | 30 FPS |
| Total Frames | 62 |
| Video Format | MP4 |

The video was used as an input for pipeline testing. It was not a formally annotated tracking benchmark dataset.

### 4. Model Configuration

The experiment used the following configuration:

| Parameter | Value |
|---|---|
| Model | YOLO11n |
| Confidence Threshold | 0.35 |
| Input Image Size | 640 |
| Tracker | ByteTrack |
| Device | CPU |
| Tracking Persistence | Enabled |

The YOLO11n model was used for object detection, while ByteTrack was responsible for associating detected objects across video frames.

### 5. Experimental Procedure

The experiment followed these steps:

1. Load the configuration file.
2. Initialize the pretrained YOLO11n model.
3. Read the input video frame by frame.
4. Perform object detection using YOLO11.
5. Apply ByteTrack to associate detections across frames.
6. Assign and maintain tracking IDs.
7. Draw bounding boxes and tracking IDs.
8. Save the processed video.
9. Record frame-level performance metrics.
10. Generate performance visualization graphs.

### 6. Performance Results

Two preliminary executions were recorded using the same 62-frame sample video.

| Metric | Run 1 | Run 2 |
|---|---:|---:|
| Total Frames | 62 | 62 |
| Processing Time | 7.59 seconds | 11.31 seconds |
| Average Processing Speed | 8.17 FPS | 5.48 FPS |

Run 1 also recorded:
- Unique tracking IDs: 21
- Average detection confidence: approximately 0.5813

The difference in processing speed indicates that CPU inference performance can vary between executions.

These measurements are preliminary observations, not results from a controlled benchmark.

### 7. Performance Visualization

The project generates the following graphs:

- Detection count per frame
- Average confidence per frame
- Processing time per frame

The graphs are available in:

`results/plots/`

### 8. Discussion

The experiment demonstrates that YOLO11 and ByteTrack can be integrated into a functional object detection and multi-object tracking pipeline.

YOLO11 identifies objects in individual frames, while ByteTrack associates detections across frames to maintain object identities.

The preliminary results demonstrate successful video processing and tracking on a CPU-based system.

However, the current experiment has limitations. The sample video is relatively short, and no ground-truth tracking annotations are available. Consequently, formal tracking accuracy metrics cannot yet be calculated.

### 9. Limitations

1. The experiment uses a small sample video.
2. Inference is performed using a CPU rather than a dedicated GPU.
3. The current evaluation does not include ground-truth annotations.
4. Tracking accuracy metrics such as MOTA, IDF1, and HOTA have not been calculated.
5. The measured FPS may vary depending on system conditions.

### 10. Future Improvements

Future work will focus on:

- Evaluating the system using MOT17 or MOT20 benchmark datasets.
- Calculating MOTA, IDF1, HOTA, and ID switches.
- Comparing different YOLO model sizes.
- Testing alternative tracking algorithms.
- Improving inference speed.
- Conducting repeated controlled experiments.
- Evaluating performance under different video conditions.

### 11. Conclusion

This experiment successfully demonstrates the integration of YOLO11 object detection and ByteTrack multi-object tracking.

The implementation provides a foundation for further research in computer vision, particularly in video analytics, object tracking, and intelligent visual systems.

Further evaluation using annotated benchmark datasets is necessary to establish tracking accuracy and make reliable comparisons with other approaches.
