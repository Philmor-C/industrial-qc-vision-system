<div align="center">

<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PySide2-Qt5-41CD52?style=for-the-badge&logo=qt&logoColor=white"/>
<img src="https://img.shields.io/badge/ONNX_Runtime-OpenVINO-005CED?style=for-the-badge&logo=onnx&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/YOLOv5-Patch_Inference-FF6F00?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-GPL--3.0-green?style=for-the-badge"/>

<br/><br/>

# 🏭 Industrial QC Vision System

### A production-grade offline desktop application for automated surface defect detection and sub-pixel dimensional measurement of industrial parts

<br/>

[**View Demo Video**](#-demo) · [**Architecture**](#-system-architecture) · [**Features**](#-key-features) · [**Quick Start**](#-quick-start) · [**Technical Deep Dive**](#-technical-deep-dive)

</div>

---

## 📌 Overview

This system was designed and built from scratch as a complete industrial quality control solution deployable on factory-floor hardware — no internet connection, no cloud dependency, no external API calls. It runs entirely offline on local compute.

The application combines two inspection disciplines in a single Qt5 desktop tool:

- **Defect Detection** — YOLO-based patch inference with OpenVINO-accelerated ONNX Runtime, detecting scratches, dents, cracks, stains, missing parts, and deformations across high-resolution industrial images
- **Dimensional Measurement** — Sub-pixel edge detection with Huber robust line fitting, measuring inter-feature distances across large-format parts with operator-placed ROI points

Both pipelines feed into an **Advanced Analytics** module with 8 interactive Plotly-based analysis views — Pareto, morphological, spatial pattern, severity scoring, trend tracking, density analysis, root cause correlation, and statistical box plots.

---

## 🎬 Demo

> **📹 Demo video coming soon** — full walkthrough of the inspection workflow, defect detection results, measurement pipeline, and advanced analytics dashboard.

<!-- Replace this block when your video is ready:
[![Demo Video](https://img.shields.io/badge/▶_Watch_Demo-YouTube-FF0000?style=for-the-badge&logo=youtube)](YOUR_VIDEO_LINK_HERE)
-->

---

## ✅ Key Features

| Feature | Detail |
|---|---|
| **Patch-based YOLO inference** | Tiles any resolution image into overlapping 1280×1280 patches; detects defects across full high-res industrial scans |
| **OpenVINO acceleration** | Primary provider with automatic CPU fallback — no code change needed between hardware configs |
| **Dynamic class support** | Postprocess reads actual model output dimensions — works with any number of defect classes |
| **Sub-pixel edge detection** | Vectorised horizontal gradient scan with scipy convolution; detects both light→dark and dark→light transitions |
| **Huber robust line fitting** | Single-pass robust fitting replaces RANSAC — faster, more stable, identical output interface |
| **Canny + morphological ROI analysis** | Median-adaptive Canny thresholds + dilate/close/open for reliable defect boundary extraction on any surface |
| **Interactive ROI editor** | Qt5 graphics scene with click-to-add, drag-to-move, delete modes; Y-alignment tools for horizontal scan lines |
| **Per-pair measurement standards** | Operator-defined expected distance + tolerance per point pair; automatic OK/FAIL verdict |
| **8 advanced analytics views** | Pareto, morphological, spatial heatmap, severity scoring, trend tracking, density, root cause, box plots |
| **Fully offline** | Zero network calls at runtime — suitable for air-gapped factory environments |
| **CSV + JSON export** | Structured defect metadata (bounding boxes, contour metrics, confidence) exportable per inspection run |
| **Background preload** | Silent background detection on startup; results cached and served instantly on demand |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OPERATOR  (PySide2 Qt5)                  │
├──────────────┬──────────────────────────┬───────────────────┤
│  Setup & ROI │   Inspection Report      │  Advanced Analysis│
│     Tab      │        Tab               │       Tab         │
│              │  QWebEngineView          │  QWebEngineView   │
│ ROIEditor    │  (HTML results)          │  (Plotly charts)  │
├──────────────┴──────────┬───────────────┴───────────────────┤
│                 InspectionWorker (QThread)                   │
│          Keeps UI responsive during heavy processing         │
├─────────────────┬───────┴──────────────┬────────────────────┤
│ defect_         │ measurement_         │ advanced_          │
│ pipeline.py     │ pipeline.py          │ analysis.py        │
│                 │                      │                    │
│ YOLOPatchDetect │ Vectorised edge scan │ 8 Plotly analyses  │
│ OpenVINO→CPU   │ scipy convolve2d     │ SciPy spatial stats│
│ Canny ROI      │ Huber line fitting   │ Trend JSON history │
├─────────────────┴──────────────────────┴────────────────────┤
│           config.py  ·  utils.py  ·  ONNX model             │
├─────────────────────────────────────────────────────────────┤
│  samples/  ·  measurement_samples/  ·  detected_defects/    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🖥 Application Walkthrough

### Tab 1 — Setup & ROI Editor

The operator loads a measurement image and interactively places numbered measurement points directly on the part. Points can be added, dragged, or deleted with three selectable modes. A Y-alignment tool snaps all points to a common horizontal scan line — a requirement of the measurement algorithm. Once points are confirmed, expected distances and tolerances are entered per consecutive point pair using the Standards dialog before any inspection is run.

<br/>

<img src="https://raw.githubusercontent.com/Philmor-C/industrial-qc-vision-system/main/assets/roi_editor.JPG" alt="ROI Editor — interactive measurement point placement on part image" width="100%"/>

> *Operator places numbered measurement points on the measurement image. The system validates that all points share a common Y coordinate before enabling the measurement run. Point positions, line connections, and coordinates are shown in real time in the side panel.*

---

### Tab 2 — Dimensional Measurement

The measurement pipeline runs at full image resolution, applying vectorised edge detection across each 250×200px ROI window centred on each operator-placed point. Left and right edges are extracted using a scipy convolution gradient kernel across all scan strips simultaneously. Fitted edge lines are drawn in green (left edges) and blue (right edges). Inter-feature distances are computed between consecutive ROI pairs and annotated directly on the scaled output image with alternating TOP/BOTTOM positioning for readability. Each measurement is automatically compared against the operator-defined standard and tolerance, returning OK or FAIL per pair.

<br/>

<img src="https://raw.githubusercontent.com/Philmor-C/industrial-qc-vision-system/main/assets/measurment_result.JPG" alt="Measurement result — sub-pixel edge detection with distance annotations" width="100%"/>

> *Sub-pixel edge detection and Huber robust line fitting applied across all ROI windows. Inter-feature distances annotated with alternating cyan/lime lines, endpoint markers, and distance values in pixels. Each pair compared against defined standard ± tolerance.*

---

### Tab 2 — Defect Detection

The defect detection pipeline tiles the input image into overlapping 1280×1280 patches and runs YOLO inference on each patch via ONNX Runtime with OpenVINO acceleration. Results from all patches are pooled and deduplicated using cross-patch NMS. Each surviving detection is expanded with a 2× context ROI and passed through a Canny + morphological contour analysis pipeline to extract boundary shape metrics — area, perimeter, circularity, and hull area.

<br/>

<img src="https://raw.githubusercontent.com/Philmor-C/industrial-qc-vision-system/main/assets/defect_01.JPG" alt="Defect detection — full annotated output image with bounding boxes" width="100%"/>

> *Inspection summary — images processed, total defect count, and average defects per image displayed as KPI cards. Collapsible per-image sections show original and annotated images side by side.*

<br/>

<img src="https://raw.githubusercontent.com/Philmor-C/industrial-qc-vision-system/main/assets/defect_02.JPG" alt="Defect detection — per-defect inspection table with ROI thumbnails" width="100%"/>

> *Full image annotated with colour-coded bounding boxes per defect class, confidence scores, and total detection count. Each class gets a consistent colour seeded for reproducibility.
Per-defect inspection table: ROI crop thumbnail, Canny contour visualization thumbnail, colour-coded defect class badge, confidence score, contour count, and total defect area in px². Each row has an Inspect button opening a full drill-down modal.*

<br/>

<img src="https://raw.githubusercontent.com/Philmor-C/industrial-qc-vision-system/main/assets/defect_03.JPG" alt="Defect detection — modal drill-down with full-size ROI and contour images" width="100%"/>

> *Modal inspection view for a single defect — full-size ROI image alongside the Canny contour visualization.*

<br/>

<img src="https://raw.githubusercontent.com/Philmor-C/industrial-qc-vision-system/main/assets/defect_04.JPG" alt="Defect detection — KPI summary and inspection overview" width="100%"/>

>  *Modal inspection view for a single defect — full-size ROI image alongside the Canny contour visualization. with original and expanded bounding box coordinates, confidence, contour count, area coverage percentage, and a per-contour breakdown table (circularity, perimeter, bounding rect).*
---

### Tab 3 — Advanced Analytics

All analytics run on the stored detection results with a single button click — no re-running inference. Charts render as fully interactive Plotly visualizations inside the embedded Qt browser, supporting zoom, hover tooltips, and pan. Eight analysis types are available, each targeting a specific quality engineering question.

<br/>

<img src="https://raw.githubusercontent.com/Philmor-C/industrial-qc-vision-system/main/assets/analysis_morphological.jpg" alt="Advanced analysis — morphological analysis by defect class" width="100%"/>

> *Morphological analysis — circularity and elongation distributions per defect class rendered as box plots with mean and standard deviation. Distinguishes rounded pits (high circularity) from elongated scratches and cracks (high elongation, low circularity).*

<br/>

<img src="https://raw.githubusercontent.com/Philmor-C/industrial-qc-vision-system/main/assets/analysis_severity.JPG" alt="Advanced analysis — severity scoring with pass/fail verdict" width="100%"/>

> *Severity scoring — each defect scored by type weight × log(area) × confidence. KPI cards show average severity, maximum severity, pass/fail ratio, and overall part verdict: Pass / Rework / Reject. Top 20 defects ranked by severity shown as a bar chart.*

<br/>

<img src="https://raw.githubusercontent.com/Philmor-C/industrial-qc-vision-system/main/assets/analysis_spatial_pattern.JPG" alt="Advanced analysis — spatial defect density heatmap" width="100%"/>

> *Spatial pattern analysis — Gaussian-smoothed defect density heatmap over normalised part coordinates. Nearest-neighbour ratio classifies the distribution as Clustered, Random, or Regular — directly useful for tracing tooling wear, fixture misalignment, or material flow issues.*

<br/>

<img src="https://raw.githubusercontent.com/Philmor-C/industrial-qc-vision-system/main/assets/analysis_trend.JPG" alt="Advanced analysis — inspection trend across sessions" width="100%"/>

> *Trend analysis — defect count and total defect area plotted across inspection sessions with linear regression. Slope-based labelling flags the process as Increasing, Stable, or Decreasing — an early warning system for process drift before it becomes a reject rate problem.*

---

## 🔍 Defect Detection Pipeline — Technical Detail

```
Input Image (any resolution)
        │
        ▼
compute_patch_grid()          ← tiles image into 1280×1280 patches
        │
        ▼  (per patch)
preprocess_patch()            ← resize → float32/255 → CHW → batch dim
        │
        ▼
ONNX Runtime                  ← OpenVINO provider, CPU fallback
        │
        ▼
postprocess()  [VECTORISED]
  ├─ sigmoid(output[:,4])     objectness
  ├─ sigmoid(output[:,5:])    class scores (dynamic — any class count)
  ├─ argmax per row           best class selection
  ├─ element-wise multiply    final confidence
  └─ boolean mask filter      threshold survivors
        │
        ▼
apply_nms()                   ← cv2.dnn.NMSBoxes, IoU=0.45
        │
        ▼
expand_bbox()                 ← 2× context expansion around each detection
        │
        ▼
process_roi_contours()
  ├─ Canny (low=0.6×median, high=1.3×median — adaptive)
  ├─ Dilate + MORPH_CLOSE (bridge edge gaps)
  ├─ MORPH_OPEN (remove noise specks)
  └─ Largest contour → area, perimeter, circularity, hull area
        │
        ▼
Output: annotated image + defects_metadata.json + ROI crops
```

---

## 📐 Measurement Pipeline — Technical Detail

```
Input Image + Operator-placed center points
        │
        ▼
BGR → Grayscale + GaussianBlur 3×3
        │
        ▼  (per center point — 250×200px ROI window)
process_roi_vectorised()
  ├─ Batch-slice all horizontal strips simultaneously (NumPy)
  ├─ scipy convolve2d: kernel [0.5, 0.5, -0.5, -0.5]
  ├─ argmax → left edge (light→dark transition)
  └─ argmax of negated gradient → right edge (dark→light)
        │
        ▼
fit_line_robust()
  ├─ cv2.fitLine(DIST_HUBER) — robust initial fit
  ├─ Single-pass perpendicular distance filter (threshold 3px)
  └─ Least-squares refit on inliers → (vx, vy, x0, y0)
        │
        ▼
measure_and_draw_distances_scaled()
  ├─ Scale all coords to display resolution (7%)
  ├─ |mean_x_right_i − mean_x_left_i+1| per consecutive pair
  └─ Compare against operator standards → OK / FAIL
```

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/Philmor-C/industrial-qc-vision-system.git
cd industrial-qc-vision-system

# 2. Install dependencies
pip install -r requirements.txt

# Optional: Intel OpenVINO acceleration
pip install onnxruntime-openvino

# 3. Place your trained ONNX model
#    models/best_4_1280.onnx

# 4. Add inspection images
#    samples/              ← defect detection images
#    measurement_samples/  ← dimensional measurement images

# 5. Run
python GUI.py
```

---

## 📁 Project Structure

```
industrial-qc-vision-system/
├── GUI.py                    # PySide2 Qt5 desktop application
├── defect_pipeline.py        # YOLO patch inference + Canny ROI analysis
├── measurement_pipeline.py   # Vectorised edge detection + Huber line fitting
├── advanced_analysis.py      # 8 Plotly-based analytics functions
├── config.py                 # Path registry and application constants
├── utils.py                  # Image encoding and HTML component helpers
├── requirements.txt
├── assets/                   # Application screenshots
├── models/
│   └── best_4_1280.onnx      ← trained model (not included — see note)
├── samples/                  ← defect inspection input images
├── measurement_samples/      ← dimensional measurement input images
└── detected_defects/         ← auto-created output directory
    ├── rois/
    ├── contours/
    ├── annotated_image.jpg
    └── defects_metadata.json
```

---

## 🔧 Technical Specifications

| Component | Specification |
|---|---|
| **Model format** | ONNX (exported from YOLOv5) |
| **Inference patch size** | 1280 × 1280 px |
| **Defect classes** | scratch, dent, crack, stain, missing\_part, deformation |
| **Inference providers** | OpenVINO (primary) → CPU (fallback) |
| **Confidence threshold** | 0.40 default (operator-adjustable 0.10–0.90) |
| **NMS IoU threshold** | 0.45 |
| **ROI expansion** | 2× bounding box context |
| **Edge detection** | Vectorised gradient scan, SAMPLING\_STEP=15px |
| **Line fitting** | Huber robust loss, inlier threshold 3px |
| **Display scale** | 7% of original resolution for measurement output |
| **Output formats** | JPEG (ROI crops, contour viz), JSON (metadata), CSV (defect report) |
| **UI framework** | PySide2 (Qt5), QWebEngineView for HTML rendering |
| **Chart engine** | Plotly 3.x — interactive zoom, hover, pan |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `PySide2` | Qt5 desktop framework + embedded browser |
| `onnxruntime` | ONNX model inference (CPU) |
| `onnxruntime-openvino` | Optional — Intel OpenVINO acceleration |
| `opencv-python` | Image processing, NMS, contour analysis, drawing |
| `numpy` | Vectorised array operations |
| `Pillow` | Image encoding for HTML embedding |
| `plotly` | Interactive analytics charts |
| `scipy` | Convolution, spatial statistics, gaussian filter |

---

## 👤 About the Author

**Filmon** — Mechanical Engineer specialising in industrial computer vision, automation engineering, and factory-floor quality control system development.

- Designed and built this system end-to-end: model training, ONNX inference engine, measurement algorithms, Qt5 desktop application, and analytics layer
- Background in YOLO-based defect detection, sub-pixel metrology, and large-scale industrial image processing
- Co-inventor of a patented bilingual defect detection system using YOLOv5, OpenCV, CUDA, and TensorRT — developed and deployed at a manufacturing technology company in Shenzhen, China
- B.Eng. Mechanical Design, Manufacturing and Automation — Zhejiang A&F University, China (2021)

📧 Open to industrial computer vision, machine vision engineering, and automation engineering roles globally — with full relocation.

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/filmon-24098a3ba)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=for-the-badge&logo=github)](https://github.com/Philmor-C)

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

*Built for production. Designed for offline factory environments. Zero cloud dependency.*

⭐ If this project is useful to you, consider starring the repository.

</div>
