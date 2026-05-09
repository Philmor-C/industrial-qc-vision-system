# Integrated QC Vision System

Combines edge‑based measurement (distance between ROIs) with YOLO patch‑based defect detection.  
Measurement and product images can come from different folders.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Place your ONNX model in `models/best_4_1280.onnx`
3. Put measurement images (e.g., large BMP) in `measurement_samples/`
4. Put product inspection images in `samples/`
5. Adjust measurement centers (x,y coordinates) in the UI or edit `config.py`

## Run
`python app.py`

## Features
- Top panel: measurement distances + visualization
- Middle: side‑by‑side original and detection result (first product image)
- Bottom: table of defect ROIs with contour images and metadata
- Tabs for all results and aggregated analytics
- CSV export of defect data