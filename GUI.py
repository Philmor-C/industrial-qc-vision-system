# Industrial QC System
import sys
import os
import tempfile
import csv
import glob
from datetime import datetime
from collections import Counter
import cv2
import numpy as np
from PySide2.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsItem,
    QGraphicsTextItem, QPushButton, QLabel, QTextEdit, QLineEdit,
    QSlider, QGroupBox, QGridLayout, QFileDialog,
    QMessageBox, QSplitter, QTabWidget, QButtonGroup,
    QDoubleSpinBox, QScrollArea, QProgressBar, QStatusBar, QDialog,
    QDialogButtonBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PySide2.QtCore import Qt, QRectF, QPointF, Signal, QSize, QUrl, QThread, QTimer
from PySide2.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QFont, QPainter
from PySide2.QtWebEngineWidgets import QWebEngineView

# Import analysis module
import advanced_analysis as aa

# Import pipelines
from measurement_pipeline import run_pipeline as measurement_pipeline
from defect_pipeline import YOLOPatchDetector
from config import *
from utils import numpy_to_base64, create_defect_badge, create_kpi_card

# ============================================================
# Global detector
# ============================================================
detector = None


def get_detector():
    global detector
    if detector is None:
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️  Warning: Model not found at {MODEL_PATH}")
            return None
        detector = YOLOPatchDetector(
            model_path=MODEL_PATH,
            patch_size=1280,
            stride_ratio=0.01,
            confidence_threshold=0.4,
            nms_threshold=0.45,
            class_names=["scratch", "dent", "crack", "stain", "missing_part", "deformation"],
            expansion_ratio=2.0
        )
    return detector


# ============================================================
# Core pipeline functions
# ============================================================
def process_measurement(measurement_dir, centers_text, display_scale=7):
    meas_images = []
    for ext in ["*.bmp", "*.jpg", "*.jpeg", "*.png"]:
        meas_images.extend(glob.glob(os.path.join(measurement_dir, ext)))
    if not meas_images:
        return None, [], "No measurement image found"
    meas_path = meas_images[0]
    centers = []
    for line in centers_text.strip().split('\n'):
        if ',' in line:
            x, y = line.strip().split(',')
            centers.append((int(x.strip()), int(y.strip())))
    if not centers:
        return None, [], "No centers provided"
    img = cv2.imread(meas_path)
    if img is None:
        return None, [], f"Failed to load {meas_path}"
    try:
        result_img, distances = measurement_pipeline(img, centers, display_scale)
        return result_img, distances, "Success"
    except Exception as e:
        return None, [], f"Measurement error: {str(e)}"


def process_defects(samples_dir, conf_threshold=0.4):
    detector = get_detector()
    if detector is None:
        return [], "Defect detector not available (model missing)"
    product_images = []
    for ext in ["*.bmp", "*.jpg", "*.jpeg", "*.png"]:
        product_images.extend(glob.glob(os.path.join(samples_dir, ext)))
    if not product_images:
        return [], "No product images found"
    results = []
    for img_path in sorted(product_images)[:20]:
        img = cv2.imread(img_path)
        if img is None:
            continue
        detections = detector.detect(img)
        defects = []
        for i, det in enumerate(detections):
            exp_x1, exp_y1, exp_x2, exp_y2 = detector.expand_bbox(
                det.x1, det.y1, det.x2, det.y2, img.shape[1], img.shape[0]
            )
            roi = img[exp_y1:exp_y2, exp_x1:exp_x2]
            if roi.size == 0:
                continue
            contour_data = detector.process_roi_contours(roi)
            defects.append({
                'id': i,
                'class': det.class_name,
                'confidence': det.confidence,
                'original_bbox': (det.x1, det.y1, det.x2, det.y2),
                'expanded_roi': (exp_x1, exp_y1, exp_x2, exp_y2),
                'roi_image': roi,
                'contour_viz': contour_data['contour_visualization'],
                'contour_count': contour_data['contour_count'],
                'total_contour_area': contour_data['total_contour_area'],
                'contour_details': contour_data['contours'],
                'area_coverage_percent': (contour_data['total_contour_area'] / (
                        roi.shape[0] * roi.shape[1])) * 100 if roi.size else 0
            })
        annotated = detector.draw_bbox_on_image(img, detections)
        results.append({
            'filename': os.path.basename(img_path),
            'original': img,
            'annotated': annotated,
            'defects': defects,
            'detection_count': len(detections)
        })
    return results, f"Processed {len(results)} images"


def generate_modal_html(defect, image_idx, defect_idx):
    roi_b64 = numpy_to_base64(defect['roi_image'], size=(600, 600))
    contour_b64 = numpy_to_base64(defect['contour_viz'], size=(600, 600))

    contour_rows = ""
    if defect.get('contour_details'):
        for idx, cnt in enumerate(defect['contour_details'][:10]):
            bbox = cnt['bounding_box']
            contour_rows += f"""
            <tr style="border-bottom:1px solid #334155;">
                <td style="padding:5px;">{idx + 1}</td>
                <td style="padding:5px; text-align:right;">{cnt['area']:.1f}</td>
                <td style="padding:5px; text-align:right;">{cnt['perimeter']:.1f}</td>
                <td style="padding:5px; text-align:right;">{cnt['circularity']:.2f}</td>
                <td style="padding:5px;">{bbox['x']},{bbox['y']} {bbox['width']}x{bbox['height']}</td>
            </tr>"""

    contour_table = f"""
    <table style="width:100%; border-collapse:collapse; margin-top:10px; color:#cbd5e1; background:#0f172a;">
        <thead><tr style="background:#1e293b;">
            <th style="padding:8px; text-align:left;">Contour</th>
            <th style="padding:8px; text-align:right;">Area (px²)</th>
            <th style="padding:8px; text-align:right;">Perimeter</th>
            <th style="padding:8px; text-align:right;">Circularity</th>
            <th style="padding:8px; text-align:left;">BBox</th>
        </tr></thead>
        <tbody>{contour_rows}</tbody>
    </table>
    """ if contour_rows else "<p>No contour details available</p>"

    modal_html = f"""
    <div id="modal-{image_idx}-{defect_idx}" class="qc-modal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.95); z-index:10000; overflow:auto; justify-content:center; align-items:center;">
        <div style="background:#121826; max-width:1200px; width:90%; margin:auto; border-radius:12px; border:1px solid #f59e0b; overflow:hidden; position:relative; box-shadow:0 0 15px rgba(245,158,11,0.3);">
            <div style="background:#1e293b; padding:12px 20px; display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #f59e0b;">
                <h3 style="margin:0; color:#f59e0b; font-family:'Segoe UI',monospace;">🔍 DEFECT INSPECTION: {defect['class']} (conf={defect['confidence']:.3f})</h3>
                <button onclick="document.getElementById('modal-{image_idx}-{defect_idx}').style.display='none'" style="background:#ef4444; border:none; color:white; width:34px; height:34px; border-radius:6px; font-size:20px; font-weight:bold; cursor:pointer;">✕</button>
            </div>
            <div style="padding:24px; color:#e2e8f0;">
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
                    <div><h4 style="color:#f59e0b;">📸 ROI Image</h4><img src="{roi_b64}" style="width:100%; border-radius:8px; border:1px solid #f59e0b;"></div>
                    <div><h4 style="color:#f59e0b;">🔬 Contour Visualization</h4><img src="{contour_b64}" style="width:100%; border-radius:8px; border:1px solid #2dd4bf;"></div>
                </div>
                <div style="margin-top:20px; background:#0f172a; border-radius:8px; padding:16px;">
                    <h4 style="color:#f59e0b;">📊 Defect Metadata</h4>
                    <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:12px;">
                        <div><span style="color:#94a3b8;">Defect Type:</span> <strong style="color:#e2e8f0;">{defect['class']}</strong></div>
                        <div><span style="color:#94a3b8;">Confidence:</span> <strong style="color:#e2e8f0;">{defect['confidence']:.4f}</strong></div>
                        <div><span style="color:#94a3b8;">Original BBox:</span> <strong style="color:#e2e8f0;">({defect['original_bbox'][0]}, {defect['original_bbox'][1]}) to ({defect['original_bbox'][2]}, {defect['original_bbox'][3]})</strong></div>
                        <div><span style="color:#94a3b8;">Expanded ROI:</span> <strong style="color:#e2e8f0;">({defect['expanded_roi'][0]}, {defect['expanded_roi'][1]}) to ({defect['expanded_roi'][2]}, {defect['expanded_roi'][3]})</strong></div>
                        <div><span style="color:#94a3b8;">Number of Contours:</span> <strong style="color:#e2e8f0;">{defect['contour_count']}</strong></div>
                        <div><span style="color:#94a3b8;">Total Defect Area:</span> <strong style="color:#e2e8f0;">{defect['total_contour_area']:.1f} px²</strong></div>
                        <div><span style="color:#94a3b8;">Area Coverage:</span> <strong style="color:#e2e8f0;">{defect['area_coverage_percent']:.1f}% of ROI</strong></div>
                    </div>
                    <h4 style="color:#f59e0b; margin-top:20px;">📐 Per‑Contour Details</h4>
                    {contour_table}
                </div>
            </div>
        </div>
    </div>
    """
    return modal_html


def build_defect_table(defects, image_idx):
    if not defects:
        return "<div style='padding:20px; text-align:center; color:#cbd5e1;'>No defects found in this image</div>"
    rows = ""
    modals = ""
    for d_idx, defect in enumerate(defects):
        roi_thumb = numpy_to_base64(defect['roi_image'], size=(100, 100))
        contour_thumb = numpy_to_base64(defect['contour_viz'], size=(100, 100))
        badge_html = create_defect_badge(defect['class'], defect['confidence'])
        rows += f"""
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:8px;"><img src="{roi_thumb}" style="width:80px; height:80px; object-fit:cover; border-radius:4px; border:1px solid #f59e0b;"></td>
            <td style="padding:8px;"><img src="{contour_thumb}" style="width:80px; height:80px; object-fit:cover; border-radius:4px; border:1px solid #2dd4bf;"></td>
            <td style="padding:8px;">{badge_html}</td>
            <td style="padding:8px; color:#e2e8f0;">{defect['confidence']:.3f}</td>
            <td style="padding:8px; color:#e2e8f0;">{defect['contour_count']}</td>
            <td style="padding:8px; color:#e2e8f0;">{defect['total_contour_area']:.0f}</td>
            <td style="padding:8px;"><button onclick="document.getElementById('modal-{image_idx}-{d_idx}').style.display='flex'" style="background:#f59e0b; border:none; color:#121826; padding:6px 12px; border-radius:4px; cursor:pointer; font-weight:bold;">🔍 Inspect</button></td>
        </tr>"""
        modals += generate_modal_html(defect, image_idx, d_idx)
    table_html = f"""
    <div style="overflow-x:auto; border-radius:8px; border:1px solid #334155; background:#0f172a;">
        <table style="width:100%; border-collapse:collapse;">
            <thead><tr style="background:#1e293b; color:#94a3b8;">
                <th style="padding:12px;">ROI</th><th style="padding:12px;">Contours</th><th style="padding:12px;">Defect</th><th style="padding:12px;">Confidence</th><th style="padding:12px;">#Contours</th><th style="padding:12px;">Area (px²)</th><th style="padding:12px;">Action</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    {modals}
    """
    return table_html


# ------------------------------------------------------------
# Defect only report generation
# ------------------------------------------------------------
def run_defect_only(samples_dir, conf_threshold):
    defect_results, defect_status = process_defects(samples_dir, conf_threshold)
    style_css = """
    <style>
    body { background-color: #121826; color: #e2e8f0; font-family: 'Segoe UI', 'Roboto', sans-serif; }
    .qc-modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.95); z-index: 10000; overflow: auto; justify-content: center; align-items: center; }
    details { background: #1e293b; border-radius: 8px; padding: 12px; margin-bottom: 16px; border-left: 4px solid #2dd4bf; }
    summary { font-weight: bold; list-style: none; display: flex; align-items: center; cursor: pointer; color: #2dd4bf; font-family: monospace; }
    summary::-webkit-details-marker { display: none; }
    summary::before { content: "▶"; margin-right: 8px; color: #2dd4bf; }
    details[open] summary::before { content: "▼"; }
    </style>
    """
    parts = [style_css, '<details open><summary style="font-size:1.2em;">🔍 DEFECT DETECTION RESULTS</summary><div>']
    csv_path = None
    if defect_results:
        total_defects = sum(r['detection_count'] for r in defect_results)
        parts.append('<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:16px;">')
        parts.append(create_kpi_card("Images Processed", len(defect_results), "", "blue"))
        parts.append(create_kpi_card("Total Defects", total_defects, "", "red"))
        parts.append(create_kpi_card("Defects per Image", f"{total_defects / len(defect_results):.2f}", "", "yellow"))
        parts.append('</div>')
        
        for idx, res in enumerate(defect_results):
            parts.append(
                f'<details><summary style="background:#1e293b; padding:10px; border-radius:8px; color:#e2e8f0;">📸 {res["filename"]} - {res["detection_count"]} defect(s)</summary><div style="padding:16px;">')
            orig_b64 = numpy_to_base64(res['original'], size=(400, 400))
            anno_b64 = numpy_to_base64(res['annotated'], size=(400, 400))
            parts.append('<div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">')
            parts.append(
                f'<div><div style="color:#94a3b8;">Original</div><img src="{orig_b64}" style="width:100%; border-radius:8px; border:1px solid #f59e0b;"></div>')
            parts.append(
                f'<div><div style="color:#94a3b8;">Annotated</div><img src="{anno_b64}" style="width:100%; border-radius:8px; border:1px solid #ef4444;"></div>')
            parts.append('</div>')
            parts.append(build_defect_table(res['defects'], idx))
            parts.append('</div></details>')
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8', newline='')
        writer = csv.writer(tmp)
        writer.writerow(
            ["Image", "DefectClass", "Confidence", "ContourCount", "TotalArea_px2", "BBox_X1", "BBox_Y1", "BBox_X2",
             "BBox_Y2"])
        for r in defect_results:
            for d in r['defects']:
                writer.writerow([r['filename'], d['class'], f"{d['confidence']:.4f}", d['contour_count'],
                                 f"{d['total_contour_area']:.1f}", d['original_bbox'][0], d['original_bbox'][1],
                                 d['original_bbox'][2], d['original_bbox'][3]])
        tmp.close()
        csv_path = tmp.name
    else:
        parts.append(
            f'<div style="background:#7f1a1a; border-radius:8px; padding:16px; color:#fca5a5;">⚠️ {defect_status}</div>')
    parts.append('</div></details>')
    return "".join(parts), csv_path, defect_results


# ------------------------------------------------------------
# Measurement with standards comparison
# ------------------------------------------------------------
def compare_measurements(distances, standards, global_tolerance):
    results = []
    for i, meas in enumerate(distances):
        if i < len(standards):
            std, tol = standards[i]
        else:
            std, tol = 0, global_tolerance
        diff = abs(meas - std)
        ok = diff <= tol
        status = "OK" if ok else "FAIL"
        color = "#10b981" if ok else "#ef4444"
        results.append({
            "measured": meas,
            "standard": std,
            "tolerance": tol,
            "diff": diff,
            "status": status,
            "color": color
        })
    return results


def create_measurement_html(distances, comparison, status_msg, annotated_img=None):
    style_css = """
    <style>
    body { background-color: #121826; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; }
    .ok { color: #10b981; font-weight: bold; }
    .fail { color: #ef4444; font-weight: bold; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { padding: 8px; text-align: center; border-bottom: 1px solid #334155; }
    th { background-color: #1e293b; color: #f59e0b; }
    .kpi { background: #0f172a; border-radius: 8px; padding: 16px; text-align: center; border: 1px solid #f59e0b; }
    </style>
    """
    if not distances:
        return f"{style_css}<div style='background:#7f1a1a; border-radius:8px; padding:16px;'>⚠️ {status_msg}</div>"
    table_rows = ""
    for i, cmp in enumerate(comparison):
        status_class = "ok" if cmp["status"] == "OK" else "fail"
        table_rows += f"""
        <tr>
            <td>{i + 1}</td>
            <td>{cmp['measured']:.1f}</td>
            <td>{cmp['standard']:.1f}</td>
            <td>±{cmp['tolerance']:.1f}</td>
            <td>{cmp['diff']:.1f}</td>
            <td class="{status_class}">{cmp['status']}</td>
        </tr>
        """
    html = f"""
    {style_css}
    <details open><summary style="font-size:1.2em;">📏 MEASUREMENT RESULTS</summary><div>
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:16px;">
        <div class="kpi">📊 Total: {len(distances)}</div>
        <div class="kpi">📈 Avg: {np.mean(distances):.1f} px</div>
        <div class="kpi">🔻 Min: {min(distances):.1f} px</div>
        <div class="kpi">🔺 Max: {max(distances):.1f} px</div>
    </div>
    <table>
        <thead><tr><th>Pair</th><th>Measured (px)</th><th>Standard (px)</th><th>Tolerance (±)</th><th>Diff</th><th>Status</th></tr></thead>
        <tbody>{table_rows}</tbody>
    </table>
    </div></details>
    """
    if annotated_img is not None:
        img_b64 = numpy_to_base64(annotated_img, size=(800, 600))
        html += f'<img src="{img_b64}" style="width:100%; border-radius:8px; border:1px solid #f59e0b; ' \
                f'margin-top:16px;"> '
    return html


def run_measurement_with_comparison(measurement_dir, centers_text, standards, global_tolerance, display_scale=7):
    meas_img, distances, meas_status = process_measurement(measurement_dir, centers_text, display_scale)
    if distances:
        comparison = compare_measurements(distances, standards, global_tolerance)
        html = create_measurement_html(distances, comparison, meas_status, meas_img)
    else:
        html = create_measurement_html([], [], meas_status, None)
    return html, None


def run_full_inspection_with_comparison(measurement_dir, samples_dir, centers_text, conf_threshold, standards,
                                        global_tolerance):
    meas_html, _ = run_measurement_with_comparison(measurement_dir, centers_text, standards, global_tolerance)
    defect_html, csv_path, defect_results = run_defect_only(samples_dir, conf_threshold)
    summary = f'<details><summary style="font-size:1.2em; font-weight:bold; color:#f59e0b;">📊 EXECUTIVE ' \
              f'SUMMARY</summary><div style="background:#0f172a; border-radius:8px; padding:20px;"><p>Report ' \
              f'generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p></div></details> '
    full_html = "<style>body{background:#121826;color:#e2e8f0;}</style>" + meas_html + defect_html + summary
    return full_html, csv_path, defect_results


# ============================================================
# Worker Thread for inspections
# ============================================================
class InspectionWorker(QThread):
    progress = Signal(str)
    finished = Signal(str, str)
    finished_with_defects = Signal(str, str, object)
    error = Signal(str)

    def __init__(self, mode, measurement_dir=None, samples_dir=None, centers_text=None,
                 conf_threshold=0.4, standards=None, global_tolerance=1.0):
        super().__init__()
        self.mode = mode
        self.measurement_dir = measurement_dir
        self.samples_dir = samples_dir
        self.centers_text = centers_text
        self.conf_threshold = conf_threshold
        self.standards = standards if standards else []
        self.global_tolerance = global_tolerance

    def run(self):
        try:
            if self.mode == "measurement":
                self.progress.emit("📏 Running measurement pipeline...")
                html, csv_path = run_measurement_with_comparison(
                    self.measurement_dir, self.centers_text, self.standards, self.global_tolerance
                )
                self.finished.emit(html, csv_path if csv_path else "")
            elif self.mode == "defect":
                self.progress.emit("🔍 Running defect detection...")
                html, csv_path, defect_results = run_defect_only(self.samples_dir, self.conf_threshold)
                self.finished_with_defects.emit(html, csv_path if csv_path else "", defect_results)
            else:  # full
                self.progress.emit("⚙️ Running full inspection...")
                html, csv_path, defect_results = run_full_inspection_with_comparison(
                    self.measurement_dir, self.samples_dir, self.centers_text,
                    self.conf_threshold, self.standards, self.global_tolerance
                )
                self.finished_with_defects.emit(html, csv_path if csv_path else "", defect_results)
        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# Standards Dialog
# ============================================================
class MeasurementStandardsDialog(QDialog):
    def __init__(self, num_pairs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Measurement Standards")
        self.resize(600, 400)
        self.num_pairs = num_pairs
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QDialog { background-color: #121826; color: #e2e8f0; }
            QLabel { color: #e2e8f0; font-weight: bold; }
            QTableWidget { background-color: #0f172a; color: #cbd5e1; border: 1px solid #f59e0b; gridline-color: #334155; }
            QTableWidget::item { background-color: #0f172a; color: #cbd5e1; }
            QHeaderView::section { background-color: #1e293b; color: #f59e0b; padding: 6px; }
            QDoubleSpinBox { background-color: #1e293b; color: white; border: 1px solid #f59e0b; border-radius: 4px; }
            QPushButton { background-color: #1e293b; color: #e2e8f0; border: 1px solid #f59e0b; border-radius: 6px; padding: 6px 14px; }
            QPushButton:hover { background-color: #2d3a4e; }
            QPushButton:pressed { background-color: #f59e0b; color: #0f172a; }
        """)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(self.num_pairs, 2)
        self.table.setHorizontalHeaderLabels(["Standard Distance (px)", "Tolerance (± px)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        default_distances = [76.0, 18.0, 334.0, 63.0, 332.0, 30.0]
        for i in range(self.num_pairs):
            dist = default_distances[i] if i < len(default_distances) else 0.0
            self.table.setItem(i, 0, QTableWidgetItem(f"{dist:.1f}"))
            self.table.setItem(i, 1, QTableWidgetItem("1.0"))
        layout.addWidget(QLabel("Enter expected distance and tolerance for each consecutive pair:"))
        layout.addWidget(self.table)
        form = QFormLayout()
        self.global_spin = QDoubleSpinBox()
        self.global_spin.setRange(0, 100)
        self.global_spin.setValue(1.0)
        self.global_spin.setSuffix(" px")
        form.addRow("Global error margin (if not set per pair):", self.global_spin)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_standards(self):
        standards = []
        for i in range(self.num_pairs):
            dist_item = self.table.item(i, 0)
            tol_item = self.table.item(i, 1)
            dist = float(dist_item.text()) if dist_item and dist_item.text() else 0.0
            tol = float(tol_item.text()) if tol_item and tol_item.text() else self.global_spin.value()
            standards.append((dist, tol))
        return standards, self.global_spin.value()


# ============================================================
# ROI Editor
# ============================================================
class MovablePoint(QGraphicsEllipseItem):
    def __init__(self, x, y, radius=7, number=0, parent=None):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius, parent)
        self.setPos(x, y)
        self.setBrush(QBrush(QColor(245, 158, 11)))
        self.setPen(QPen(Qt.white, 2))
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setZValue(10)
        self.number = number
        self.text = QGraphicsTextItem(str(number + 1), self)
        self.text.setDefaultTextColor(Qt.white)
        self.text.setFont(QFont("Arial", 10, QFont.Bold))
        self.text.setPos(-4, -10)
        self.text.setZValue(11)


class ROIScene(QGraphicsScene):
    pointsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap_item = None
        self.display_size = QSize(1, 1)
        self.original_size = (1, 1)
        self.scale_factor = 1.0
        self.point_items = []
        self.line_items = []
        self.current_mode = "point_add"
        self.drag_item = None
        self.drag_offset = None

    def load_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return False
        self.original_size = (img.shape[1], img.shape[0])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        max_width = 1200
        if w > max_width:
            scale = max_width / w
            self.display_size = QSize(max_width, int(h * scale))
        else:
            scale = 1.0
            self.display_size = QSize(w, h)
        self.scale_factor = 1.0 / scale

        pixmap = QPixmap.fromImage(qimg)
        if self.display_size != QSize(w, h):
            pixmap = pixmap.scaled(self.display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.addItem(self.pixmap_item)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.point_items = []
        self.line_items = []
        return True

    def to_display(self, orig_x, orig_y):
        return QPointF(orig_x / self.scale_factor, orig_y / self.scale_factor)

    def to_original(self, scene_pt):
        x = int(scene_pt.x() * self.scale_factor)
        y = int(scene_pt.y() * self.scale_factor)
        return (x, y)

    def get_points_original(self):
        return [self.to_original(item.scenePos()) for item in self.point_items]

    def _update_lines(self):
        for line in self.line_items:
            self.removeItem(line)
        self.line_items.clear()
        pts = self.get_points_original()
        for i in range(len(pts) - 1):
            p1 = self.to_display(pts[i][0], pts[i][1])
            p2 = self.to_display(pts[i + 1][0], pts[i + 1][1])
            line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
            pen = QPen(QColor(0, 255, 255), 2, Qt.DashLine)
            line.setPen(pen)
            self.addItem(line)
            self.line_items.append(line)

    def mousePressEvent(self, event):
        if not self.pixmap_item:
            super().mousePressEvent(event)
            return
        scene_pos = event.scenePos()
        if self.current_mode == "point_add":
            self._add_point(scene_pos)
            self.pointsChanged.emit()
        elif self.current_mode == "point_move":
            item = self._point_at(scene_pos)
            if item:
                self.drag_item = item
                self.drag_offset = scene_pos - item.scenePos()
                event.accept()
                return
        elif self.current_mode == "point_delete":
            item = self._point_at(scene_pos)
            if item:
                self.point_items.remove(item)
                self.removeItem(item)
                self.removeItem(item.text)
                self.pointsChanged.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.pixmap_item:
            super().mouseMoveEvent(event)
            return
        scene_pos = event.scenePos()
        if self.current_mode == "point_move" and self.drag_item:
            new_pos = scene_pos - self.drag_offset
            new_pos = self._clamp_to_scene(new_pos)
            self.drag_item.setPos(new_pos)
            self.pointsChanged.emit()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_item:
            self.drag_item = None
            self.drag_offset = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _add_point(self, scene_pos):
        clamped = self._clamp_to_scene(scene_pos)
        number = len(self.point_items)
        point = MovablePoint(clamped.x(), clamped.y(), number=number)
        self.addItem(point)
        self.addItem(point.text)
        self.point_items.append(point)
        self._update_lines()

    def _point_at(self, scene_pos, radius=15):
        for item in self.point_items:
            if (item.scenePos() - scene_pos).manhattanLength() <= radius:
                return item
        return None

    def _clamp_to_scene(self, pos):
        r = self.sceneRect()
        x = max(r.left(), min(r.right(), pos.x()))
        y = max(r.top(), min(r.bottom(), pos.y()))
        return QPointF(x, y)

    def clear_points(self):
        for item in self.point_items:
            self.removeItem(item.text)
            self.removeItem(item)
        self.point_items.clear()
        self._update_lines()
        self.pointsChanged.emit()

    def set_mode(self, mode):
        self.current_mode = mode

    def apply_common_y(self, y_orig):
        for item in self.point_items:
            orig_x = self.to_original(item.scenePos())[0]
            new_display = self.to_display(orig_x, y_orig)
            item.setPos(new_display)
        self._update_lines()
        self.pointsChanged.emit()


class ROIEditorWidget(QWidget):
    pointsConfirmed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = ROIScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setMouseTracking(True)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setAlignment(Qt.AlignCenter)
        self.points_confirmed = False

        scroll = QScrollArea()
        scroll.setWidget(self.view)
        scroll.setWidgetResizable(True)

        button_style = """
            QPushButton {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #f59e0b;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: bold;
                font-family: 'Segoe UI';
            }
            QPushButton:hover {
                background-color: #2d3a4e;
                border-color: #fbbf24;
            }
            QPushButton:pressed {
                background-color: #f59e0b;
                color: #0f172a;
            }
            QPushButton:checked {
                background-color: #f59e0b;
                color: #0f172a;
                border-color: #f59e0b;
            }
        """
        self.btn_load_image = QPushButton("📁 Load Image")
        self.btn_add_pt = QPushButton("➕ Add Point")
        self.btn_move_pt = QPushButton("✋ Move Point")
        self.btn_del_pt = QPushButton("🗑 Delete Point")
        self.btn_clear_pts = QPushButton("🧹 Clear Points")
        self.btn_load_default_pts = QPushButton("📌 Default Points")
        self.btn_save_points = QPushButton("💾 Confirm Points")
        self.btn_set_standards = QPushButton("📏 Set Standards & Error")

        for btn in [self.btn_load_image, self.btn_add_pt, self.btn_move_pt, self.btn_del_pt,
                    self.btn_clear_pts, self.btn_load_default_pts, self.btn_save_points,
                    self.btn_set_standards]:
            btn.setStyleSheet(button_style)
        self.btn_save_points.setStyleSheet(button_style + "background-color: #0f766e; border-color: #2dd4bf;")
        self.btn_set_standards.setStyleSheet(button_style + "background-color: #0f766e; border-color: #2dd4bf;")

        self.mode_group = QButtonGroup(self)
        self.btn_add_pt.setCheckable(True)
        self.btn_move_pt.setCheckable(True)
        self.btn_del_pt.setCheckable(True)
        self.mode_group.addButton(self.btn_add_pt, 0)
        self.mode_group.addButton(self.btn_move_pt, 1)
        self.mode_group.addButton(self.btn_del_pt, 2)
        self.btn_add_pt.setChecked(True)
        self.current_mode = "point_add"

        self.btn_load_image.clicked.connect(self._load_image_dialog)
        self.btn_add_pt.clicked.connect(lambda: self._set_mode("point_add"))
        self.btn_move_pt.clicked.connect(lambda: self._set_mode("point_move"))
        self.btn_del_pt.clicked.connect(lambda: self._set_mode("point_delete"))
        self.btn_clear_pts.clicked.connect(self.scene.clear_points)
        self.btn_load_default_pts.clicked.connect(self._load_default_points)
        self.btn_save_points.clicked.connect(self._confirm_points)
        self.btn_set_standards.clicked.connect(self._open_standards_dialog)

        self.avg_y_label = QLabel("📊 Avg Y: --")
        self.avg_y_label.setStyleSheet("color: #f59e0b; font-weight: bold;")
        self.custom_y_input = QDoubleSpinBox()
        self.custom_y_input.setRange(0, 100000)
        self.custom_y_input.setDecimals(0)
        self.custom_y_input.setStyleSheet(
            "background-color: #1e293b; color: white; border: 1px solid #f59e0b; border-radius: 4px;")
        self.btn_avg_y = QPushButton("📐 Avg Y")
        self.btn_set_y = QPushButton("✅ Apply Y")
        self.btn_avg_y.setStyleSheet(button_style)
        self.btn_set_y.setStyleSheet(button_style)
        self.btn_avg_y.clicked.connect(self._set_avg_y)
        self.btn_set_y.clicked.connect(lambda: self._set_custom_y(self.custom_y_input.value()))

        self.points_text = QTextEdit()
        self.points_text.setReadOnly(True)
        self.points_text.setStyleSheet(
            "background-color: #0f172a; color: #cbd5e1; border: 1px solid #f59e0b; border-radius: 6px; font-family: "
            "monospace;")

        toolbar = QHBoxLayout()
        for btn in [self.btn_load_image, self.btn_add_pt, self.btn_move_pt, self.btn_del_pt,
                    self.btn_clear_pts, self.btn_load_default_pts, self.btn_save_points,
                    self.btn_set_standards]:
            toolbar.addWidget(btn)
        toolbar.addStretch()

        info_panel = QVBoxLayout()
        instructions = QLabel(
            "📌 INSTRUCTIONS:\n"
            "1️⃣ Load an image (Measurement folder).\n"
            "2️⃣ Select mode: Add / Move / Delete.\n"
            "3️⃣ Add at least 2 points (same Y coordinate).\n"
            "4️⃣ Use 'Avg Y' or enter custom Y.\n"
            "5️⃣ Click 'Confirm Points' to validate.\n"
            "6️⃣ Click 'Set Standards' to define expected distances & tolerance.\n"
            "7️⃣ Switch to Inspection Report tab and run tasks."
        )
        instructions.setStyleSheet(
            "background-color: #0f172a; padding: 12px; border-radius: 6px; color: #cbd5e1; border-left: 4px solid "
            "#f59e0b; font-family: 'Segoe UI';")
        instructions.setWordWrap(True)
        info_panel.addWidget(instructions)
        info_panel.addWidget(QLabel("🎯 Common Y (original coords):"))
        y_controls = QHBoxLayout()
        y_controls.addWidget(self.btn_avg_y)
        y_controls.addWidget(self.custom_y_input)
        y_controls.addWidget(self.btn_set_y)
        info_panel.addLayout(y_controls)
        info_panel.addWidget(self.avg_y_label)
        info_panel.addWidget(QLabel("📍 Points (original x,y):"))
        info_panel.addWidget(self.points_text)
        info_panel.addStretch()

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(toolbar)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        info_widget = QWidget()
        info_widget.setLayout(info_panel)
        splitter.addWidget(info_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.scene.pointsChanged.connect(self._update_points_display)
        self._update_points_display()
        self.setStyleSheet("background-color: #121826;")

        self.standards = []
        self.global_tolerance = 1.0

    def _set_mode(self, mode):
        self.scene.set_mode(mode)

    def load_image(self, path):
        success = self.scene.load_image(path)
        if not success:
            QMessageBox.warning(self, "Error", f"Could not load image: {path}")
        self._update_points_display()

    def _load_image_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Measurement Image", "",
                                                   "Images (*.bmp *.jpg *.jpeg *.png)")
        if file_path:
            self.load_image(file_path)

    def _load_default_points(self):
        default_pts = [
            (2144, 4176),
            (3222, 4176),
            (3482, 4176),
            (8247, 4176),
            (9120, 4176),
            (13831, 4176),
            (14322, 4176)
        ]
        self.scene.clear_points()
        for x, y in default_pts:
            display_pt = self.scene.to_display(x, y)
            self.scene._add_point(display_pt)
        self._update_points_display()
        QMessageBox.information(self, "Default Points",
                                f"Loaded {len(default_pts)} default points (Y = {default_pts[0][1]}).")

    def _confirm_points(self):
        pts = self.scene.get_points_original()
        if len(pts) < 2:
            QMessageBox.warning(self, "Insufficient Points", "Please add at least 2 points.")
            return
        y_vals = [p[1] for p in pts]
        if len(set(y_vals)) != 1:
            QMessageBox.warning(self, "Invalid Y Coordinate",
                                "All points must have the same Y coordinate (horizontal line).")
            return
        self.points_confirmed = True
        self.pointsConfirmed.emit(pts)
        QMessageBox.information(self, "Points Confirmed", f"Saved {len(pts)} points. You may now run inspections.")

    def _update_points_display(self):
        pts = self.scene.get_points_original()
        text = "\n".join(f"{x},{y}" for x, y in pts)
        self.points_text.setPlainText(text)
        if pts:
            avg_y = sum(p[1] for p in pts) / len(pts)
            self.avg_y_label.setText(f"📊 Avg Y: {avg_y:.1f}")
            self.custom_y_input.setValue(avg_y)
        else:
            self.avg_y_label.setText("📊 Avg Y: --")
        self.points_confirmed = False

    def _set_avg_y(self):
        pts = self.scene.get_points_original()
        if not pts:
            return
        avg_y = sum(p[1] for p in pts) / len(pts)
        self._set_common_y(int(avg_y))

    def _set_custom_y(self, value):
        self._set_common_y(int(value))

    def _set_common_y(self, y):
        self.scene.apply_common_y(y)
        self._update_points_display()

    def _open_standards_dialog(self):
        pts = self.scene.get_points_original()
        num_pairs = len(pts) - 1
        if num_pairs < 1:
            QMessageBox.warning(self, "No Points", "Please add at least 2 points first.")
            return
        dialog = MeasurementStandardsDialog(num_pairs, self)
        if dialog.exec_() == QDialog.Accepted:
            self.standards, self.global_tolerance = dialog.get_standards()
            msg = f"Standards saved for {len(self.standards)} pairs.\nGlobal tolerance: ±{self.global_tolerance} px."
            QMessageBox.information(self, "Standards Saved", msg)
            if self.parent() and hasattr(self.parent(), 'status_label'):
                self.parent().status_label.setText(
                    f"📏 Standards: {len(self.standards)} pairs, tol ±{self.global_tolerance}")

    def get_standards(self):
        return self.standards, self.global_tolerance

    def get_points_string(self):
        return self.points_text.toPlainText().strip()

    def get_points_list(self):
        return self.scene.get_points_original()


# ============================================================
# Main Window
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏭 INDUSTRIAL QC SYSTEM")

        self.worker = None
        self.preload_worker = None
        self.defect_cache = None
        self.defect_cache_folder = None
        self.delay_timer = None
        self.last_defect_results = None  # for analysis tab

        self.measurement_dir_edit = QLineEdit("measurement_samples")
        self.samples_dir_edit = QLineEdit("samples")
        self.browse_meas_btn = QPushButton("📂 Browse...")
        self.browse_samples_btn = QPushButton("📂 Browse...")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_label = QLabel("0.40")

        self.roi_editor = ROIEditorWidget(self)
        self.roi_editor.pointsConfirmed.connect(self._on_points_confirmed)

        self.web_view = QWebEngineView()
        self.web_view.setMinimumSize(1000, 600)
        self._csv_path = None

        self._init_ui()
        self._apply_style()

        QTimer.singleShot(500, self._preload_defect_silently)

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #121826; }
            QLabel { color: #e2e8f0; font-family: 'Segoe UI'; font-weight: 500; }
            QLineEdit, QTextEdit, QDoubleSpinBox {
                background-color: #0f172a; color: #e2e8f0; border: 1px solid #f59e0b;
                border-radius: 6px; padding: 6px; font-family: monospace;
            }
            QSlider::groove:horizontal {
                border: 1px solid #f59e0b; height: 6px; background: #0f172a; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #f59e0b; width: 14px; margin: -4px 0; border-radius: 7px;
            }
            QPushButton {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #f59e0b;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: bold;
                font-family: 'Segoe UI';
            }
            QPushButton:hover {
                background-color: #2d3a4e;
                border-color: #fbbf24;
            }
            QPushButton:pressed {
                background-color: #f59e0b;
                color: #0f172a;
            }
            QPushButton:checked {
                background-color: #f59e0b;
                color: #0f172a;
                border-color: #f59e0b;
            }
            QGroupBox {
                color: #f59e0b; border: 1px solid #f59e0b; border-radius: 8px;
                margin-top: 16px; font-weight: bold; font-family: monospace;
                background-color: rgba(15,23,42,0.5);
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 8px; color: #f59e0b; }
            QTabWidget::pane { background-color: #121826; border: 1px solid #f59e0b; border-radius: 8px; }
            QTabBar::tab {
                background-color: #0f172a; color: #cbd5e1; padding: 10px 24px; margin-right: 4px;
                border-top-left-radius: 8px; border-top-right-radius: 8px;
                font-family: 'Segoe UI', monospace; font-size: 14px; font-weight: bold;
                min-width: 150px;
            }
            QTabBar::tab:selected {
                background-color: #1e293b; border-bottom: 2px solid #f59e0b; color: #f59e0b;
            }
            QStatusBar { background-color: #0f172a; color: #e2e8f0; border-top: 1px solid #f59e0b; }
            QProgressBar { border: 1px solid #f59e0b; border-radius: 4px; text-align: center; color: white; background-color: #0f172a; }
            QProgressBar::chunk { background-color: #f59e0b; width: 10px; }
        """)

    def _init_ui(self):
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # Setup tab
        setup_tab = QWidget()
        tabs.addTab(setup_tab, "⚙️ SETUP & ROI")
        layout = QVBoxLayout(setup_tab)

        folder_group = QGroupBox("📂 FOLDERS")
        f_layout = QGridLayout()
        f_layout.addWidget(QLabel("📁 Measurement Folder:"), 0, 0)
        f_layout.addWidget(self.measurement_dir_edit, 0, 1)
        f_layout.addWidget(self.browse_meas_btn, 0, 2)
        f_layout.addWidget(QLabel("📁 Samples Folder:"), 1, 0)
        f_layout.addWidget(self.samples_dir_edit, 1, 1)
        f_layout.addWidget(self.browse_samples_btn, 1, 2)
        folder_group.setLayout(f_layout)
        layout.addWidget(folder_group)

        conf_group = QGroupBox("🎚️ DEFECT THRESHOLD")
        c_layout = QHBoxLayout()
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(40)
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v / 100:.2f}"))
        self.conf_slider.valueChanged.connect(self._on_threshold_changed)
        c_layout.addWidget(self.conf_slider)
        c_layout.addWidget(self.conf_label)
        conf_group.setLayout(c_layout)
        layout.addWidget(conf_group)

        layout.addWidget(self.roi_editor, stretch=1)

        self.browse_meas_btn.clicked.connect(lambda: self._browse_folder(self.measurement_dir_edit))
        self.browse_samples_btn.clicked.connect(lambda: self._browse_folder(self.samples_dir_edit))
        self.samples_dir_edit.textChanged.connect(self._on_samples_folder_changed)

        # Report tab
        report_tab = QWidget()
        tabs.addTab(report_tab, "  📊 INSPECTION ")
        report_layout = QVBoxLayout(report_tab)
        report_layout.setContentsMargins(0, 0, 0, 0)

        btn_panel = QHBoxLayout()
        self.run_meas_btn = QPushButton("📏 MEASUREMENT ONLY")
        self.run_defect_btn = QPushButton("🔍 DEFECT ONLY")
        self.run_both_btn = QPushButton("▶ FULL INSPECTION")
        self.save_csv_btn = QPushButton("💾 SAVE CSV")

        self.run_meas_btn.setCheckable(True)
        self.run_defect_btn.setCheckable(True)
        self.run_both_btn.setCheckable(True)
        self.inspection_group = QButtonGroup(self)
        self.inspection_group.addButton(self.run_meas_btn)
        self.inspection_group.addButton(self.run_defect_btn)
        self.inspection_group.addButton(self.run_both_btn)
        self.run_defect_btn.setChecked(True)

        for btn in [self.run_meas_btn, self.run_defect_btn, self.run_both_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #0f766e;
                    border-color: #2dd4bf;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #0e5e5a;
                }
                QPushButton:pressed {
                    background-color: #2dd4bf;
                    color: #0f172a;
                }
                QPushButton:checked {
                    background-color: #f59e0b;
                    color: #0f172a;
                    border-color: #f59e0b;
                }
            """)

        self.save_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #0f766e;
                border-color: #2dd4bf;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0e5e5a;
            }
            QPushButton:pressed {
                background-color: #2dd4bf;
                color: #0f172a;
            }
        """)

        btn_panel.addWidget(self.run_meas_btn)
        btn_panel.addWidget(self.run_defect_btn)
        btn_panel.addWidget(self.run_both_btn)
        btn_panel.addWidget(self.save_csv_btn)
        btn_panel.addStretch()
        report_layout.addLayout(btn_panel)

        self.web_view.setHtml(
            "<html><body style='background-color:#0f172a; color:#f59e0b; text-align:center; padding:40px; "
            "font-size:18px;'>🔍 Run an inspection to see results here.</body></html>")
        report_layout.addWidget(self.web_view, stretch=1)

        self.run_meas_btn.clicked.connect(lambda: self._start_worker("measurement"))
        self.run_defect_btn.clicked.connect(self._on_defect_button)
        self.run_both_btn.clicked.connect(lambda: self._start_worker("full"))
        self.save_csv_btn.clicked.connect(self._save_csv)

        self.run_meas_btn.setEnabled(False)
        self.run_both_btn.setEnabled(False)
        self.run_defect_btn.setEnabled(True)

        # ---------- Advanced Analysis Tab ----------
        analysis_tab = QWidget()
        tabs.addTab(analysis_tab, "  📈 ANALYSIS ")
        analysis_layout = QVBoxLayout(analysis_tab)

        # Collapsible severity threshold group (optional)
        self.severity_group = QGroupBox("⚙️ Severity Threshold")
        self.severity_group.setCheckable(True)
        self.severity_group.setChecked(False)
        self.severity_group.setStyleSheet("""
            QGroupBox::indicator { width: 18px; height: 18px; }
            QGroupBox::indicator:checked { image: none; background-color: #2dd4bf; }
            QGroupBox::indicator:unchecked { image: none; background-color: #f59e0b; }
        """)
        self.severity_content = QWidget()
        thres_layout = QHBoxLayout(self.severity_content)
        thres_layout.addWidget(QLabel("Threshold:"))
        self.severity_slider = QSlider(Qt.Horizontal)
        self.severity_slider.setRange(0, 100)
        self.severity_slider.setValue(0.10)
        self.severity_slider.setTickInterval(10)
        self.severity_slider.valueChanged.connect(self._on_severity_threshold_changed)
        self.severity_label = QLabel("0.10")
        thres_layout.addWidget(self.severity_slider)
        thres_layout.addWidget(self.severity_label)
        thres_layout.addStretch()
        group_layout = QVBoxLayout(self.severity_group)
        group_layout.addWidget(self.severity_content)
        self.severity_group.toggled.connect(self.severity_content.setVisible)
        analysis_layout.addWidget(self.severity_group)

        btn_layout = QHBoxLayout()
        self.analysis_button_group = QButtonGroup(self)
        self.btn_pareto = QPushButton("📊 Pareto Analysis")
        self.btn_morph = QPushButton("🔬 Morphological")
        self.btn_spatial = QPushButton("🗺️ Spatial Pattern")
        self.btn_severity = QPushButton("⚠️ Severity")
        self.btn_trend = QPushButton("📉 Trend")
        self.btn_density = QPushButton("📏 Defect Density")
        self.btn_correlation = QPushButton("🔗 Root Cause")
        self.btn_box = QPushButton("📦 Box Plots")
        for btn in [self.btn_pareto, self.btn_morph, self.btn_spatial, self.btn_severity,
                    self.btn_trend, self.btn_density, self.btn_correlation, self.btn_box]:
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #0f766e;
                    border-color: #2dd4bf;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #0e5e5a;
                }
                QPushButton:pressed {
                    background-color: #2dd4bf;
                    color: #0f172a;
                }
                QPushButton:checked {
                    background-color: #f59e0b;
                    color: #0f172a;
                    border-color: #f59e0b;
                }
            """)
            self.analysis_button_group.addButton(btn)
            btn_layout.addWidget(btn)
        btn_layout.addStretch()
        analysis_layout.addLayout(btn_layout)

        self.analysis_scroll = QScrollArea()
        self.analysis_scroll.setWidgetResizable(True)
        self.analysis_scroll.setStyleSheet("QScrollArea { background-color: #0f172a; border: none; }")
        self.analysis_view = QWebEngineView()
        self.analysis_view.setHtml(
            "<html><body style='background-color:#0f172a; color:#f59e0b; text-align:center; padding:40px; "
            "font-size:18px;'>📊 Select an analysis to see results.</body></html>")
        self.analysis_scroll.setWidget(self.analysis_view)
        analysis_layout.addWidget(self.analysis_scroll, stretch=1)

        self.btn_pareto.clicked.connect(self._run_pareto)
        self.btn_morph.clicked.connect(self._run_morphological)
        self.btn_spatial.clicked.connect(self._run_spatial)
        self.btn_severity.clicked.connect(self._run_severity)
        self.btn_trend.clicked.connect(self._run_trend)
        self.btn_density.clicked.connect(self._run_density)
        self.btn_correlation.clicked.connect(self._run_correlation)
        self.btn_box.clicked.connect(self._run_box_plots)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("🟢 Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addWidget(self.status_label, 1)
        self.status_bar.addPermanentWidget(self.progress_bar)

    # ---------- Helper methods ----------
    def _browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            line_edit.setText(folder)

    def _on_points_confirmed(self, pts):
        self.status_label.setText(f"✅ ROI confirmed: {len(pts)} points, Y = {pts[0][1]}")
        self.run_meas_btn.setEnabled(True)
        self.run_both_btn.setEnabled(True)

    def _show_progress(self, msg):
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat(msg)
        self.status_label.setText(msg)

    def _hide_progress(self):
        self.progress_bar.setVisible(False)

    def _display_results(self, html, csv_path):
        self._hide_progress()
        self._csv_path = csv_path if csv_path else None
        self.web_view.setHtml(html, QUrl())
        self.status_label.setText("✅ Inspection completed.")
        for btn in [self.run_meas_btn, self.run_defect_btn, self.run_both_btn]:
            btn.setEnabled(True)
        self.centralWidget().setCurrentIndex(1)

    def _preload_defect_silently(self):
        samples_dir = self.samples_dir_edit.text()
        if not os.path.isdir(samples_dir):
            return
        if self.preload_worker and self.preload_worker.isRunning():
            return
        self.preload_worker = InspectionWorker("defect", samples_dir=samples_dir,
                                               conf_threshold=self.conf_slider.value() / 100.0)
        self.preload_worker.finished_with_defects.connect(self._on_preload_finished)
        self.preload_worker.start()

    def _on_preload_finished(self, html, csv_path, defect_results):
        self.defect_cache = (html, csv_path)
        self.defect_cache_folder = self.samples_dir_edit.text()
        self.last_defect_results = defect_results

    def _on_samples_folder_changed(self):
        self.defect_cache = None
        self.defect_cache_folder = None
        self.last_defect_results = None
        self._preload_defect_silently()

    def _on_threshold_changed(self):
        self.defect_cache = None
        self.defect_cache_folder = None
        self.last_defect_results = None
        QTimer.singleShot(500, self._preload_defect_silently())

    def _on_defect_button(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Another inspection is already running.")
            return
        samples_dir = self.samples_dir_edit.text()
        if not os.path.isdir(samples_dir):
            QMessageBox.warning(self, "Error", "Samples folder not found.")
            return

        if self.defect_cache is not None and self.defect_cache_folder == samples_dir:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("")
            self.status_label.setText("Loading defect results...")
            self.run_defect_btn.setEnabled(False)
            self.delay_timer = QTimer()
            self.delay_timer.setSingleShot(True)
            self.delay_timer.timeout.connect(self._display_cached_defect_results)
            self.delay_timer.start(1000)
            return
        self._start_worker("defect")

    def _display_cached_defect_results(self):
        self._hide_progress()
        if self.defect_cache is None:
            self._start_worker("defect")
            return
        html, csv_path = self.defect_cache
        self._display_results(html, csv_path)
        self.run_defect_btn.setEnabled(True)

    # ---------- Start workers for inspections ----------
    def _start_worker(self, mode):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Another inspection is already running.")
            return
        if mode in ("measurement", "full") and not self.roi_editor.points_confirmed:
            QMessageBox.warning(self, "ROI not set", "Please confirm measurement points first.")
            return

        meas_dir = self.measurement_dir_edit.text()
        samples_dir = self.samples_dir_edit.text()
        centers_text = self.roi_editor.get_points_string()
        conf = self.conf_slider.value() / 100.0
        standards, global_tol = self.roi_editor.get_standards()

        self.status_label.setText(f"Starting {mode} with {len(standards)} standards, tolerance ±{global_tol}")

        if mode in ("measurement", "full") and (not os.path.isdir(meas_dir) or not centers_text):
            QMessageBox.warning(self, "Error", "Measurement folder or points missing.")
            return
        if mode in ("defect", "full") and not os.path.isdir(samples_dir):
            QMessageBox.warning(self, "Error", "Samples folder not found.")
            return

        self.worker = InspectionWorker(mode, meas_dir, samples_dir, centers_text, conf, standards, global_tol)
        if mode == "measurement":
            self.worker.finished.connect(self._on_inspection_finished)
        else:
            self.worker.finished_with_defects.connect(self._on_inspection_finished_with_defects)
        self.worker.error.connect(self._on_inspection_error)
        self._show_progress(f"🚀 Starting {mode} inspection...")
        for btn in [self.run_meas_btn, self.run_defect_btn, self.run_both_btn]:
            btn.setEnabled(False)
        self.worker.start()

    def _on_inspection_finished(self, html, csv_path):
        self._display_results(html, csv_path)

    def _on_inspection_finished_with_defects(self, html, csv_path, defect_results):
        self.last_defect_results = defect_results
        self._display_results(html, csv_path)

    def _on_inspection_error(self, error_msg):
        self._hide_progress()
        QMessageBox.critical(self, "Inspection Error", error_msg)
        self.status_label.setText("❌ Inspection failed.")
        for btn in [self.run_meas_btn, self.run_defect_btn, self.run_both_btn]:
            btn.setEnabled(True)

    def _save_csv(self):
        if not self._csv_path or not os.path.exists(self._csv_path):
            QMessageBox.information(self, "CSV", "No CSV report available. Run a defect detection first.")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Save CSV Report", "defect_report.csv", "CSV Files (*.csv)")
        if save_path:
            import shutil
            shutil.copy2(self._csv_path, save_path)
            self.status_label.setText(f"💾 CSV saved to {save_path}")

    # ---------- Advanced Analysis Methods ----------
    def _ensure_defect_results(self):
        """Return defect results if available and contain defects; otherwise show error and return None."""
        if self.last_defect_results is not None:
            if isinstance(self.last_defect_results, list) and len(self.last_defect_results) > 0:
                total_defects = sum(len(r.get('defects', [])) for r in self.last_defect_results)
                if total_defects == 0:
                    self.analysis_view.setHtml(
                        "<html><body style='background-color:#0f172a; color:#f59e0b; text-align:center; "
                        "padding:40px;'>⚠️ No defects found in the current inspection.</body></html>")
                    return None
                return self.last_defect_results
            else:
                self.analysis_view.setHtml(
                    "<html><body style='background-color:#0f172a; color:#f59e0b; text-align:center; padding:40px;'>⚠️ "
                    "No defect data available.</body></html>")
                return None

        samples_dir = self.samples_dir_edit.text()
        if not os.path.isdir(samples_dir):
            self.analysis_view.setHtml(
                "<html><body style='background-color:#0f172a; color:#f59e0b; text-align:center; padding:40px;'>⚠️ "
                "Samples folder not set. Please set a valid folder and run a defect inspection first.</body></html>")
            return None
        self.status_label.setText("Running defect detection for analysis...")
        QApplication.processEvents()
        self.last_defect_results, status_msg = process_defects(samples_dir, self.conf_slider.value() / 100.0)
        self.status_label.setText("Defect detection completed.")
        if not self.last_defect_results or len(self.last_defect_results) == 0 or sum(
                len(r.get('defects', [])) for r in self.last_defect_results) == 0:
            self.analysis_view.setHtml(
                f"<html><body style='background-color:#0f172a; color:#f59e0b; text-align:center; padding:40px;'>⚠️ No "
                f"defects found. {status_msg}</body></html>")
            return None
        return self.last_defect_results

    def _run_pareto(self):
        results = self._ensure_defect_results()
        if results is None:
            return
        html = aa.pareto_analysis(results)
        self.analysis_view.setHtml(html)

    def _run_morphological(self):
        results = self._ensure_defect_results()
        if results is None:
            return
        html = aa.morphological_analysis(results)
        self.analysis_view.setHtml(html)

    def _run_spatial(self):
        results = self._ensure_defect_results()
        if results is None:
            return
        if results[0].get('original') is None:
            self.analysis_view.setHtml(
                "<html><body style='background-color:#0f172a; color:#f59e0b; text-align:center; padding:40px;'>No "
                "image data found for spatial analysis.</body></html>")
            return
        first_img = results[0]['original']
        shape = first_img.shape[:2]
        html = aa.spatial_pattern_analysis(results, shape)
        self.analysis_view.setHtml(html)

    def _run_severity(self):
        self.severity_group.setChecked(True)  # expand the group
        results = self._ensure_defect_results()
        if results is None:
            return
        threshold = self.severity_slider.value() / 10.0
        html = aa.severity_analysis(results, threshold)
        self.analysis_view.setHtml(html)

    def _run_trend(self):
        results = self._ensure_defect_results()
        if results is None:
            return
        html = aa.trend_analysis(results)
        self.analysis_view.setHtml(html)

    def _run_density(self):
        results = self._ensure_defect_results()
        if results is None:
            return
        if results[0].get('original') is None:
            self.analysis_view.setHtml(
                "<html><body style='background-color:#0f172a; color:#f59e0b; text-align:center; padding:40px;'>No "
                "image for density calculation.</body></html>")
            return
        first_img = results[0]['original']
        shape = first_img.shape[:2]
        html = aa.defect_density_analysis(results, shape, pixel_to_cm2=0.001)
        self.analysis_view.setHtml(html)

    def _run_correlation(self):
        results = self._ensure_defect_results()
        if results is None:
            return
        html = aa.root_cause_correlation(results)
        self.analysis_view.setHtml(html)

    def _run_box_plots(self):
        results = self._ensure_defect_results()
        if results is None:
            return
        html = aa.box_plots(results)
        self.analysis_view.setHtml(html)

    def _on_severity_threshold_changed(self, val):
        self.severity_label.setText(f"{val / 10:.1f}")

    def closeEvent(self, event):
        if self.delay_timer and self.delay_timer.isActive():
            self.delay_timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1350, 900)
    window.show()
    sys.exit(app.exec_())
