# DEFECT DETECTION PIPELINE
import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import json
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Detection:
    """Detection result class"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


class YOLOPatchDetector:
    def __init__(
            self,
            model_path: str,
            patch_size: int = 1280,
            stride_ratio: float = 0.01,
            confidence_threshold: float = 0.4,
            nms_threshold: float = 0.45,
            class_names: List[str] = None,
            expansion_ratio: float = 4.0
    ):
        self.patch_size = patch_size
        self.stride = int(patch_size * (1 - stride_ratio))
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.expansion_ratio = expansion_ratio

        # 📌 OpenVINO provider if available, else fallback CPU
        providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        if class_names:
            self.class_names = class_names
        else:
            self.class_names = [f"class_{i}" for i in range(80)]

    def preprocess_patch(self, patch: np.ndarray):
        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))

        input_blob = patch.astype(np.float32) / 255.0
        input_blob = np.transpose(input_blob, (2, 0, 1))
        input_blob = np.expand_dims(input_blob, axis=0)

        return input_blob

    def postprocess(self, outputs, patch_offset: Tuple[int, int]) -> List[Detection]:
        x_start, y_start = patch_offset

        # Handle both list/array and possible batch dimension
        if isinstance(outputs, (list, tuple)):
            output = outputs[0]
        else:
            output = outputs
        if output.ndim == 3:
            output = output[0]  # shape (predictions, features)

        # Vectorised sigmoid + class selection
        raw_obj_conf = output[:, 4]
        obj_conf = 1.0 / (1.0 + np.exp(-raw_obj_conf))

        raw_cls = output[:, 5:]  # dynamic number of classes
        cls_scores = 1.0 / (1.0 + np.exp(-raw_cls))
        class_ids = np.argmax(cls_scores, axis=1)
        class_confs = cls_scores[np.arange(len(cls_scores)), class_ids]

        final_conf = obj_conf * class_confs

        # Thresholding
        mask = final_conf >= self.confidence_threshold
        if not np.any(mask):
            return []

        valid = output[mask]
        conf = final_conf[mask]
        cids = class_ids[mask]

        # Decode bounding boxes: xc,yc,w,h -> x1,y1,x2,y2
        cx, cy, bw, bh = valid[:, 0], valid[:, 1], valid[:, 2], valid[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        detections = []
        for i in range(len(valid)):
            if x2[i] > x1[i] and y2[i] > y1[i] and x1[i] >= 0 and y1[i] >= 0:
                final_x1 = x_start + int(np.round(x1[i]))
                final_y1 = y_start + int(np.round(y1[i]))
                final_x2 = x_start + int(np.round(x2[i]))
                final_y2 = y_start + int(np.round(y2[i]))

                detections.append(Detection(
                    x1=final_x1,
                    y1=final_y1,
                    x2=final_x2,
                    y2=final_y2,
                    confidence=float(conf[i]),
                    class_id=int(cids[i]),
                    class_name=self.class_names[cids[i]] if cids[i] < len(self.class_names) else f"class_{cids[i]}"
                ))

        return detections

    def compute_patch_grid(self, image_height: int, image_width: int) -> List[Tuple[int, int]]:
        patches = []
        rows = max(1, (image_height - self.patch_size + self.stride - 1) // self.stride + 1)
        cols = max(1, (image_width - self.patch_size + self.stride - 1) // self.stride + 1)

        for row_idx in range(rows):
            for col_idx in range(cols):
                row_start = row_idx * self.stride
                col_start = col_idx * self.stride

                if row_start + self.patch_size > image_height:
                    row_start = image_height - self.patch_size
                if col_start + self.patch_size > image_width:
                    col_start = image_width - self.patch_size

                patches.append((row_start, col_start))

        return patches

    def apply_nms(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []

        boxes = []
        scores = []

        for det in detections:
            boxes.append([det.x1, det.y1, det.x2, det.y2])
            scores.append(det.confidence)

        boxes = np.array(boxes)
        scores = np.array(scores)

        keep_indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold
        )

        if len(keep_indices) == 0:
            return []

        if isinstance(keep_indices, tuple):
            keep_indices = keep_indices[0]

        return [detections[i] for i in keep_indices.flatten()]

    def expand_bbox(self, x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int) -> Tuple[
        int, int, int, int]:
        width = x2 - x1
        height = y2 - y1

        expand_w = int(width * (self.expansion_ratio - 1) / 2)
        expand_h = int(height * (self.expansion_ratio - 1) / 2)

        new_x1 = max(0, x1 - expand_w)
        new_y1 = max(0, y1 - expand_h)
        new_x2 = min(img_width, x2 + expand_w)
        new_y2 = min(img_height, y2 + expand_h)

        return new_x1, new_y1, new_x2, new_y2

    def process_roi_contours(self, roi: np.ndarray) -> Dict:
        """
        Process ROI using Canny + morphological connection,
        then return only the largest contour (if any) with its metrics.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 1. Smooth
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Canny with median-based thresholds
        v = np.median(blur)
        edges = cv2.Canny(blur, int(0.6 * v), int(1.3 * v))

        # 3. Morphological connection (bridge gaps & solidify)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        connected = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

        # cleanup
        connected = cv2.morphologyEx(
            connected,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1
        )

        # Find all external contours
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter small noise
        min_area = 1
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Prepare outputs
        contour_data = []
        total_area = 0.0
        contour_image = roi.copy()

        if valid_contours:
            # Keep only the largest contour
            largest = max(valid_contours, key=cv2.contourArea)

            # Compute metrics
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            x, y, w, h = cv2.boundingRect(largest)
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0

            contour_data.append({
                "area": float(area),
                "perimeter": float(perimeter),
                "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "circularity": float(circularity),
                "hull_area": float(hull_area)
            })

            total_area = area

            # Draw only the largest contour (green, thickness 2)
            cv2.drawContours(contour_image, [largest], -1, (0, 255, 0), 2)

        return {
            "contour_count": len(valid_contours),  # will be 1 if largest exists, else 0
            "total_contour_area": float(total_area),
            "contours": contour_data,
            "contour_visualization": contour_image
        }

    def draw_bbox_on_image(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        vis_image = image.copy()
        np.random.seed(42)
        colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(self.class_names))}
        for det in detections:
            color = colors.get(det.class_id, (0, 255, 0))
            color = tuple(int(c) for c in color)
            cv2.rectangle(vis_image, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            label = f"{det.class_name}: {det.confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (det.x1, det.y1 - label_h - baseline), (det.x1 + label_w, det.y1), color, -1)
            cv2.putText(vis_image, label, (det.x1, det.y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1)
        cv2.putText(vis_image, f"Total Detections: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return vis_image

    def save_defects(self, image: np.ndarray, detections: List[Detection], output_dir: str = "detected_defects"):
        if not detections:
            return
        os.makedirs(output_dir, exist_ok=True)
        rois_dir = os.path.join(output_dir, "rois")
        contours_dir = os.path.join(output_dir, "contours")
        os.makedirs(rois_dir, exist_ok=True)
        os.makedirs(contours_dir, exist_ok=True)
        img_height, img_width = image.shape[:2]
        metadata_list = []
        for idx, det in enumerate(detections):
            orig_width = int(det.x2 - det.x1)
            orig_height = int(det.y2 - det.y1)
            orig_area = orig_width * orig_height
            exp_x1, exp_y1, exp_x2, exp_y2 = self.expand_bbox(det.x1, det.y1, det.x2, det.y2, img_width, img_height)
            exp_width = int(exp_x2 - exp_x1)
            exp_height = int(exp_y2 - exp_y1)
            exp_area = exp_width * exp_height
            roi = image[exp_y1:exp_y2, exp_x1:exp_x2]
            if roi.size == 0:
                continue
            contour_results = self.process_roi_contours(roi)
            roi_filename = f"defect_{idx:04d}_{det.class_name}_roi.jpg"
            roi_path = os.path.join(rois_dir, roi_filename)
            cv2.imwrite(roi_path, roi)
            contour_filename = f"defect_{idx:04d}_{det.class_name}_contours.jpg"
            contour_path = os.path.join(contours_dir, contour_filename)
            cv2.imwrite(contour_path, contour_results["contour_visualization"])
            metadata = {
                "defect_id": int(idx),
                "class_name": str(det.class_name),
                "class_id": int(det.class_id),
                "confidence": float(det.confidence),
                "original_bbox": {
                    "x1": int(det.x1), "y1": int(det.y1), "x2": int(det.x2), "y2": int(det.y2),
                    "width": orig_width, "height": orig_height, "area": orig_area
                },
                "expanded_roi": {
                    "x1": int(exp_x1), "y1": int(exp_y1), "x2": int(exp_x2), "y2": int(exp_y2),
                    "width": exp_width, "height": exp_height, "area": exp_area
                },
                "contour_analysis": {
                    "total_contours_found": contour_results["contour_count"],
                    "total_defect_area": contour_results["total_contour_area"],
                    "area_coverage_percentage": float(
                        (contour_results["total_contour_area"] / exp_area) * 100) if exp_area > 0 else 0,
                    "individual_contours": contour_results["contours"]
                },
                "image_files": {
                    "roi_image": roi_filename,
                    "contour_visualization": contour_filename
                },
                "expansion_ratio_used": float(self.expansion_ratio)
            }
            metadata_list.append(metadata)
        metadata_path = os.path.join(output_dir, "defects_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        vis_image = self.draw_bbox_on_image(image, detections)
        vis_path = os.path.join(output_dir, "annotated_image.jpg")
        cv2.imwrite(vis_path, vis_image)
        return metadata_list

    def detect(self, image: np.ndarray) -> List[Detection]:
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        h, w = image.shape[:2]
        patches = self.compute_patch_grid(h, w)
        all_detections = []

        for row_start, col_start in patches:
            patch = image[row_start:row_start + self.patch_size, col_start:col_start + self.patch_size]
            input_blob = self.preprocess_patch(patch)
            outputs = self.session.run([self.output_name], {self.input_name: input_blob})
            detections = self.postprocess(outputs, (col_start, row_start))
            all_detections.extend(detections)

        final_detections = self.apply_nms(all_detections)
        return final_detections
