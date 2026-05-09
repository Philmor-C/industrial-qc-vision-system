# MEASUREMENT PIPELINE
import cv2
import numpy as np
import time
from scipy.signal import convolve2d

# -----------------------------
# CONFIG
# -----------------------------
SAMPLING_STEP = 15
SCAN_PIXEL_NUM = 5
EDGE_THRESHOLD = 20
HALF_EDGE_WIDTH = 2

ROI_WIDTH = 250
ROI_HEIGHT = 200


# -----------------------------
# EDGE DETECTION
# -----------------------------
def process_roi_vectorised(gray, roi):
    """
    Extract left/right edge points from one ROI.
    """
    x, y, w, h = roi
    if w < 2 * HALF_EDGE_WIDTH:
        return [], []

    # All middle rows of the strips
    offsets = np.arange(0, h - SCAN_PIXEL_NUM + 1, SAMPLING_STEP)
    y_starts = y + offsets
    middle_rows = y_starts + HALF_EDGE_WIDTH  # SCAN_PIXEL_NUM//2

    # Gather intensity profiles – shape (num_strips, w)
    profiles = gray[middle_rows, x:x + w].astype(np.float32)

    # Convolution kernel for left-right gradient
    kernel = np.array([0.5, 0.5, -0.5, -0.5], dtype=np.float32)  # [left avg, -right avg]
    # 2D convolution along the profiles
    gradients = convolve2d(profiles, kernel[np.newaxis, :], mode='valid')  # (num_strips, w-3)

    # --- light→dark (left brighter) ---
    max_ld = np.max(gradients, axis=1)
    idx_ld = np.argmax(gradients, axis=1)
    valid_ld = max_ld >= EDGE_THRESHOLD
    x_ld = idx_ld + x + HALF_EDGE_WIDTH  # mapping: conv index 0 ⇔ column = HALF_EDGE_WIDTH
    y_ld = middle_rows
    edge_points_1 = [(float(x_ld[i]), float(y_ld[i])) for i in range(len(offsets)) if valid_ld[i]]

    # --- dark→light (right brighter) ---
    neg_grad = -gradients
    max_dl = np.max(neg_grad, axis=1)
    idx_dl = np.argmax(neg_grad, axis=1)
    valid_dl = max_dl >= EDGE_THRESHOLD
    x_dl = idx_dl + x + HALF_EDGE_WIDTH
    y_dl = middle_rows
    edge_points_2 = [(float(x_dl[i]), float(y_dl[i])) for i in range(len(offsets)) if valid_dl[i]]

    return edge_points_1, edge_points_2


# -----------------------------
# LINE FITTING
# -----------------------------
def fit_line_robust(points, threshold=3.0):
    """
    line fitting using OpenCV's DIST_HUBER + refit on inliers.
    Returns (inliers_list, (vx, vy, x0, y0)).
    """
    if not points or len(points) < 2:
        return [], None

    # Filter valid points
    pts = np.array([p for p in points if p and not np.isnan(p[0]) and not np.isnan(p[1])],
                   dtype=np.float32)
    if len(pts) < 2:
        return [], None

    # initial fit using Huber loss
    line = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 1e-2, 1e-2)
    vx, vy, x0, y0 = line.flatten()

    # Distance from each point to the line
    dists = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
    inlier_mask = dists < threshold
    inliers = pts[inlier_mask]

    if len(inliers) < 2:
        # Fallback: return all valid points as inliers
        valid_list = [(int(p[0]), int(p[1])) for p in pts]
        return valid_list, None

    # Refit using classic least squares on inliers
    if np.std(inliers[:, 0]) < 2.0:  # near-vertical
        vx_final, vy_final = 0.0, 1.0
        x0_final = np.mean(inliers[:, 0])
        y0_final = 0.0
    else:
        x = inliers[:, 0]
        y = inliers[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        theta = np.arctan2(1, m)
        vx_final = np.cos(theta)
        vy_final = np.sin(theta)
        x0_final = 0.0
        y0_final = c

    inlier_list = [(int(p[0]), int(p[1])) for p in inliers]
    return inlier_list, (float(vx_final), float(vy_final), float(x0_final), float(y0_final))


# -----------------------------
# DRAWING & HELPER FUNCTIONS
# -----------------------------
def draw_edge_points(image, edge_points, color, radius=5):
    """Draw edge points on the image"""
    for point in edge_points:
        if point:
            cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)


def fit_line_and_remove_outliers(points, threshold=5.0):
    """
    Fit edge points to a straight line using RANSAC approach and remove outliers
    (kept for possible backward compatibility, but not used in optimised pipeline)
    """
    if not points or len(points) < 2:
        return [], None
    valid_points = []
    for p in points:
        if p and len(p) >= 2 and not np.isnan(p[0]) and not np.isnan(p[1]):
            valid_points.append(p)
    if len(valid_points) < 2:
        return [], None
    points_array = np.array(valid_points, dtype=np.float32)
    if len(points_array) >= 3:
        max_trials = 100
        best_inliers = []
        best_line = None
        for _ in range(max_trials):
            idx = np.random.choice(len(points_array), 2, replace=False)
            p1, p2 = points_array[idx[0]], points_array[idx[1]]
            if p1[0] == p2[0]:
                vx, vy = 0, 1
                x0, y0 = p1[0], 0
            else:
                vx, vy = p2[0] - p1[0], p2[1] - p1[1]
                norm = np.sqrt(vx ** 2 + vy ** 2)
                if norm > 0:
                    vx, vy = vx / norm, vy / norm
                else:
                    vx, vy = 0, 1
                x0, y0 = p1[0], p1[1]
            distances = []
            for pt in points_array:
                dx = pt[0] - x0
                dy = pt[1] - y0
                dist = abs(dx * vy - dy * vx)
                distances.append(dist)
            inlier_idx = np.array(distances) < threshold
            inliers = points_array[inlier_idx]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_line = (vx, vy, x0, y0)
                if len(best_inliers) == len(points_array):
                    break
        if len(best_inliers) >= 2:
            x = best_inliers[:, 0]
            y = best_inliers[:, 1]
            if np.std(x) < 2.0:
                vx, vy = 0, 1
                x0 = np.mean(x)
                y0 = 0
            else:
                try:
                    A = np.vstack([x, np.ones(len(x))]).T
                    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                    theta = np.arctan2(1, m)
                    vx = np.cos(theta)
                    vy = np.sin(theta)
                    x0 = 0
                    y0 = c
                except:
                    vx, vy = 0, 1
                    x0 = np.mean(x)
                    y0 = 0
            inliers_list = [(int(pt[0]), int(pt[1])) for pt in best_inliers if
                            not np.isnan(pt[0]) and not np.isnan(pt[1])]
            return inliers_list, (vx, vy, x0, y0)
    valid_points_list = [(int(p[0]), int(p[1])) for p in valid_points]
    return valid_points_list, None


def draw_vertical_straight_line(image, inliers, line_params, color=(0, 255, 255), thickness=3):
    if not inliers or line_params is None:
        return
    vx, vy, x0, y0 = line_params
    if abs(vx) < 0.1:
        x_pos = int(np.mean([p[0] for p in inliers]))
        y_min = min([p[1] for p in inliers]) - 20
        y_max = max([p[1] for p in inliers]) + 20
        cv2.line(image, (x_pos, y_min), (x_pos, y_max), color, thickness)
    else:
        y_min = min([p[1] for p in inliers]) - 20
        y_max = max([p[1] for p in inliers]) + 20
        if abs(vy) > 0.001:
            slope = vx / vy
            x_at_y_min = np.mean([p[0] - slope * (p[1] - y_min) for p in inliers])
            x_at_y_max = np.mean([p[0] - slope * (p[1] - y_max) for p in inliers])
            cv2.line(image, (int(x_at_y_min), y_min), (int(x_at_y_max), y_max), color, thickness)


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def draw_horizontal_distances_with_centers(image, all_fitted_lines, centers, roi_height=200):
    distances = []
    for i in range(len(all_fitted_lines) - 1):
        e2_line_current = all_fitted_lines[i][1]
        e1_line_next = all_fitted_lines[i + 1][0]
        if e2_line_current is None or e1_line_next is None:
            continue
        vx1, vy1, x01, y01 = e2_line_current
        vx2, vy2, x02, y02 = e1_line_next
        x1 = get_line_x_at_y(e2_line_current, all_fitted_lines[i][2], centers[i][1])
        x2 = get_line_x_at_y(e1_line_next, all_fitted_lines[i + 1][2], centers[i + 1][1])
        if x1 is None or x2 is None:
            continue
        distance_val = abs(x2 - x1)
        distances.append(distance_val)
        y_mid = (centers[i][1] + centers[i + 1][1]) // 2
        cv2.line(image, (x1, y_mid), (x2, y_mid), (255, 0, 255), 10)
        text = f"{distance_val:.1f} px"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 10
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (x1 + x2) // 2 - text_w // 2
        text_y = y_mid - 15
        cv2.rectangle(image, (text_x - 5, text_y - text_h - 5), (text_x + text_w + 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 0, 255), thickness)
        cv2.drawMarker(image, (x1, y_mid), (0, 255, 255), cv2.MARKER_DIAMOND, 10, 2)
        cv2.drawMarker(image, (x2, y_mid), (0, 255, 255), cv2.MARKER_DIAMOND, 10, 2)
    return distances


def get_line_x_at_y(line_params, inlier_points, y_target):
    if line_params is None or not inlier_points:
        return None
    vx, vy, x0, y0 = line_params
    if abs(vy) < 0.001:
        return np.mean([p[0] for p in inlier_points])
    if abs(vx) < 0.001:
        return np.mean([p[0] for p in inlier_points])
    x = x0 + vx * (y_target - y0) / vy
    return int(x)


def resize_and_adjust_points(image, points_list, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    new_size = (width, height)
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    adjusted_points = []
    if points_list:
        scale_x = scale_percent / 100
        scale_y = scale_percent / 100
        for point in points_list:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                adjusted_x = int(point[0] * scale_x)
                adjusted_y = int(point[1] * scale_y)
                adjusted_points.append((adjusted_x, adjusted_y))
            else:
                adjusted_points.append(point)
    return resized_image, adjusted_points


def measure_and_draw_distances_scaled(image, all_fitted_lines, all_rois, centers, scale_percent=7):
    scale = scale_percent / 100
    scaled_rois = []
    for roi in all_rois:
        x, y, w, h = roi
        scaled_rois.append((int(x * scale), int(y * scale), int(w * scale), int(h * scale)))
    scaled_centers = []
    for cx, cy in centers:
        scaled_centers.append((int(cx * scale), int(cy * scale)))
    scaled_fitted_lines = []
    for e1_line, e2_line, e1_inliers, e2_inliers in all_fitted_lines:
        scaled_e1_inliers = []
        if e1_inliers:
            for p in e1_inliers:
                if p and len(p) >= 2:
                    scaled_e1_inliers.append((int(p[0] * scale), int(p[1] * scale)))
        scaled_e2_inliers = []
        if e2_inliers:
            for p in e2_inliers:
                if p and len(p) >= 2:
                    scaled_e2_inliers.append((int(p[0] * scale), int(p[1] * scale)))
        scaled_e1_line = None
        if e1_line:
            vx, vy, x0, y0 = e1_line
            scaled_e1_line = (vx, vy, x0 * scale, y0 * scale)
        scaled_e2_line = None
        if e2_line:
            vx, vy, x0, y0 = e2_line
            scaled_e2_line = (vx, vy, x0 * scale, y0 * scale)
        scaled_fitted_lines.append((scaled_e1_line, scaled_e2_line, scaled_e1_inliers, scaled_e2_inliers))
    distances = []
    for i, roi in enumerate(scaled_rois):
        x, y, w, h = roi
        e1_line, e2_line, e1_inliers, e2_inliers = scaled_fitted_lines[i]
        if e1_line and e1_inliers:
            draw_vertical_line_scaled(image, e1_line, e1_inliers, (0, 255, 0), 2)
        if e2_line and e2_inliers:
            draw_vertical_line_scaled(image, e2_line, e2_inliers, (255, 0, 0), 2)
    for i in range(len(scaled_fitted_lines) - 1):
        try:
            e1_line_current, e2_line_current, e1_inliers_current, e2_inliers_current = scaled_fitted_lines[i]
            e1_line_next, e2_line_next, e1_inliers_next, e2_inliers_next = scaled_fitted_lines[i + 1]
            x_right = None
            x_left = None
            if e2_inliers_current and len(e2_inliers_current) > 0:
                valid_x = [p[0] for p in e2_inliers_current if p and not np.isnan(p[0])]
                if valid_x:
                    x_right = int(np.mean(valid_x))
            if x_right is None and e1_inliers_current and len(e1_inliers_current) > 0:
                valid_x = [p[0] for p in e1_inliers_current if p and not np.isnan(p[0])]
                if valid_x:
                    x_right = max(valid_x)
            if e1_inliers_next and len(e1_inliers_next) > 0:
                valid_x = [p[0] for p in e1_inliers_next if p and not np.isnan(p[0])]
                if valid_x:
                    x_left = int(np.mean(valid_x))
            if x_left is None and e2_inliers_next and len(e2_inliers_next) > 0:
                valid_x = [p[0] for p in e2_inliers_next if p and not np.isnan(p[0])]
                if valid_x:
                    x_left = min(valid_x)
            if x_right is None or x_left is None:
                continue
            distance_val = abs(x_left - x_right)
            distances.append(distance_val)
            roi_current = scaled_rois[i]
            roi_next = scaled_rois[i + 1]
            if i % 2 == 0:
                y_top_current = roi_current[1]
                y_top_next = roi_next[1]
                y_draw = min(y_top_current, y_top_next) + 15
                position_label = "TOP"
                text_y_offset = 20
            else:
                y_bottom_current = roi_current[1] + roi_current[3]
                y_bottom_next = roi_next[1] + roi_next[3]
                y_draw = min(y_bottom_current, y_bottom_next) - 15
                position_label = "BOTTOM"
                text_y_offset = -20
            colors = [(0, 200, 255), (0, 255, 100)]
            line_color = colors[i % len(colors)]
            cv2.line(image, (x_right, y_draw), (x_left, y_draw), line_color, 2)
            cv2.circle(image, (x_right, y_draw), 5, (0, 255, 255), -1)
            cv2.circle(image, (x_left, y_draw), 5, (0, 255, 255), -1)
            cv2.drawMarker(image, (x_right, y_draw), line_color, cv2.MARKER_DIAMOND, 8, 2)
            cv2.drawMarker(image, (x_left, y_draw), line_color, cv2.MARKER_DIAMOND, 8, 2)
            center_x = (x_right + x_left) // 2
            center_y = y_draw + text_y_offset
            text = f"{distance_val:.1f} px"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            overlay = image.copy()
            cv2.rectangle(overlay,
                          (center_x - text_w // 2 - 3, center_y - text_h - 3),
                          (center_x + text_w // 2 + 3, center_y + 3),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            cv2.putText(image, text, (center_x - text_w // 2, center_y),
                        font, font_scale, line_color, thickness)
            arrow_offset = min(15, distance_val // 8)
            if arrow_offset < distance_val // 2:
                cv2.arrowedLine(image, (x_right + arrow_offset, y_draw),
                                (x_left - arrow_offset, y_draw), line_color, 1, tipLength=0.05)
            # print(f"📏 Distance {i + 1}-{i + 2}: {distance_val:.2f} pixels [{position_label}]")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            continue
    return image, distances


def draw_vertical_line_scaled(image, line_params, inlier_points, color, thickness):
    if not line_params or not inlier_points:
        return
    vx, vy, x0, y0 = line_params
    valid_x = [p[0] for p in inlier_points if p and not np.isnan(p[0])]
    if valid_x:
        x_pos = int(np.mean(valid_x))
    else:
        x_pos = int(x0)
    valid_y = [p[1] for p in inlier_points if p and not np.isnan(p[1])]
    if valid_y:
        y_min = min(valid_y) - 10
        y_max = max(valid_y) + 10
    else:
        y_min = 0
        y_max = image.shape[0]
    cv2.line(image, (x_pos, y_min - 50), (x_pos, y_max + 70), color, thickness)


def run_pipeline(image, centers, display_scale_percent=5):
    """
    Run the complete pipeline with scaling for display.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    all_fitted_lines = []
    all_rois = []

    # Sequential processing
    for cx, cy in centers:
        roi = (int(cx - ROI_WIDTH / 2), int(cy - ROI_HEIGHT / 2), ROI_WIDTH, ROI_HEIGHT)
        e1, e2 = process_roi_vectorised(gray, roi)
        e1_inliers, e1_line = fit_line_robust(e1, threshold=3.0)
        e2_inliers, e2_line = fit_line_robust(e2, threshold=3.0)
        all_rois.append(roi)
        all_fitted_lines.append((e1_line, e2_line, e1_inliers, e2_inliers))

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    scaled_vis, _ = resize_and_adjust_points(vis, [], display_scale_percent)
    final_image, distances = measure_and_draw_distances_scaled(
        scaled_vis, all_fitted_lines, all_rois, centers, display_scale_percent
    )
    return final_image, distances


# -----------------------------
# Entry point (if run standalone)
# -----------------------------
if __name__ == "__main__":
    img = cv2.imread("test.bmp")
    if img is None:
        print("❌ Image not found")
        exit()
    centers = [
        (2100, 3000),
        (3200, 3000),
        (3400, 3000),
        (8200, 3000),
        (9100, 3000),
        (13800, 3000),
        (14300, 3000),
    ]
    start = time.time()
    result, distances = run_pipeline(img, centers, display_scale_percent=7)
    end = time.time()
    print(f"\n⏱ Processing time: {end - start:.3f}s")
    cv2.imshow("Industrial Measurement", result)
    cv2.imwrite("output_scaled.png", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
