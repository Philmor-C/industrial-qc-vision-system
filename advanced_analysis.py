# advanced_analysis
import os
import json
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance_matrix


def extract_defect_data(defect_results):
    defects = []
    if not defect_results:
        return defects
    for img_idx, img_res in enumerate(defect_results):
        if not isinstance(img_res, dict):
            continue
        img = img_res.get('original')
        if img is None:
            continue
        h, w = img.shape[:2]
        for d in img_res.get('defects', []):
            bbox = d.get('original_bbox', (0, 0, 0, 0))
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            contours = d.get('contour_details', [])
            circ = contours[0]['circularity'] if contours else 0
            perim = contours[0]['perimeter'] if contours else 0
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            elongation = max(width, height) / (min(width, height) + 1e-6)
            defects.append({
                'image_idx': img_idx,
                'filename': img_res.get('filename', ''),
                'class': d.get('class', 'unknown'),
                'confidence': d.get('confidence', 0),
                'area': d.get('total_contour_area', 0),
                'contour_count': d.get('contour_count', 0),
                'center_x': cx / w,
                'center_y': cy / h,
                'circularity': circ,
                'perimeter': perim,
                'elongation': elongation,
                'bbox': bbox
            })
    return defects


def html_wrapper(title, content):
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>
    <style>
        body {{ background-color: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; margin: 20px; }}
        h3 {{ color: #f59e0b; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background-color: #1e293b; color: #f59e0b; }}
        .kpi {{ background-color: #1e293b; border-radius: 8px; padding: 16px; text-align: center; margin: 10px; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    </style>
</head>
<body>
{content}
</body>
</html>"""


# ---------- 1. Pareto Analysis ----------
def pareto_analysis(defect_results):
    defects = extract_defect_data(defect_results)
    if not defects:
        return html_wrapper("Pareto Analysis", "<p>No defect data available.</p>")
    freq = Counter(d['class'] for d in defects)
    labels, counts = zip(*sorted(freq.items(), key=lambda x: x[1], reverse=True))
    cum_percent = np.cumsum(counts) / sum(counts) * 100
    fig_freq = go.Figure()
    fig_freq.add_trace(
        go.Bar(x=labels, y=counts, name='Frequency', marker_color='#f59e0b', text=counts, textposition='auto'))
    fig_freq.add_trace(
        go.Scatter(x=labels, y=cum_percent, name='Cumulative %', yaxis='y2', line=dict(color='#2dd4bf', dash='dash'),
                   mode='lines+markers'))
    fig_freq.update_layout(title='Pareto – Defect Frequency', xaxis_title='Defect Type', yaxis_title='Count',
                           yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 100]),
                           template='plotly_dark')
    area_sum = defaultdict(float)
    for d in defects:
        area_sum[d['class']] += d['area']
    labels_a, areas_a = zip(*sorted(area_sum.items(), key=lambda x: x[1], reverse=True))
    cum_area = np.cumsum(areas_a) / sum(areas_a) * 100
    fig_area = go.Figure()
    fig_area.add_trace(
        go.Bar(x=labels_a, y=areas_a, name='Total Area', marker_color='#ef4444', text=[f"{a:.0f}" for a in areas_a],
               textposition='auto'))
    fig_area.add_trace(
        go.Scatter(x=labels_a, y=cum_area, name='Cumulative %', yaxis='y2', line=dict(color='#10b981', dash='dash')))
    fig_area.update_layout(title='Pareto – Total Defect Area', xaxis_title='Defect Type', yaxis_title='Area (px²)',
                           yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 100]),
                           template='plotly_dark')
    content = fig_freq.to_html(full_html=False, include_plotlyjs='cdn') + fig_area.to_html(full_html=False,
                                                                                           include_plotlyjs='cdn')
    return html_wrapper("Pareto Analysis", content)


# ---------- 2. Morphological Analysis ----------
def morphological_analysis(defect_results):
    defects = extract_defect_data(defect_results)
    if not defects:
        return html_wrapper("Morphological Analysis", "<p>No defect data.</p>")
    classes = set(d['class'] for d in defects)
    stats = []
    for cls in classes:
        circ = [d['circularity'] for d in defects if d['class'] == cls]
        elong = [d['elongation'] for d in defects if d['class'] == cls]
        stats.append({
            'class': cls,
            'count': len(circ),
            'circ_mean': np.mean(circ) if circ else 0,
            'circ_std': np.std(circ) if circ else 0,
            'elong_mean': np.mean(elong) if elong else 0,
            'elong_std': np.std(elong) if elong else 0
        })
    table = "<table><thead><tr><th>Defect Type</th><th>Count</th><th>Circularity (mean±std)</th><th>Elongation (" \
            "mean±std)</th></tr></thead><tbody> "
    for s in stats:
        table += f"<tr><td>{s['class']}</td><td>{s['count']}</td><td>{s['circ_mean']:.2f} ± {s['circ_std']:.2f}</td><td>{s['elong_mean']:.2f} ± {s['elong_std']:.2f}</td></tr>"
    table += "</tbody></table>"
    fig_circ = go.Figure()
    for cls in classes:
        circ_vals = [d['circularity'] for d in defects if d['class'] == cls]
        fig_circ.add_trace(go.Box(y=circ_vals, name=cls, boxmean='sd', marker_color='#f59e0b'))
    fig_circ.update_layout(title='Circularity Distribution by Defect Type', yaxis_title='Circularity',
                           template='plotly_dark')
    fig_elong = go.Figure()
    for cls in classes:
        elong_vals = [d['elongation'] for d in defects if d['class'] == cls]
        fig_elong.add_trace(go.Box(y=elong_vals, name=cls, boxmean='sd', marker_color='#2dd4bf'))
    fig_elong.update_layout(title='Elongation Distribution by Defect Type', yaxis_title='Elongation',
                            template='plotly_dark')
    content = table + fig_circ.to_html(full_html=False, include_plotlyjs='cdn') + fig_elong.to_html(full_html=False,
                                                                                                    include_plotlyjs='cdn')
    return html_wrapper("Morphological Analysis", content)


# ---------- 3. Spatial Pattern Analysis ----------
def spatial_pattern_analysis(defect_results, image_shape):
    defects = extract_defect_data(defect_results)
    if not defects:
        return html_wrapper("Spatial Pattern Analysis", "<p>No defects to analyze.</p>")
    h, w = image_shape
    heatmap = np.zeros((100, 100), dtype=np.float32)
    for d in defects:
        x = int(d['center_x'] * 99)
        y = int(d['center_y'] * 99)
        if 0 <= x < 100 and 0 <= y < 100:
            heatmap[y, x] += 1
    heatmap = gaussian_filter(heatmap, sigma=3)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    fig_heat = go.Figure(data=go.Heatmap(z=heatmap, colorscale='hot', showscale=True))
    fig_heat.update_layout(title='Defect Density Heatmap', template='plotly_dark')
    centers = [(d['center_x'], d['center_y']) for d in defects]
    if len(centers) > 1:
        dists = distance_matrix(centers, centers)
        np.fill_diagonal(dists, np.inf)
        nearest = dists.min(axis=1)
        avg_nn = np.mean(nearest)
        area = 1.0
        density = len(centers) / area
        expected_nn = 1 / (2 * np.sqrt(density)) if density > 0 else 1e6
        ratio = avg_nn / expected_nn if expected_nn > 0 else 1
        pattern = "Clustered" if ratio < 0.8 else ("Regular" if ratio > 1.2 else "Random")
        summary = f"<div style='background:#1e293b; padding:12px; border-radius:8px; margin-bottom:16px;'><b>Nearest " \
                  f"Neighbor Ratio:</b> {ratio:.3f}<br><b>Spatial Pattern:</b> {pattern}</div> "
    else:
        summary = "<div style='background:#1e293b; padding:12px; border-radius:8px; margin-bottom:16px;'>Insufficient " \
                  "points for clustering analysis.</div> "
    content = summary + fig_heat.to_html(full_html=False, include_plotlyjs='cdn')
    return html_wrapper("Spatial Pattern Analysis", content)


# ---------- 4. Severity Analysis ----------
def severity_analysis(defect_results, threshold):
    defects = extract_defect_data(defect_results)
    if not defects:
        return html_wrapper("Severity Analysis", "<p>No defect data.</p>")
    type_weights = {"scratch": 3, "crack": 5, "dent": 4, "stain": 2, "missing_part": 10, "deformation": 6, "unknown": 3}
    for d in defects:
        w = type_weights.get(d['class'], 3)
        area_score = np.log1p(d['area']) / 10
        conf_score = d['confidence']
        d['severity'] = w * (area_score + conf_score / 2)
    scores = [d['severity'] for d in defects]
    avg_sev = np.mean(scores)
    max_sev = np.max(scores)
    passed = sum(1 for s in scores if s <= threshold)
    total = len(scores)
    status_color = "#10b981" if passed == total else "#f59e0b" if passed >= total * 0.7 else "#ef4444"
    kpi = f"""
    <div class="grid-2">
        <div class="kpi"><div>Avg Severity</div><div style="font-size:28px;">{avg_sev:.2f}</div></div>
        <div class="kpi"><div>Max Severity</div><div style="font-size:28px;">{max_sev:.2f}</div></div>
        <div class="kpi"><div>Passed (≤{threshold:.1f})</div><div style="font-size:28px; color:{status_color};">{passed}/{total}</div></div>
        <div class="kpi"><div>Overall Status</div><div style="font-size:28px; color:{status_color};">{'PASS' if passed == total else 'REWORK' if passed >= total * 0.7 else 'REJECT'}</div></div>
    </div>
    """
    sorted_defects = sorted(defects, key=lambda x: x['severity'], reverse=True)[:20]
    labels = [f"{d['class']}" for d in sorted_defects]
    sev_vals = [d['severity'] for d in sorted_defects]
    fig_bar = go.Figure(
        go.Bar(x=labels, y=sev_vals, marker_color='#f59e0b', text=[f"{v:.2f}" for v in sev_vals], textposition='auto'))
    fig_bar.update_layout(title=f'Top 20 Defects by Severity (threshold={threshold:.1f})', xaxis_title='Defect',
                          yaxis_title='Severity Score', template='plotly_dark')
    fig_box = go.Figure()
    for cls in set(d['class'] for d in defects):
        sev_cls = [d['severity'] for d in defects if d['class'] == cls]
        fig_box.add_trace(go.Box(y=sev_cls, name=cls, boxmean='sd'))
    fig_box.update_layout(title='Severity Distribution by Defect Type', yaxis_title='Severity', template='plotly_dark')
    fig_pie = go.Figure(data=[go.Pie(labels=['Pass', 'Fail'], values=[passed, total - passed], hole=0.4,
                                     marker_colors=['#10b981', '#ef4444'])])
    fig_pie.update_layout(title='Pass/Fail Ratio', template='plotly_dark')
    content = kpi + fig_bar.to_html(full_html=False, include_plotlyjs='cdn') + fig_box.to_html(full_html=False,
                                                                                               include_plotlyjs='cdn') + fig_pie.to_html(
        full_html=False, include_plotlyjs='cdn')
    return html_wrapper("Severity Analysis", content)


# ---------- 5. Trend Analysis ----------
HISTORY_FILE = "inspection_history.json"


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []


def save_history(entry):
    history = load_history()
    history.append(entry)
    if len(history) > 50:
        history = history[-50:]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def trend_analysis(defect_results):
    if not defect_results:
        return html_wrapper("Trend Analysis", "<p>No defects for trend analysis.</p>")
    defects = extract_defect_data(defect_results)
    total_defects = len(defects)
    total_area = sum(d['area'] for d in defects)
    timestamp = datetime.now().isoformat()
    entry = {"timestamp": timestamp, "total_defects": total_defects, "total_area": total_area}
    save_history(entry)
    history = load_history()
    if len(history) < 2:
        return html_wrapper("Trend Analysis", "<p>Need at least 2 inspections for trend chart.</p>")
    times = [h['timestamp'] for h in history]
    defects_series = [h['total_defects'] for h in history]
    area_series = [h['total_area'] for h in history]
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Defects per Inspection", "Total Defect Area per Inspection"))
    fig.add_trace(go.Scatter(x=times, y=defects_series, mode='lines+markers', name='Defects', marker_color='#f59e0b'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=area_series, mode='lines+markers', name='Area (px²)', marker_color='#ef4444'),
                  row=2, col=1)
    fig.update_layout(title="Inspection Trend Analysis", template="plotly_dark")
    if len(defects_series) > 1:
        x = np.arange(len(defects_series))
        slope = np.polyfit(x, defects_series, 1)[0]
        trend = "increasing" if slope > 0.1 else ("decreasing" if slope < -0.1 else "stable")
        summary = f"<div style='background:#1e293b; padding:12px; border-radius:8px; margin-bottom:16px;'>Defect " \
                  f"trend: <strong>{trend}</strong> (slope={slope:.2f} defects/inspection)</div> "
    else:
        summary = ""
    content = summary + fig.to_html(full_html=False, include_plotlyjs='cdn')
    return html_wrapper("Trend Analysis", content)


# ---------- 6. Defect Density Analysis ----------
def defect_density_analysis(defect_results, image_shape, pixel_to_cm2=0.001):
    defects = extract_defect_data(defect_results)
    if not defects:
        return html_wrapper("Defect Density Analysis", "<p>No defects to compute density.</p>")
    h, w = image_shape
    area_cm2 = h * w * pixel_to_cm2
    total_defects = len(defects)
    total_area_px = sum(d['area'] for d in defects)
    density_defects = total_defects / area_cm2 if area_cm2 > 0 else 0
    density_area = total_area_px * pixel_to_cm2 / area_cm2 if area_cm2 > 0 else 0
    # Bar chart per class
    class_counts = Counter(d['class'] for d in defects)
    fig_bar = go.Figure(go.Bar(x=list(class_counts.keys()), y=list(class_counts.values()), marker_color='#f59e0b'))
    fig_bar.update_layout(title='Defect Count per Type', xaxis_title='Defect Type', yaxis_title='Count',
                          template='plotly_dark')
    # Pie for area share
    area_by_class = defaultdict(float)
    for d in defects:
        area_by_class[d['class']] += d['area']
    fig_pie = go.Figure(go.Pie(labels=list(area_by_class.keys()), values=list(area_by_class.values()), hole=0.3))
    fig_pie.update_layout(title='Defect Area Share by Type', template='plotly_dark')
    kpi = f"""
    <div class="grid-2">
        <div class="kpi"><div>Defects per cm²</div><div style="font-size:32px;">{density_defects:.3f}</div></div>
        <div class="kpi"><div>Defect Area per cm² (mm²)</div><div style="font-size:32px;">{density_area * 100:.2f}</div></div>
    </div>
    """
    content = kpi + fig_bar.to_html(full_html=False, include_plotlyjs='cdn') + fig_pie.to_html(full_html=False,
                                                                                               include_plotlyjs='cdn')
    return html_wrapper("Defect Density Analysis", content)


# ---------- 7. Root Cause Correlation ----------
def root_cause_correlation(defect_results):
    defects = extract_defect_data(defect_results)
    if not defects:
        return html_wrapper("Root Cause Correlation", "<p>No defects to correlate.</p>")
    left = sum(1 for d in defects if d['center_x'] < 0.5)
    right = sum(1 for d in defects if d['center_x'] >= 0.5)
    top = sum(1 for d in defects if d['center_y'] < 0.5)
    bottom = sum(1 for d in defects if d['center_y'] >= 0.5)
    total = len(defects)
    fig_bar = go.Figure(
        go.Bar(x=['Left', 'Right', 'Top', 'Bottom'], y=[left, right, top, bottom], marker_color='#2dd4bf'))
    fig_bar.update_layout(title='Defect Distribution by Location', xaxis_title='Zone', yaxis_title='Count',
                          template='plotly_dark')
    fig_pie = go.Figure(
        go.Pie(labels=['Left', 'Right'], values=[left, right], hole=0.4, marker_colors=['#f59e0b', '#ef4444']))
    fig_pie.update_layout(title='Left/Right Distribution', template='plotly_dark')
    table = f"""
    <table>
        <tr><th>Zone</th><th>Defect Count</th><th>Percentage</th></tr>
        <tr><td>Left Half</td><td>{left}</td><td>{left / total * 100:.1f}%</td></tr>
        <tr><td>Right Half</td><td>{right}</td><td>{right / total * 100:.1f}%</td></tr>
        <tr><td>Top Half</td><td>{top}</td><td>{top / total * 100:.1f}%</td></tr>
        <tr><td>Bottom Half</td><td>{bottom}</td><td>{bottom / total * 100:.1f}%</td></tr>
    </table>
    """
    content = table + fig_bar.to_html(full_html=False, include_plotlyjs='cdn') + fig_pie.to_html(full_html=False,
                                                                                                 include_plotlyjs='cdn')
    return html_wrapper("Root Cause Correlation", content)


# ---------- 8. Box Plots ----------
def box_plots(defect_results):
    defects = extract_defect_data(defect_results)
    if not defects:
        return html_wrapper("Box Plots", "<p>No defect data.</p>")
    classes = set(d['class'] for d in defects)
    fig_size = go.Figure()
    for cls in classes:
        sizes = [d['area'] for d in defects if d['class'] == cls]
        fig_size.add_trace(go.Box(y=sizes, name=cls, boxmean='sd', marker_color='#f59e0b'))
    fig_size.update_layout(title='Defect Size Distribution by Type', yaxis_title='Area (px²)', template='plotly_dark')
    fig_circ = go.Figure()
    for cls in classes:
        circs = [d['circularity'] for d in defects if d['class'] == cls]
        fig_circ.add_trace(go.Box(y=circs, name=cls, boxmean='sd', marker_color='#2dd4bf'))
    fig_circ.update_layout(title='Defect Circularity Distribution by Type', yaxis_title='Circularity',
                           template='plotly_dark')
    content = fig_size.to_html(full_html=False, include_plotlyjs='cdn') + fig_circ.to_html(full_html=False,
                                                                                           include_plotlyjs='cdn')
    return html_wrapper("Box Plots", content)
