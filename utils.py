import cv2
import base64
import io
import numpy as np
from PIL import Image


def numpy_to_base64(img_array, size=(200, 150)):
    """Convert numpy image to base64 string for HTML display"""
    if img_array is None:
        return ""

    # Convert BGR to RGB for PIL
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_array

    pil_img = Image.fromarray(img_rgb)
    pil_img.thumbnail(size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{img_str}"


def create_defect_badge(defect_type, confidence=None):
    """Create colored badge for defect type"""
    colors = {
        "scratch": "#EF4444",
        "dent": "#F59E0B",
        "crack": "#DC2626",
        "stain": "#8B5CF6",
        "missing_part": "#EC4899",
        "deformation": "#06B6D4",
        "none": "#10B981"
    }
    color = colors.get(defect_type, "#6B7280")
    conf_text = f" {confidence:.2f}" if confidence else ""
    return f'<span style="background:{color}20; color:{color}; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600;">{defect_type}{conf_text}</span>'


def create_kpi_card(label, value, unit="", color="blue"):
    """Create KPI card HTML"""
    colors = {
        "blue": "#3B82F6",
        "red": "#EF4444",
        "green": "#10B981",
        "yellow": "#F59E0B"
    }
    return f"""
    <div style="background:linear-gradient(135deg,#1e2330,#181c26); border-radius:12px; padding:20px; border-left:4px solid {colors.get(color, '#3B82F6')};">
        <div style="color:#94A3B8; font-size:11px; text-transform:uppercase; letter-spacing:1px;">{label}</div>
        <div style="color:#E2E8F0; font-size:32px; font-weight:800;">{value}</div>
        <div style="color:#64748B; font-size:11px;">{unit}</div>
    </div>
    """