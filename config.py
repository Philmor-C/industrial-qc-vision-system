# Configuration
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
MEASUREMENT_DIR = os.path.join(BASE_DIR, "measurement_samples")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_4_1280.onnx")

# UI Settings
UI_TITLE = "Industrial QC Vision System"
UI_THEME = "dark"
SERVER_PORT = 7860
SERVER_NAME = "127.0.0.1"

# Display settings
DISPLAY_SCALE_PERCENT = 7   # Scale measurement image for display