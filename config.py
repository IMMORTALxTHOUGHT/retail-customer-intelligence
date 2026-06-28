import numpy as np

# ── Video Source ──────────────────────────────────────────────
VIDEO_SOURCE = "input.mp4"  # path to input video or 0 for webcam
OUTPUT_PATH = "output/annotated.mp4"

# ── Model ─────────────────────────────────────────────────────
MODEL_NAME = "yolo26n.pt"
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
TRACKER = "bytetrack.yaml"
CLASSES = [0]  # only person

# ── Tracking ──────────────────────────────────────────────────
REAPPEAR_FRAMES = 60  # frames a person can leave before ID resets

# ── Store Zones ──────────────────────────────────────────────
# Each zone: (x1, y1, x2, y2, label)
# Adjust these coordinates to match your video resolution.
# These defaults are for a 1280x720 video.
ZONES = [
    (50,  500, 300, 700,  "Entrance"),
    (350, 500, 650, 700,  "Checkout"),
    (0,   200, 350, 480,  "Aisle-Left"),
    (350, 200, 650, 480,  "Aisle-Center"),
    (650, 200, 1000, 480, "Aisle-Right"),
    (1050, 200, 1280, 480, "Promo-Display"),
]

# ── Dwell Time ────────────────────────────────────────────────
SHORT_DWELL_SEC = 10     # under this = just passing through
LONG_DWELL_SEC = 60      # over this = strong engagement

# ── Heatmap ───────────────────────────────────────────────────
HEATMAP_ALPHA = 0.4       # blend weight for heatmap overlay
HEATMAP_UPDATE_EVERY = 5  # blend every N frames

# ── Dashboard ─────────────────────────────────────────────────
DASHBOARD_WIDTH = 350
DASHBOARD_HEIGHT = 250
DASHBOARD_POSITION = (10, 10)  # top-left corner

# ── Business Health Score Weights ─────────────────────────────
BHS_OCCUPANCY_W = 0.4
BHS_DWELL_W = 0.3
BHS_ZONE_COVERAGE_W = 0.3
