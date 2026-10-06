"""
Project configuration and shared constants for VisDrone + SAHI + DAS.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_COCO = ROOT / "data" / "coco"
DATA_TUNING = ROOT / "data" / "tuning"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CONFIGS_DIR = ROOT / "configs"

# VisDrone categories (1-10 kept; drop ignored=0, others=11)
VISDRONE_CATEGORIES = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "people"},
    {"id": 3, "name": "bicycle"},
    {"id": 4, "name": "car"},
    {"id": 5, "name": "van"},
    {"id": 6, "name": "truck"},
    {"id": 7, "name": "tricycle"},
    {"id": 8, "name": "awning-tricycle"},
    {"id": 9, "name": "bus"},
    {"id": 10, "name": "motor"},
]

# Contiguous 0-indexed class names used by VisDrone-finetuned YOLO weights
YOLO_VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

# Map VisDrone class id (1-10) -> COCO class name for D-FINE cross-dataset eval
VISDRONE_TO_COCO = {
    1: "person",      # pedestrian
    2: "person",      # people
    3: "bicycle",
    4: "car",
    5: "car",         # van ~ car
    6: "truck",
    7: None,          # tricycle - no COCO
    8: None,          # awning-tricycle
    9: "bus",
    10: "motorcycle",  # motor
}

# HuggingFace model zoo
IN_DOMAIN_MODELS = {
    "yolov8n": {"repo": "dronefreak/visdrone-yolov8n", "filename": "best.pt", "family": "yolo"},
    "yolov8s": {"repo": "dronefreak/visdrone-yolov8s", "filename": "best.pt", "family": "yolo"},
    "yolov11n": {"repo": "dronefreak/visdrone-yolov11n", "filename": "best.pt", "family": "yolo"},
    "yolov11s": {"repo": "dronefreak/visdrone-yolov11s", "filename": "best.pt", "family": "yolo"},
}

CROSS_DATASET_MODELS = {
    "dfine_n": {"repo": "ustc-community/dfine-nano-coco", "family": "dfine"},
    "dfine_s": {"repo": "ustc-community/dfine-small-coco", "family": "dfine"},
}

# Ultralytics VisDrone download URLs
VISDRONE_URLS = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    "test-dev": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
}

DEFAULT_CONF = 0.25
COARSE_CONF = 0.05
DEFAULT_DEVICE = "cpu"
NUM_THREADS = 4
LATENCY_REPEATS = 3
WARMUP_ITERS = 3
TUNING_SEED = 42
TUNING_N_IMAGES = 200
