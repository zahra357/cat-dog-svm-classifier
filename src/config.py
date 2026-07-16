from pathlib import Path

# Root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = PROJECT_ROOT / "data" / "raw"

CAT_DIR = DATA_DIR / "cat"
DOG_DIR = DATA_DIR / "dog"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_DIR = OUTPUT_DIR / "features"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

# Image settings
IMAGE_SIZE = (64, 64)
GRAYSCALE = True

# ML settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Class labels
CLASS_NAMES = {
    0: "cat",
    1: "dog"
}

CLASS_FOLDERS = {
    0: CAT_DIR,
    1: DOG_DIR
}

# HOG settings
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "feature_vector": True
}