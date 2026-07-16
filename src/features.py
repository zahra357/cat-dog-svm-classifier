import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog

from src.config import IMAGE_SIZE, HOG_PARAMS


def load_image_as_array(image_path, image_size=IMAGE_SIZE):
    """
    Load image, convert to grayscale, resize, and return as numpy array.

    Returns:
        arr: 2D numpy array with values in [0, 1]
    """
    try:
        img = Image.open(image_path).convert("L")
        img = img.resize(image_size)

        arr = np.asarray(img, dtype=np.float32) / 255.0

        return arr

    except (UnidentifiedImageError, OSError, ValueError) as error:
        print(f"Warning: Could not read image: {image_path}. Error: {error}")
        return None


def extract_raw_pixels(image_array):
    """
    Convert image array to raw pixel feature vector.
    """
    return image_array.flatten()


def extract_hog_features(image_array):
    """
    Extract HOG features from grayscale image array.
    """
    return hog(
        image_array,
        orientations=HOG_PARAMS["orientations"],
        pixels_per_cell=HOG_PARAMS["pixels_per_cell"],
        cells_per_block=HOG_PARAMS["cells_per_block"],
        block_norm=HOG_PARAMS["block_norm"],
        feature_vector=HOG_PARAMS["feature_vector"]
    )


def build_features(image_paths, labels):
    """
    Build raw pixel and HOG features for all valid images.

    Returns:
        X_raw: raw pixel features
        X_hog: HOG features
        y: labels for successfully loaded images
        valid_paths: image paths for successfully loaded images
    """
    raw_features = []
    hog_features = []
    valid_labels = []
    valid_paths = []

    for idx, image_path in enumerate(image_paths):
        image_array = load_image_as_array(image_path)

        if image_array is None:
            continue

        raw_vector = extract_raw_pixels(image_array)
        hog_vector = extract_hog_features(image_array)

        raw_features.append(raw_vector)
        hog_features.append(hog_vector)
        valid_labels.append(labels[idx])
        valid_paths.append(image_path)

    X_raw = np.array(raw_features, dtype=np.float32)
    X_hog = np.array(hog_features, dtype=np.float32)
    y = np.array(valid_labels, dtype=np.int64)
    valid_paths = np.array(valid_paths)

    return X_raw, X_hog, y, valid_paths