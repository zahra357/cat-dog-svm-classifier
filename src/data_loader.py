from pathlib import Path
import numpy as np

from src.config import CLASS_FOLDERS, CLASS_NAMES


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_image_paths():
    """
    Collect image file paths and corresponding labels from class folders.

    Returns:
        image_paths: np.ndarray of image file paths
        labels: np.ndarray of integer labels
    """
    image_paths = []
    labels = []

    for label, folder_path in CLASS_FOLDERS.items():
        folder_path = Path(folder_path)

        if not folder_path.exists():
            raise FileNotFoundError(
                f"Class folder not found: {folder_path}. "
                f"Expected folder for class '{CLASS_NAMES[label]}'."
            )

        for file_path in folder_path.rglob("*"):
            if file_path.suffix.lower() in VALID_EXTENSIONS:
                image_paths.append(str(file_path))
                labels.append(label)

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    if len(image_paths) == 0:
        raise ValueError("No valid image files were found in the dataset folders.")

    return image_paths, labels


def print_dataset_summary(labels):
    """
    Print number of samples per class.
    """
    print("Dataset summary:")
    for label, class_name in CLASS_NAMES.items():
        count = int(np.sum(labels == label))
        print(f"  Class {label} ({class_name}): {count} images")

    print(f"  Total: {len(labels)} images")