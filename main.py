import numpy as np
from src.visualization import plot_class_distribution

from src.config import (
    FEATURES_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    RESULTS_DIR
)

from src.utils import create_output_dirs
from src.data_loader import collect_image_paths, print_dataset_summary
from src.features import build_features
from src.train import train_all_models


def main():
    create_output_dirs(
        FEATURES_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        RESULTS_DIR
    )

    print("Collecting image paths...")
    image_paths, labels = collect_image_paths()

    print_dataset_summary(labels)

    plot_class_distribution(
    labels,
    FIGURES_DIR / "class_distribution.png"
)

    print("\nBuilding raw pixel and HOG features...")
    X_raw, X_hog, y, valid_paths = build_features(image_paths, labels)

    print("\nFeature extraction completed.")
    print(f"X_raw shape: {X_raw.shape}")
    print(f"X_hog shape: {X_hog.shape}")
    print(f"y shape: {y.shape}")
    print(f"valid image paths: {valid_paths.shape}")

    np.save(FEATURES_DIR / "X_raw.npy", X_raw)
    np.save(FEATURES_DIR / "X_hog.npy", X_hog)
    np.save(FEATURES_DIR / "y.npy", y)
    np.save(FEATURES_DIR / "image_paths.npy", valid_paths)

    print("\nSaved features to outputs/features/")

    print("\nStarting model training and evaluation...")
    train_all_models()


if __name__ == "__main__":
    main()