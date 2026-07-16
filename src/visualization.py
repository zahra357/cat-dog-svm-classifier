from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.config import CLASS_NAMES


def plot_class_distribution(y, save_path):
    """
    Plot and save class distribution.
    """
    labels = sorted(CLASS_NAMES.keys())
    counts = [int(np.sum(y == label)) for label in labels]
    names = [CLASS_NAMES[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, counts)

    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")

    for i, count in enumerate(counts):
        ax.text(i, count + 5, str(count), ha="center")

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_model_comparison(results_df, save_path, top_n=15):
    """
    Plot top models based on test accuracy.
    """
    df = results_df.copy()
    df = df.sort_values(
        by=["test_accuracy", "f1_score"],
        ascending=False
    ).head(top_n)

    labels = [
        f"{row['feature_type']} | {row['model_name']}"
        for _, row in df.iterrows()
    ]

    test_scores = df["test_accuracy"].values

    fig_height = max(6, 0.45 * len(df))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_positions = np.arange(len(labels))
    ax.barh(y_positions, test_scores)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Test Accuracy")
    ax.set_title(f"Top {top_n} Model Comparison")

    for i, score in enumerate(test_scores):
        ax.text(score + 0.005, i, f"{score:.3f}", va="center")

    ax.set_xlim(0, min(1.0, max(test_scores) + 0.12))

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def save_misclassified_examples(
    image_paths,
    y_true,
    y_pred,
    save_path,
    max_examples=12
):
    """
    Save a grid of misclassified test images.
    """
    image_paths = np.array(image_paths)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    wrong_indices = np.where(y_true != y_pred)[0]

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(wrong_indices) == 0:
        with open(save_path.with_suffix(".txt"), "w", encoding="utf-8") as file:
            file.write("No misclassified examples were found.")
        return

    selected_indices = wrong_indices[:max_examples]

    cols = 4
    rows = math.ceil(len(selected_indices) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))

    if rows == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for plot_idx, data_idx in enumerate(selected_indices):
        ax = axes[plot_idx]

        try:
            img = Image.open(image_paths[data_idx]).convert("RGB")
            img = img.resize((128, 128))

            true_label = CLASS_NAMES[int(y_true[data_idx])]
            pred_label = CLASS_NAMES[int(y_pred[data_idx])]

            ax.imshow(img)
            ax.set_title(f"True: {true_label}\nPred: {pred_label}")
            ax.axis("off")

        except Exception as error:
            ax.set_title(f"Could not load image\n{error}")
            ax.axis("off")

    plt.suptitle("Misclassified Test Examples", fontsize=14)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close(fig)