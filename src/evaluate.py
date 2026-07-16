import re
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from src.config import CLASS_NAMES


def make_safe_filename(name):
    """
    Convert a model name to a safe filename.
    """
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name,
    feature_type
):
    """
    Evaluate a trained model on train and test sets.

    Returns:
        result: dictionary of evaluation metrics
        y_test_pred: predictions on test set
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    overfitting_gap = train_accuracy - test_accuracy

    result = {
        "feature_type": feature_type,
        "model_name": model_name,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "overfitting_gap": overfitting_gap
    }

    return result, y_test_pred


def save_confusion_matrix(
    y_true,
    y_pred,
    title,
    save_path
):
    """
    Save confusion matrix as an image.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[CLASS_NAMES[0], CLASS_NAMES[1]]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", values_format="d")

    ax.set_title(title)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def save_classification_report(
    y_true,
    y_pred,
    save_path
):
    """
    Save classification report as a text file.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        zero_division=0
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as file:
        file.write(report)