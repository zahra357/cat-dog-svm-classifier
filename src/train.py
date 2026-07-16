import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.config import (
    FEATURES_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    RANDOM_STATE,
    TEST_SIZE
)

from src.evaluate import (
    evaluate_model,
    save_confusion_matrix,
    save_classification_report,
    make_safe_filename
)

from src.visualization import (
    plot_model_comparison,
    save_misclassified_examples
)

def load_saved_features():
    """
    Load previously extracted raw pixel and HOG features.
    """
    X_raw = np.load(FEATURES_DIR / "X_raw.npy")
    X_hog = np.load(FEATURES_DIR / "X_hog.npy")
    y = np.load(FEATURES_DIR / "y.npy")
    image_paths = np.load(FEATURES_DIR / "image_paths.npy", allow_pickle=True)

    return X_raw, X_hog, y, image_paths


def split_dataset(X_raw, X_hog, y, image_paths):
    """
    Split raw features, HOG features, labels, and image paths consistently.
    """
    return train_test_split(
        X_raw,
        X_hog,
        y,
        image_paths,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def build_svm_pipeline(kernel, C=1.0, gamma="scale"):
    """
    Build an SVM pipeline with StandardScaler.
    """
    if kernel == "linear":
        svm = SVC(
            kernel="linear",
            C=C,
            random_state=RANDOM_STATE
        )
    elif kernel == "rbf":
        svm = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            random_state=RANDOM_STATE
        )
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm)
    ])

    return pipeline


def run_manual_svm_experiments(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_type
):
    """
    Train SVM models manually with different kernels and hyperparameters.
    """
    results = []
    trained_models = {}

    C_values = [0.1, 1, 10, 100]
    gamma_values = ["scale", 0.001, 0.01, 0.1]

    print(f"\nRunning manual SVM experiments on {feature_type} features...")

    for C in C_values:
        model_name = f"SVM_linear_C={C}"
        print(f"Training {feature_type} - {model_name}")

        model = build_svm_pipeline(kernel="linear", C=C)
        model.fit(X_train, y_train)

        result, y_pred = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            model_name=model_name,
            feature_type=feature_type
        )

        result["kernel"] = "linear"
        result["C"] = C
        result["gamma"] = "-"
        results.append(result)

        trained_models[f"{feature_type}_{model_name}"] = model

    for C in C_values:
        for gamma in gamma_values:
            model_name = f"SVM_rbf_C={C}_gamma={gamma}"
            print(f"Training {feature_type} - {model_name}")

            model = build_svm_pipeline(kernel="rbf", C=C, gamma=gamma)
            model.fit(X_train, y_train)

            result, y_pred = evaluate_model(
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                model_name=model_name,
                feature_type=feature_type
            )

            result["kernel"] = "rbf"
            result["C"] = C
            result["gamma"] = gamma
            results.append(result)

            trained_models[f"{feature_type}_{model_name}"] = model

    return results, trained_models


def run_grid_search_svm(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Run GridSearchCV for SVM on HOG features.
    """
    print("\nRunning GridSearchCV on HOG features...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(random_state=RANDOM_STATE))
    ])

    param_grid = [
        {
            "svm__kernel": ["linear"],
            "svm__C": [0.1, 1, 10, 100]
        },
        {
            "svm__kernel": ["rbf"],
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": ["scale", 0.001, 0.01, 0.1]
        }
    ]

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("\nBest GridSearchCV parameters:")
    print(best_params)

    result, y_pred = evaluate_model(
        best_model,
        X_train,
        X_test,
        y_train,
        y_test,
        model_name="SVM_GridSearchCV_best",
        feature_type="HOG"
    )

    result["kernel"] = best_params.get("svm__kernel", "-")
    result["C"] = best_params.get("svm__C", "-")
    result["gamma"] = best_params.get("svm__gamma", "-")
    result["best_cv_score"] = grid_search.best_score_

    grid_results = pd.DataFrame(grid_search.cv_results_)
    grid_results.to_csv(
        RESULTS_DIR / "grid_search_results.csv",
        index=False
    )

    save_confusion_matrix(
        y_test,
        y_pred,
        title="Confusion Matrix - Best SVM GridSearchCV on HOG",
        save_path=FIGURES_DIR / "confusion_matrix_best_svm_gridsearch_hog.png"
    )

    save_classification_report(
        y_test,
        y_pred,
        save_path=RESULTS_DIR / "classification_report_best_svm_gridsearch_hog.txt"
    )

    joblib.dump(
        best_model,
        MODELS_DIR / "best_svm_gridsearch_hog.joblib"
    )

    return result, best_model


def run_comparison_models(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Compare SVM with Logistic Regression and KNN on HOG features.
    """
    print("\nRunning comparison models on HOG features...")

    models = {
        "Logistic_Regression_HOG": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE
            ))
        ]),
        "KNN_HOG_k=5": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ])
    }

    results = []
    trained_models = {}

    for model_name, model in models.items():
        print(f"Training {model_name}")

        model.fit(X_train, y_train)

        result, y_pred = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            model_name=model_name,
            feature_type="HOG"
        )

        result["kernel"] = "-"
        result["C"] = "-"
        result["gamma"] = "-"

        results.append(result)
        trained_models[model_name] = model

        safe_name = make_safe_filename(model_name)

        save_confusion_matrix(
            y_test,
            y_pred,
            title=f"Confusion Matrix - {model_name}",
            save_path=FIGURES_DIR / f"confusion_matrix_{safe_name}.png"
        )

        save_classification_report(
            y_test,
            y_pred,
            save_path=RESULTS_DIR / f"classification_report_{safe_name}.txt"
        )

        joblib.dump(
            model,
            MODELS_DIR / f"{safe_name}.joblib"
        )

    return results, trained_models


def save_best_manual_model_confusion_matrix(
    results_df,
    trained_models,
    X_train_raw,
    X_test_raw,
    X_train_hog,
    X_test_hog,
    y_train,
    y_test,
    paths_test
):
    """
    Save confusion matrix and model file for the best manual SVM model.
    """
    manual_svm_df = results_df[
        results_df["model_name"].str.contains("SVM_", regex=False)
        & ~results_df["model_name"].str.contains("GridSearchCV", regex=False)
    ].copy()

    best_row = manual_svm_df.sort_values(
        by=["test_accuracy", "f1_score", "overfitting_gap"],
        ascending=[False, False, True]
    ).iloc[0]

    feature_type = best_row["feature_type"]
    model_name = best_row["model_name"]
    model_key = f"{feature_type}_{model_name}"

    best_model = trained_models[model_key]

    if feature_type == "Raw Pixels":
        X_train = X_train_raw
        X_test = X_test_raw
    else:
        X_train = X_train_hog
        X_test = X_test_hog

    _, y_pred = evaluate_model(
        best_model,
        X_train,
        X_test,
        y_train,
        y_test,
        model_name=model_name,
        feature_type=feature_type
    )

    safe_name = make_safe_filename(f"best_manual_{feature_type}_{model_name}")

    save_confusion_matrix(
        y_test,
        y_pred,
        title=f"Confusion Matrix - Best Manual SVM ({feature_type})",
        save_path=FIGURES_DIR / f"confusion_matrix_{safe_name}.png"
    )

    save_classification_report(
        y_test,
        y_pred,
        save_path=RESULTS_DIR / f"classification_report_{safe_name}.txt"
    )

    joblib.dump(
        best_model,
        MODELS_DIR / f"{safe_name}.joblib"
    )

    save_misclassified_examples(
    image_paths=paths_test,
    y_true=y_test,
    y_pred=y_pred,
    save_path=FIGURES_DIR / "misclassified_examples_best_manual_svm.png",
    max_examples=12
    )

    print("\nBest manual SVM model selected:")
    print(best_row)


def train_all_models():
    """
    Full training and evaluation pipeline.
    """
    X_raw, X_hog, y, image_paths = load_saved_features()

    (
        X_raw_train,
        X_raw_test,
        X_hog_train,
        X_hog_test,
        y_train,
        y_test,
        paths_train,
        paths_test
    ) = split_dataset(X_raw, X_hog, y, image_paths)

    print("\nTrain/Test split completed.")
    print(f"X_raw_train shape: {X_raw_train.shape}")
    print(f"X_raw_test shape: {X_raw_test.shape}")
    print(f"X_hog_train shape: {X_hog_train.shape}")
    print(f"X_hog_test shape: {X_hog_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    all_results = []
    all_manual_models = {}

    raw_results, raw_models = run_manual_svm_experiments(
        X_raw_train,
        X_raw_test,
        y_train,
        y_test,
        feature_type="Raw Pixels"
    )

    hog_results, hog_models = run_manual_svm_experiments(
        X_hog_train,
        X_hog_test,
        y_train,
        y_test,
        feature_type="HOG"
    )

    all_results.extend(raw_results)
    all_results.extend(hog_results)

    all_manual_models.update(raw_models)
    all_manual_models.update(hog_models)

    grid_result, best_grid_model = run_grid_search_svm(
        X_hog_train,
        X_hog_test,
        y_train,
        y_test
    )

    all_results.append(grid_result)

    comparison_results, comparison_models = run_comparison_models(
        X_hog_train,
        X_hog_test,
        y_train,
        y_test
    )

    all_results.extend(comparison_results)

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by=["test_accuracy", "f1_score"],
        ascending=False
    )

    results_df.to_csv(
        RESULTS_DIR / "final_model_comparison.csv",
        index=False
    )

    plot_model_comparison(
    results_df,
    save_path=FIGURES_DIR / "model_comparison_top15.png",
    top_n=15
    )

    save_best_manual_model_confusion_matrix(
        results_df=results_df,
        trained_models=all_manual_models,
        X_train_raw=X_raw_train,
        X_test_raw=X_raw_test,
        X_train_hog=X_hog_train,
        X_test_hog=X_hog_test,
        y_train=y_train,
        y_test=y_test,
        paths_test=paths_test
    )

    np.save(RESULTS_DIR / "paths_test.npy", paths_test)
    np.save(RESULTS_DIR / "y_test.npy", y_test)

    print("\nTraining and evaluation completed.")
    print("\nTop results:")
    print(results_df.head(10))

    print("\nSaved final comparison to:")
    print(RESULTS_DIR / "final_model_comparison.csv")

    return results_df