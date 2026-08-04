"""
Train and evaluate the CalibratedClassifierCV model.

This script:
1. Loads the processed training and testing data.
2. Loads the saved label encoder.
3. Trains CalibratedClassifierCV.
4. Evaluates the model on the test set.
5. Saves the trained model.
6. Saves metrics, predictions, misclassifications,
   classification report, and confusion matrix.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def load_processed_data(processed_folder: Path):
    """
    Load all processed datasets and supporting files.
    """

    required_files = {
        "X_train": processed_folder / "X_train.csv",
        "X_test": processed_folder / "X_test.csv",
        "y_train": processed_folder / "y_train.csv",
        "y_test": processed_folder / "y_test.csv",
        "label_encoder": processed_folder / "label_encoder.pkl",
        "test_metadata": processed_folder / "test_metadata.csv",
    }

    for name, file_path in required_files.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file for {name} was not found: {file_path}"
            )

    X_train = pd.read_csv(required_files["X_train"])
    X_test = pd.read_csv(required_files["X_test"])

    y_train_df = pd.read_csv(required_files["y_train"])
    y_test_df = pd.read_csv(required_files["y_test"])

    test_metadata = pd.read_csv(required_files["test_metadata"])

    label_encoder = joblib.load(required_files["label_encoder"])

    if "label" not in y_train_df.columns:
        raise ValueError(
            "y_train.csv must contain a column named 'label'."
        )

    if "label" not in y_test_df.columns:
        raise ValueError(
            "y_test.csv must contain a column named 'label'."
        )

    y_train = y_train_df["label"]
    y_test = y_test_df["label"]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        test_metadata,
    )


def validate_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    test_metadata: pd.DataFrame,
) -> None:
    """
    Validate that all input files are compatible.
    """

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train have different numbers of rows."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test have different numbers of rows."
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "X_train and X_test do not have matching feature columns."
        )

    if X_train.isna().any().any():
        raise ValueError("X_train contains missing values.")

    if X_test.isna().any().any():
        raise ValueError("X_test contains missing values.")

    if y_train.isna().any():
        raise ValueError("y_train contains missing values.")

    if y_test.isna().any():
        raise ValueError("y_test contains missing values.")

    if len(test_metadata) != len(X_test):
        raise ValueError(
            "test_metadata.csv and X_test.csv have different "
            "numbers of rows."
        )


def decode_labels(
    label_encoder,
    encoded_labels,
) -> np.ndarray:
    """
    Convert encoded integer labels back into readable chord names.
    """

    encoded_labels = np.asarray(encoded_labels, dtype=int)

    return label_encoder.inverse_transform(encoded_labels)


def main(
    processed_folder: Path,
    models_folder: Path,
    results_folder: Path,
) -> None:
    """
    Train, evaluate, and save CalibratedClassifierCV.
    """

    print("=" * 72)
    print("CALIBRATED CLASSIFIER TRAINING AND EVALUATION")
    print("=" * 72)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        test_metadata,
    ) = load_processed_data(processed_folder)

    validate_data(
        X_train,
        X_test,
        y_train,
        y_test,
        test_metadata,
    )

    print(f"Training recordings: {len(X_train)}")
    print(f"Testing recordings: {len(X_test)}")
    print(f"Audio features: {X_train.shape[1]}")
    print(f"Chord classes: {y_train.nunique()}")

    models_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    model = CalibratedClassifierCV(
        cv=5,
        method="sigmoid",
        n_jobs=-1,
    )

    print("\nTraining CalibratedClassifierCV...")

    training_start = perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    training_time = perf_counter() - training_start

    print(
        f"Training completed in "
        f"{training_time:.3f} seconds."
    )

    print("\nGenerating predictions...")

    prediction_start = perf_counter()

    y_pred = model.predict(X_test)

    prediction_time = perf_counter() - prediction_start

    print(
        f"Prediction completed in "
        f"{prediction_time:.3f} seconds."
    )

    # ------------------------------------------------------------
    # Calculate overall evaluation metrics
    # ------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred,
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    y_test_array = np.asarray(y_test, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)

    correct_predictions = int(
        np.sum(y_test_array == y_pred_array)
    )

    incorrect_predictions = int(
        len(y_test_array) - correct_predictions
    )

    # ------------------------------------------------------------
    # Decode labels into readable chord names
    # ------------------------------------------------------------

    actual_labels = decode_labels(
        label_encoder,
        y_test_array,
    )

    predicted_labels = decode_labels(
        label_encoder,
        y_pred_array,
    )

    class_names = label_encoder.classes_.tolist()

    encoded_classes = list(
        range(len(class_names))
    )

    # ------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------

    report_dictionary = classification_report(
        y_test_array,
        y_pred_array,
        labels=encoded_classes,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    classification_report_df = pd.DataFrame(
        report_dictionary
    ).transpose()

    classification_report_path = (
        results_folder
        / "calibrated_classifier_classification_report.csv"
    )

    classification_report_df.to_csv(
        classification_report_path,
    )

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    matrix = confusion_matrix(
        y_test_array,
        y_pred_array,
        labels=encoded_classes,
    )

    confusion_matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual_{class_name}"
            for class_name in class_names
        ],
        columns=[
            f"Predicted_{class_name}"
            for class_name in class_names
        ],
    )

    confusion_matrix_path = (
        results_folder
        / "calibrated_classifier_confusion_matrix.csv"
    )

    confusion_matrix_df.to_csv(
        confusion_matrix_path,
    )

    # ------------------------------------------------------------
    # Save all test predictions
    # ------------------------------------------------------------

    predictions_df = test_metadata.reset_index(
        drop=True
    ).copy()

    predictions_df["actual_encoded"] = y_test_array
    predictions_df["predicted_encoded"] = y_pred_array
    predictions_df["actual_label"] = actual_labels
    predictions_df["predicted_label"] = predicted_labels
    predictions_df["correct"] = (
        y_test_array == y_pred_array
    )

    predictions_path = (
        results_folder
        / "calibrated_classifier_test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Save only incorrect predictions
    # ------------------------------------------------------------

    misclassified_df = predictions_df[
        predictions_df["correct"] == False
    ].copy()

    misclassified_path = (
        results_folder
        / "calibrated_classifier_misclassified_recordings.csv"
    )

    misclassified_df.to_csv(
        misclassified_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Save overall metrics
    # ------------------------------------------------------------

    metrics = {
        "model": "CalibratedClassifierCV",
        "training_recordings": int(len(X_train)),
        "testing_recordings": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "class_count": int(y_train.nunique()),
        "correct_predictions": correct_predictions,
        "incorrect_predictions": incorrect_predictions,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "training_time_seconds": float(training_time),
        "prediction_time_seconds": float(prediction_time),
    }

    metrics_json_path = (
        results_folder
        / "calibrated_classifier_metrics.json"
    )

    with open(
        metrics_json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "CalibratedClassifierCV",
                "Training Recordings": len(X_train),
                "Testing Recordings": len(X_test),
                "Feature Count": X_train.shape[1],
                "Class Count": y_train.nunique(),
                "Accuracy": accuracy,
                "Balanced Accuracy": balanced_accuracy,
                "Macro Precision": macro_precision,
                "Macro Recall": macro_recall,
                "Macro F1": macro_f1,
                "Weighted F1": weighted_f1,
                "Correct Predictions": correct_predictions,
                "Incorrect Predictions": incorrect_predictions,
                "Training Time Seconds": training_time,
                "Prediction Time Seconds": prediction_time,
            }
        ]
    )

    metrics_csv_path = (
        results_folder
        / "calibrated_classifier_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_csv_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Save trained model
    # ------------------------------------------------------------

    model_path = (
        models_folder
        / "calibrated_classifier_model.pkl"
    )

    joblib.dump(
        model,
        model_path,
    )

    # ------------------------------------------------------------
    # Print final results
    # ------------------------------------------------------------

    print("\n" + "=" * 72)
    print("CALIBRATED CLASSIFIER RESULTS")
    print("=" * 72)

    print(f"Accuracy:          {accuracy:.4f}")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy:.4f}"
    )
    print(
        f"Macro precision:   "
        f"{macro_precision:.4f}"
    )
    print(f"Macro recall:      {macro_recall:.4f}")
    print(f"Macro F1 score:    {macro_f1:.4f}")
    print(
        f"Weighted F1 score: "
        f"{weighted_f1:.4f}"
    )

    print(
        f"\nCorrect predictions:   "
        f"{correct_predictions}"
    )
    print(
        f"Incorrect predictions: "
        f"{incorrect_predictions}"
    )

    print("\nSaved files:")
    print(f"  Model: {model_path.resolve()}")
    print(f"  Metrics CSV: {metrics_csv_path.resolve()}")
    print(f"  Metrics JSON: {metrics_json_path.resolve()}")
    print(
        f"  Classification report: "
        f"{classification_report_path.resolve()}"
    )
    print(
        f"  Confusion matrix: "
        f"{confusion_matrix_path.resolve()}"
    )
    print(
        f"  Test predictions: "
        f"{predictions_path.resolve()}"
    )
    print(
        f"  Misclassified recordings: "
        f"{misclassified_path.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate CalibratedClassifierCV "
            "for guitar chord recognition."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("processed_data"),
        help=(
            "Folder containing processed training "
            "and testing files."
        ),
    )

    parser.add_argument(
        "--models",
        type=Path,
        default=Path("models"),
        help=(
            "Folder where the trained model "
            "will be saved."
        ),
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help=(
            "Folder where evaluation results "
            "will be saved."
        ),
    )

    args = parser.parse_args()

    main(
        processed_folder=args.input,
        models_folder=args.models,
        results_folder=args.results,
    )