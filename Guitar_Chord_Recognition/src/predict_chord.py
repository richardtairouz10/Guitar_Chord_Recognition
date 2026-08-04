"""
Predict a guitar chord from a WAV audio file.

This script:
1. Loads and preprocesses a new audio recording.
2. Reuses the same feature-extraction functions used for training.
3. Arranges the features in the original training order.
4. Applies the saved StandardScaler.
5. Loads the trained CalibratedClassifierCV model.
6. Predicts the most likely chord.
7. Displays the top chord probabilities.
8. Supports WAV, M4A, MP3, FLAC, and OGG files.
9. Can evaluate an entire folder and save results to CSV.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


SUPPORTED_AUDIO_FORMATS = {
    ".wav",
    ".m4a",
    ".mp3",
    ".flac",
    ".ogg",
}


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_FOLDER = PROJECT_ROOT / "src"

# Make sure Python can import extract_features.py from src.
if str(SRC_FOLDER) not in sys.path:
    sys.path.insert(0, str(SRC_FOLDER))

from extract_features import (  # noqa: E402
    extract_audio_features,
    load_and_preprocess_audio,
)


def load_feature_columns(feature_columns_path: Path) -> list[str]:
    """
    Load the ordered feature-column names created during preprocessing.

    The exact order is important because the model was trained with
    features arranged in this same order.
    """
    if not feature_columns_path.exists():
        raise FileNotFoundError(
            f"Feature-column file was not found: {feature_columns_path}"
        )

    with open(
        feature_columns_path,
        "r",
        encoding="utf-8",
    ) as file:
        saved_data = json.load(file)

    # Support a JSON file containing a direct list:
    # ["mfcc_1_mean", "mfcc_1_std", ...]
    if isinstance(saved_data, list):
        feature_columns = saved_data

    # Also support dictionary formats such as:
    # {"feature_columns": [...]}
    elif isinstance(saved_data, dict):
        possible_keys = [
            "feature_columns",
            "features",
            "columns",
        ]

        feature_columns = None

        for key in possible_keys:
            if key in saved_data and isinstance(saved_data[key], list):
                feature_columns = saved_data[key]
                break

        if feature_columns is None:
            raise ValueError(
                "feature_columns.json does not contain a recognized "
                "feature-column list."
            )

    else:
        raise ValueError(
            "feature_columns.json must contain either a list or dictionary."
        )

    if not feature_columns:
        raise ValueError(
            "No feature columns were found in feature_columns.json."
        )

    return feature_columns



def infer_expected_label(audio_path: Path) -> str:
    """
    Infer the expected class from the recording filename.

    Examples:
        A.m4a          -> A
        A_barre.m4a    -> A
        F#m_barre.m4a  -> F#m
        Noise1.m4a     -> Noise
    """
    filename = audio_path.stem.strip()

    if filename.lower().startswith("noise"):
        return "Noise"

    filename = filename.replace("_barre", "")
    filename = filename.replace("-barre", "")

    return filename


def infer_recording_type(audio_path: Path) -> str:
    """
    Determine whether the recording is an open chord, barre chord,
    or noise sample from its filename.
    """
    filename = audio_path.stem.lower()

    if filename.startswith("noise"):
        return "Noise"

    if "barre" in filename:
        return "Barre"

    return "Open"


def validate_required_files(
    audio_path: Path,
    model_path: Path,
    scaler_path: Path,
    label_encoder_path: Path,
    feature_columns_path: Path,
) -> None:
    """
    Confirm that the audio file and all trained-model resources exist.
    """
    required_files = {
        "Audio recording": audio_path,
        "Calibrated classifier model": model_path,
        "Feature scaler": scaler_path,
        "Label encoder": label_encoder_path,
        "Feature columns": feature_columns_path,
    }

    for description, file_path in required_files.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"{description} was not found: {file_path}"
            )

    if not audio_path.is_file():
        raise ValueError(
            f"The provided audio path is not a file: {audio_path}"
        )

    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_FORMATS))

        raise ValueError(
            f"Unsupported audio format: {audio_path.suffix}. "
            f"Supported formats are: {supported}"
        )


def prepare_audio_features(
    audio_path: Path,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Load the audio, extract its features, and arrange those features
    in exactly the same order used during model training.
    """
    print("\nLoading and preprocessing audio...")

    load_result = load_and_preprocess_audio(audio_path)

    if not isinstance(load_result, tuple) or len(load_result) < 2:
        raise ValueError(
            "load_and_preprocess_audio() must return the audio signal "
            "and sample rate."
        )

    audio = load_result[0]
    sample_rate = load_result[1]

    if audio is None or len(audio) == 0:
        raise ValueError(
            "The audio file could not be loaded or contains no samples."
        )

    print(f"Samples loaded: {len(audio)}")
    print(f"Sample rate: {sample_rate} Hz")
    print(
        f"Duration: {len(audio) / sample_rate:.2f} seconds"
    )

    print("\nExtracting audio features...")

    feature_dictionary = extract_audio_features(
        audio,
        sample_rate,
    )

    if not isinstance(feature_dictionary, dict):
        raise TypeError(
            "extract_audio_features() must return a dictionary."
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in feature_dictionary
    ]

    if missing_features:
        preview = ", ".join(missing_features[:10])

        raise ValueError(
            f"The new recording is missing "
            f"{len(missing_features)} required feature(s). "
            f"Examples: {preview}"
        )

    # Ignore any extra values that were not used to train the model.
    ordered_features = {
        column: feature_dictionary[column]
        for column in feature_columns
    }

    feature_frame = pd.DataFrame(
        [ordered_features],
        columns=feature_columns,
    )

    # Force every feature to be numeric.
    feature_frame = feature_frame.apply(
        pd.to_numeric,
        errors="coerce",
    )

    missing_value_count = int(
        feature_frame.isna().sum().sum()
    )

    if missing_value_count > 0:
        invalid_columns = feature_frame.columns[
            feature_frame.isna().any()
        ].tolist()

        raise ValueError(
            "Feature extraction produced missing or non-numeric "
            f"values in: {invalid_columns}"
        )

    print(
        f"Successfully extracted "
        f"{feature_frame.shape[1]} features."
    )

    return feature_frame


def predict_chord(
    audio_path: Path,
    model_path: Path,
    scaler_path: Path,
    label_encoder_path: Path,
    feature_columns_path: Path,
    top_results: int,
    display_details: bool = True,
) -> dict:
    """
    Generate and display the chord prediction.
    """
    validate_required_files(
        audio_path=audio_path,
        model_path=model_path,
        scaler_path=scaler_path,
        label_encoder_path=label_encoder_path,
        feature_columns_path=feature_columns_path,
    )

    if display_details:
        print("=" * 72)
        print("GUITAR CHORD PREDICTION")
        print("=" * 72)
        print(f"Audio file: {audio_path.resolve()}")

    feature_columns = load_feature_columns(
        feature_columns_path
    )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)

    feature_frame = prepare_audio_features(
        audio_path=audio_path,
        feature_columns=feature_columns,
    )

    print("\nApplying saved feature scaling...")

    scaled_features = scaler.transform(
        feature_frame
    )

    # Convert the scaled values back to a DataFrame so the model receives
    # the same feature names used during training.
    scaled_feature_frame = pd.DataFrame(
        scaled_features,
        columns=feature_columns,
    )

    print("Generating chord prediction...")

    encoded_prediction = int(
        model.predict(scaled_feature_frame)[0]
    )

    predicted_chord = label_encoder.inverse_transform(
        [encoded_prediction]
    )[0]

    prediction_confidence = None

    if display_details:
        print("\n" + "=" * 72)
        print("PREDICTION RESULT")
        print("=" * 72)
        print(f"File: {audio_path.name}")
        print(f"Predicted chord: {predicted_chord}")

    # CalibratedClassifierCV supports probability predictions.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            scaled_feature_frame
        )[0]

        model_classes = np.asarray(
            model.classes_,
            dtype=int,
        )

        probability_order = np.argsort(
            probabilities
        )[::-1]

        top_results = min(
            top_results,
            len(probability_order),
        )

        predicted_class_position = np.where(
            model_classes == encoded_prediction
        )[0]

        if len(predicted_class_position) == 1:
            prediction_confidence = float(
                probabilities[predicted_class_position[0]]
            )

        if display_details:
            print(
                f"\nTop {top_results} predicted classes:"
            )
            print("-" * 45)

            for position, probability_index in enumerate(
                probability_order[:top_results],
                start=1,
            ):
                encoded_class = int(
                    model_classes[probability_index]
                )

                chord_name = label_encoder.inverse_transform(
                    [encoded_class]
                )[0]

                probability = probabilities[
                    probability_index
                ]

                print(
                    f"{position:>2}. "
                    f"{chord_name:<10} "
                    f"{probability * 100:>7.2f}%"
                )

            if prediction_confidence is not None:
                print(
                    f"\nPrediction confidence: "
                    f"{prediction_confidence * 100:.2f}%"
                )

    elif display_details:
        print(
            "\nThe loaded model does not provide "
            "probability estimates."
        )

    if display_details:
        print("=" * 72)

    expected_chord = infer_expected_label(audio_path)
    recording_type = infer_recording_type(audio_path)

    return {
        "file_name": audio_path.name,
        "file_path": str(audio_path.resolve()),
        "expected_chord": expected_chord,
        "predicted_chord": str(predicted_chord),
        "confidence": prediction_confidence,
        "confidence_percent": (
            prediction_confidence * 100
            if prediction_confidence is not None
            else None
        ),
        "correct": str(predicted_chord) == expected_chord,
        "recording_type": recording_type,
    }


def predict_folder(
    folder_path: Path,
    model_path: Path,
    scaler_path: Path,
    label_encoder_path: Path,
    feature_columns_path: Path,
    top_results: int,
    output_path: Path,
) -> None:
    """
    Predict every supported audio file in a folder and save the
    evaluation results to a CSV file.
    """
    if not folder_path.exists():
        raise FileNotFoundError(
            f"Folder was not found: {folder_path}"
        )

    if not folder_path.is_dir():
        raise ValueError(
            f"The provided path is not a folder: {folder_path}"
        )

    audio_files = sorted(
        file_path
        for file_path in folder_path.iterdir()
        if (
            file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_AUDIO_FORMATS
        )
    )

    if not audio_files:
        raise ValueError(
            "No supported audio files were found in "
            f"{folder_path}"
        )

    print("=" * 72)
    print("PERSONAL RECORDING EVALUATION")
    print("=" * 72)
    print(f"Folder: {folder_path.resolve()}")
    print(f"Audio files found: {len(audio_files)}")

    results = []

    for number, audio_file in enumerate(
        audio_files,
        start=1,
    ):
        print(
            f"\nProcessing {number}/{len(audio_files)}: "
            f"{audio_file.name}"
        )

        try:
            result = predict_chord(
                audio_path=audio_file,
                model_path=model_path,
                scaler_path=scaler_path,
                label_encoder_path=label_encoder_path,
                feature_columns_path=feature_columns_path,
                top_results=top_results,
                display_details=False,
            )

            results.append(result)

            status = (
                "CORRECT"
                if result["correct"]
                else "INCORRECT"
            )

            confidence = result["confidence_percent"]
            confidence_text = (
                f"{confidence:.2f}%"
                if confidence is not None
                else "Not available"
            )

            print(
                f"Expected: {result['expected_chord']} | "
                f"Predicted: {result['predicted_chord']} | "
                f"Confidence: {confidence_text} | "
                f"{status}"
            )

        except Exception as error:
            print(
                f"Could not process {audio_file.name}: "
                f"{error}"
            )

            results.append(
                {
                    "file_name": audio_file.name,
                    "file_path": str(audio_file.resolve()),
                    "expected_chord": infer_expected_label(
                        audio_file
                    ),
                    "predicted_chord": None,
                    "confidence": None,
                    "confidence_percent": None,
                    "correct": False,
                    "recording_type": infer_recording_type(
                        audio_file
                    ),
                    "error": str(error),
                }
            )

    results_frame = pd.DataFrame(results)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_frame.to_csv(
        output_path,
        index=False,
    )

    successful_results = results_frame[
        results_frame["predicted_chord"].notna()
    ]

    total_predictions = len(successful_results)
    correct_predictions = int(
        successful_results["correct"].sum()
    )
    incorrect_predictions = (
        total_predictions - correct_predictions
    )

    accuracy = (
        correct_predictions / total_predictions * 100
        if total_predictions > 0
        else 0
    )

    print("\n" + "=" * 72)
    print("PERSONAL RECORDING SUMMARY")
    print("=" * 72)
    print(f"Files found: {len(audio_files)}")
    print(f"Successfully predicted: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {incorrect_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

    if total_predictions > 0:
        print("\nAccuracy by recording type:")
        print("-" * 45)

        type_summary = successful_results.groupby(
            "recording_type"
        )["correct"].agg(
            ["count", "sum"]
        )

        for recording_type, row in type_summary.iterrows():
            type_accuracy = (
                row["sum"] / row["count"] * 100
            )

            print(
                f"{recording_type:<10} "
                f"{int(row['sum'])}/{int(row['count'])} "
                f"correct ({type_accuracy:.2f}%)"
            )

    print(f"\nResults saved to: {output_path.resolve()}")
    print("=" * 72)



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict a guitar chord from one audio file or evaluate "
            "every supported recording in a folder."
        )
    )

    parser.add_argument(
        "audio_path",
        type=Path,
        help=(
            "Path to one audio file or a folder containing "
            "multiple audio recordings."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=(
            PROJECT_ROOT
            / "models"
            / "calibrated_classifier_model.pkl"
        ),
        help="Path to the trained calibrated classifier model.",
    )

    parser.add_argument(
        "--scaler",
        type=Path,
        default=(
            PROJECT_ROOT
            / "processed_data"
            / "feature_scaler.pkl"
        ),
        help="Path to the fitted feature scaler.",
    )

    parser.add_argument(
        "--encoder",
        type=Path,
        default=(
            PROJECT_ROOT
            / "processed_data"
            / "label_encoder.pkl"
        ),
        help="Path to the fitted label encoder.",
    )

    parser.add_argument(
        "--features",
        type=Path,
        default=(
            PROJECT_ROOT
            / "processed_data"
            / "feature_columns.json"
        ),
        help="Path to the ordered feature-column JSON file.",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of highest-probability classes to display.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "personal_recording_predictions.csv"
        ),
        help="CSV output path used when processing a folder.",
    )

    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be at least 1.")

    if args.audio_path.is_dir():
        predict_folder(
            folder_path=args.audio_path,
            model_path=args.model,
            scaler_path=args.scaler,
            label_encoder_path=args.encoder,
            feature_columns_path=args.features,
            top_results=args.top,
            output_path=args.output,
        )
    else:
        predict_chord(
            audio_path=args.audio_path,
            model_path=args.model,
            scaler_path=args.scaler,
            label_encoder_path=args.encoder,
            feature_columns_path=args.features,
            top_results=args.top,
        )


if __name__ == "__main__":
    main()