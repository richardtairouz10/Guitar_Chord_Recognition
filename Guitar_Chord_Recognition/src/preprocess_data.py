"""
preprocess_data.py

This script prepares the extracted guitar-chord audio features
for machine-learning model training.

It performs the following tasks:

1. Loads the feature CSV created by extract_features.py.
2. Checks for missing or invalid values.
3. Separates numerical features from identifying columns.
4. Encodes chord labels as integers.
5. Splits the dataset into training and testing sets.
6. Standardizes the numerical features.
7. Saves the processed datasets and preprocessing objects.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


DEFAULT_TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_feature_data(input_file):
    """
    Load the feature CSV into a Pandas DataFrame.

    Parameters
    ----------
    input_file : pathlib.Path
        Path to the CSV created by extract_features.py.

    Returns
    -------
    pandas.DataFrame
        Loaded feature dataset.
    """

    if not input_file.exists():
        raise FileNotFoundError(
            f"The feature file does not exist:\n{input_file}"
        )

    if input_file.suffix.lower() != ".csv":
        raise ValueError("The input file must be a CSV file.")

    dataframe = pd.read_csv(input_file)

    if dataframe.empty:
        raise ValueError("The feature CSV is empty.")

    print(f"Loaded dataset: {input_file}")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")

    return dataframe


def inspect_dataset(dataframe):
    """
    Print basic information about the dataset before preprocessing.
    """

    print()
    print("=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)

    if "label" not in dataframe.columns:
        raise ValueError(
            "The dataset does not contain a 'label' column."
        )

    print(f"Number of recordings: {len(dataframe)}")
    print(f"Number of chord classes: {dataframe['label'].nunique()}")

    print()
    print("Recordings per class:")
    print(dataframe["label"].value_counts().sort_index())

    duplicate_count = dataframe.duplicated().sum()
    missing_count = dataframe.isna().sum().sum()

    print()
    print(f"Duplicate rows: {duplicate_count}")
    print(f"Missing values: {missing_count}")


def clean_dataset(dataframe):
    """
    Clean missing, duplicate, and invalid values.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Original feature dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned feature dataset.
    """

    cleaned_dataframe = dataframe.copy()

    original_row_count = len(cleaned_dataframe)

    # Remove completely duplicated rows.
    cleaned_dataframe = cleaned_dataframe.drop_duplicates()

    removed_duplicates = original_row_count - len(cleaned_dataframe)

    if removed_duplicates > 0:
        print(f"Removed {removed_duplicates} duplicate rows.")

    # Replace positive and negative infinity with missing values.
    cleaned_dataframe = cleaned_dataframe.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Identify numerical columns.
    numerical_columns = cleaned_dataframe.select_dtypes(
        include=[np.number]
    ).columns

    # Fill missing numerical values using each column's median.
    for column in numerical_columns:
        if cleaned_dataframe[column].isna().any():
            median_value = cleaned_dataframe[column].median()
            cleaned_dataframe[column] = (
                cleaned_dataframe[column].fillna(median_value)
            )

    # Remove rows without a valid label.
    cleaned_dataframe = cleaned_dataframe.dropna(subset=["label"])

    cleaned_dataframe["label"] = (
        cleaned_dataframe["label"]
        .astype(str)
        .str.strip()
    )

    # Remove rows whose label is blank.
    cleaned_dataframe = cleaned_dataframe[
        cleaned_dataframe["label"] != ""
    ]

    if cleaned_dataframe.empty:
        raise ValueError(
            "No valid rows remain after cleaning the dataset."
        )

    return cleaned_dataframe


def separate_features_and_labels(dataframe):
    """
    Separate input features from target labels.

    Identifying columns are not used to train the model because filenames
    and file paths do not describe the sound of a chord.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Cleaned feature dataset.

    Returns
    -------
    X : pandas.DataFrame
        Numerical input features.

    y : pandas.Series
        Chord labels.

    metadata : pandas.DataFrame
        Filename and file-path information retained for reference.
    """

    identifying_columns = [
        column
        for column in ["file_name", "file_path"]
        if column in dataframe.columns
    ]

    metadata = dataframe[identifying_columns].copy()

    columns_to_remove = identifying_columns + ["label"]

    X = dataframe.drop(
        columns=columns_to_remove,
        errors="ignore"
    )

    y = dataframe["label"].copy()

    # Keep only numerical columns.
    X = X.select_dtypes(include=[np.number])

    if X.empty:
        raise ValueError(
            "No numerical feature columns were found."
        )

    # Remove constant columns.
    # A constant feature has the same value for every recording and gives
    # the model no useful information.
    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique() <= 1
    ]

    if constant_columns:
        print()
        print(
            f"Removing {len(constant_columns)} constant feature columns."
        )

        X = X.drop(columns=constant_columns)

    print()
    print(f"Numerical features selected: {X.shape[1]}")

    return X, y, metadata


def encode_labels(labels):
    """
    Convert text chord labels into integer values.

    Example:
        A_major -> 0
        A_minor -> 1
        B_major -> 2

    Parameters
    ----------
    labels : pandas.Series
        Original text labels.

    Returns
    -------
    encoded_labels : numpy.ndarray
        Integer labels.

    label_encoder : sklearn.preprocessing.LabelEncoder
        Fitted label encoder.
    """

    label_encoder = LabelEncoder()

    encoded_labels = label_encoder.fit_transform(labels)

    print()
    print("Label encoding:")

    for encoded_value, class_name in enumerate(
        label_encoder.classes_
    ):
        print(f"  {encoded_value}: {class_name}")

    return encoded_labels, label_encoder


def split_dataset(
    features,
    labels,
    metadata,
    test_size=DEFAULT_TEST_SIZE
):
    """
    Divide the dataset into training and testing portions.

    Stratification preserves approximately the same class distribution
    in the training and testing sets.

    Parameters
    ----------
    features : pandas.DataFrame
        Numerical feature matrix.

    labels : numpy.ndarray
        Encoded target labels.

    metadata : pandas.DataFrame
        Filename and file-path information.

    test_size : float
        Fraction of recordings reserved for testing.

    Returns
    -------
    tuple
        Training and testing data.
    """

    label_counts = pd.Series(labels).value_counts()

    # Stratification requires at least two samples in every class.
    can_stratify = label_counts.min() >= 2

    if can_stratify:
        stratify_values = labels
        print()
        print("Using a stratified train-test split.")
    else:
        stratify_values = None
        print()
        print(
            "Warning: At least one class contains fewer than two "
            "recordings. Stratification will not be used."
        )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test
    ) = train_test_split(
        features,
        labels,
        metadata,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify_values
    )

    print()
    print(f"Training recordings: {len(X_train)}")
    print(f"Testing recordings: {len(X_test)}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test
    )


def scale_features(X_train, X_test):
    """
    Standardize the numerical features.

    Standardization changes each feature so that the training data has
    approximately:

        Mean = 0
        Standard deviation = 1

    The scaler is fitted only on the training data to prevent data leakage.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.

    X_test : pandas.DataFrame
        Testing features.

    Returns
    -------
    X_train_scaled : pandas.DataFrame
        Standardized training features.

    X_test_scaled : pandas.DataFrame
        Standardized testing features.

    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler.
    """

    scaler = StandardScaler()

    # Learn scaling values only from the training set.
    X_train_scaled_array = scaler.fit_transform(X_train)

    # Apply the same transformation to the testing set.
    X_test_scaled_array = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled_array,
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled_array,
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler


def save_processed_data(
    output_directory,
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    metadata_train,
    metadata_test,
    label_encoder,
    scaler
):
    """
    Save processed datasets and preprocessing objects.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save unscaled features.
    X_train.to_csv(
        output_directory / "X_train_unscaled.csv",
        index=False
    )

    X_test.to_csv(
        output_directory / "X_test_unscaled.csv",
        index=False
    )

    # Save scaled features.
    X_train_scaled.to_csv(
        output_directory / "X_train.csv",
        index=False
    )

    X_test_scaled.to_csv(
        output_directory / "X_test.csv",
        index=False
    )

    # Save target labels.
    pd.DataFrame({"label": y_train}).to_csv(
        output_directory / "y_train.csv",
        index=False
    )

    pd.DataFrame({"label": y_test}).to_csv(
        output_directory / "y_test.csv",
        index=False
    )

    # Save identifying information separately.
    metadata_train.reset_index(drop=True).to_csv(
        output_directory / "train_metadata.csv",
        index=False
    )

    metadata_test.reset_index(drop=True).to_csv(
        output_directory / "test_metadata.csv",
        index=False
    )

    # Save preprocessing objects for future predictions.
    joblib.dump(
        scaler,
        output_directory / "feature_scaler.pkl"
    )

    joblib.dump(
        label_encoder,
        output_directory / "label_encoder.pkl"
    )

    # Save the feature-column order.
    with open(
        output_directory / "feature_columns.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            list(X_train.columns),
            file,
            indent=4
        )

    # Save label mappings in a human-readable JSON file.
    label_mapping = {
        int(index): class_name
        for index, class_name in enumerate(label_encoder.classes_)
    }

    with open(
        output_directory / "label_mapping.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            label_mapping,
            file,
            indent=4
        )


def print_summary(
    output_directory,
    X_train,
    X_test,
    label_encoder
):
    """
    Print a final preprocessing summary.
    """

    print()
    print("=" * 70)
    print("DATA PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")
    print(f"Feature columns: {X_train.shape[1]}")
    print(f"Chord classes: {len(label_encoder.classes_)}")
    print(f"Files saved in: {output_directory}")


def preprocess_dataset(
    input_file,
    output_directory,
    test_size
):
    """
    Run the complete preprocessing pipeline.
    """

    dataframe = load_feature_data(input_file)

    inspect_dataset(dataframe)

    cleaned_dataframe = clean_dataset(dataframe)

    X, y, metadata = separate_features_and_labels(
        cleaned_dataframe
    )

    encoded_labels, label_encoder = encode_labels(y)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test
    ) = split_dataset(
        X,
        encoded_labels,
        metadata,
        test_size
    )

    (
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = scale_features(
        X_train,
        X_test
    )

    save_processed_data(
        output_directory=output_directory,
        X_train=X_train,
        X_test=X_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        metadata_train=metadata_train,
        metadata_test=metadata_test,
        label_encoder=label_encoder,
        scaler=scaler
    )

    print_summary(
        output_directory,
        X_train,
        X_test,
        label_encoder
    )


def parse_arguments():
    """
    Read command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Prepare extracted guitar-chord audio features "
            "for machine learning."
        )
    )

    parser.add_argument(
        "--input",
        default="results/audio_features.csv",
        help=(
            "Path to the extracted feature CSV. "
            "Default: results/audio_features.csv"
        )
    )

    parser.add_argument(
        "--output",
        default="processed_data",
        help=(
            "Directory where processed files will be saved. "
            "Default: processed_data"
        )
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=(
            "Fraction of the dataset used for testing. "
            "Default: 0.20"
        )
    )

    return parser.parse_args()


def main():
    """
    Main program entry point.
    """

    arguments = parse_arguments()

    if not 0 < arguments.test_size < 1:
        print(
            "Error: --test-size must be greater than 0 "
            "and less than 1."
        )
        sys.exit(1)

    input_file = Path(
        arguments.input
    ).expanduser().resolve()

    output_directory = Path(
        arguments.output
    ).expanduser().resolve()

    try:
        preprocess_dataset(
            input_file=input_file,
            output_directory=output_directory,
            test_size=arguments.test_size
        )

    except Exception as error:
        print(f"\nData preprocessing failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()