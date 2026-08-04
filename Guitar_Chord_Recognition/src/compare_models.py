"""
Compare baseline classification models using LazyPredict.

The script loads the processed training and testing datasets,
evaluates several machine-learning classifiers, and saves the
results as a CSV file.
"""

import argparse
from pathlib import Path

import pandas as pd
from lazypredict.Supervised import LazyClassifier


def load_processed_data(processed_folder: Path):
    """
    Load the four processed datasets created by preprocess_data.py.
    """
    required_files = {
        "X_train": processed_folder / "X_train.csv",
        "X_test": processed_folder / "X_test.csv",
        "y_train": processed_folder / "y_train.csv",
        "y_test": processed_folder / "y_test.csv",
    }

    for name, file_path in required_files.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found for {name}: {file_path}"
            )

    X_train = pd.read_csv(required_files["X_train"])
    X_test = pd.read_csv(required_files["X_test"])
    y_train_df = pd.read_csv(required_files["y_train"])
    y_test_df = pd.read_csv(required_files["y_test"])

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

    return X_train, X_test, y_train, y_test


def main(processed_folder: Path, output_file: Path) -> None:
    print("=" * 70)
    print("LAZYPREDICT MODEL COMPARISON")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_processed_data(
        processed_folder
    )

    print(f"Training recordings: {len(X_train)}")
    print(f"Testing recordings: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes in training data: {y_train.nunique()}")
    print(f"Classes in testing data: {y_test.nunique()}")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train contain different numbers of rows."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test contain different numbers of rows."
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    print("\nTraining baseline classifiers...")
    print("This may take several minutes.\n")

    classifier = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42,
    )

    models, _ = classifier.fit(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Convert the model names from the index into a regular CSV column.
    results = models.reset_index()

    first_column = results.columns[0]
    results = results.rename(columns={first_column: "Model"})

    # Rank models primarily by balanced accuracy and then by F1 score.
    sorting_columns = [
        column
        for column in ["Balanced Accuracy", "F1 Score", "Accuracy"]
        if column in results.columns
    ]

    if sorting_columns:
        results = results.sort_values(
            by=sorting_columns,
            ascending=False,
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 70)

    display_columns = [
        column
        for column in [
            "Model",
            "Accuracy",
            "Balanced Accuracy",
            "F1 Score",
            "Time Taken",
        ]
        if column in results.columns
    ]

    print("\nTop 10 models:")
    print(
        results[display_columns]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nFull results saved to: {output_file.resolve()}")

    if not results.empty:
        print(f"Highest-ranked model: {results.iloc[0]['Model']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline chord-classification models "
            "using LazyPredict."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("processed_data"),
        help="Folder containing the processed training and testing files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lazypredict_results.csv"),
        help="CSV file where model-comparison results will be saved.",
    )

    args = parser.parse_args()

    main(
        processed_folder=args.input,
        output_file=args.output,
    )