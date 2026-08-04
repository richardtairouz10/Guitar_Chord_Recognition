"""
Dataset Explorer for the Guitar Chord Recognition Project

This script examines the guitar chord audio dataset before feature
extraction and model training

It performs the following tasks:

1. Finds the training and testing folders.
2. Detects all chord classes.
3. Counts the WAV files in each class.
4. Reads audio metadata such as:
   - sample rate
   - duration
   - number of channels
5. Detects unreadable or corrupted files.
6. Saves a dataset summary as a CSV file.
"""

from pathlib import Path
import csv
import wave
import contextlib
import statistics


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

# __file__ represents the location of this Python script.
# .resolve() creates the complete absolute path.
SCRIPT_PATH = Path(__file__).resolve()

# The script is inside:
# Guitar_Chord_Recognition/src/dataset_explorer.py
#
# SCRIPT_PATH.parent is the src folder.
# SCRIPT_PATH.parent.parent is the main project folder.
PROJECT_ROOT = SCRIPT_PATH.parent.parent

# Main folders used by the project.
DATA_FOLDER = PROJECT_ROOT / "dataset_download" / "data"
TRAIN_FOLDER = DATA_FOLDER / "train"
TEST_FOLDER = DATA_FOLDER / "test"
RESULTS_FOLDER = PROJECT_ROOT / "results"

# CSV file where the final dataset summary will be stored.
SUMMARY_FILE = RESULTS_FOLDER / "dataset_summary.csv"


# ---------------------------------------------------------
# AUDIO INSPECTION FUNCTION
# ---------------------------------------------------------

def inspect_wav_file(file_path: Path) -> dict:
    """
    Read basic information from one WAV file.

    Parameters
    ----------
    file_path : Path
        Path to the WAV audio file.

    Returns
    -------
    dict
        A dictionary containing:
        - sample rate
        - number of channels
        - number of audio frames
        - duration in seconds
    """

    # contextlib.closing makes sure the audio file is closed
    # properly after it is inspected.
    with contextlib.closing(wave.open(str(file_path), "rb")) as audio_file:

        # Number of channels:
        # 1 = mono
        # 2 = stereo
        number_of_channels = audio_file.getnchannels()

        # Sample rate:
        # Number of audio samples recorded each second.
        sample_rate = audio_file.getframerate()

        # Total number of audio frames in the file.
        number_of_frames = audio_file.getnframes()

        # Duration is calculated by dividing the number of frames
        # by the number of frames recorded per second.
        duration = number_of_frames / float(sample_rate)

        return {
            "sample_rate": sample_rate,
            "channels": number_of_channels,
            "frames": number_of_frames,
            "duration": duration,
        }


# ---------------------------------------------------------
# DATASET SPLIT EXPLORATION
# ---------------------------------------------------------

def explore_split(split_name: str, split_folder: Path) -> tuple[list[dict], list[dict]]:
    """
    Explore one dataset split, such as train or test.

    Parameters
    ----------
    split_name : str
        Name of the dataset split, such as "train" or "test".

    split_folder : Path
        Path to the split folder.

    Returns
    -------
    tuple
        The first list contains summary information for each class.
        The second list contains information about corrupted files.
    """

    print()
    print("=" * 70)
    print(f"EXPLORING {split_name.upper()} DATASET")
    print("=" * 70)

    # This list will store one summary row for each chord class.
    class_summaries = []

    # This list will store files that cannot be read.
    corrupted_files = []

    # Make sure the folder exists before continuing.
    if not split_folder.exists():
        print(f"Folder not found: {split_folder}")
        return class_summaries, corrupted_files

    # Find all immediate subfolders.
    # Each subfolder should represent one chord class.
    class_folders = sorted(
        [
            folder
            for folder in split_folder.iterdir()
            if folder.is_dir()
        ],
        key=lambda folder: folder.name.lower(),
    )

    if not class_folders:
        print(f"No class folders were found inside: {split_folder}")
        return class_summaries, corrupted_files

    print(f"Number of classes found: {len(class_folders)}")
    print()

    # Process one chord folder at a time.
    for class_folder in class_folders:

        class_name = class_folder.name

        # Find WAV files using both lowercase and uppercase extensions.
        wav_files = sorted(
            list(class_folder.glob("*.wav"))
            + list(class_folder.glob("*.WAV"))
        )

        durations = []
        sample_rates = []
        channel_counts = []
        readable_file_count = 0

        # Inspect every WAV file inside the current chord class.
        for wav_file in wav_files:
            try:
                audio_info = inspect_wav_file(wav_file)

                durations.append(audio_info["duration"])
                sample_rates.append(audio_info["sample_rate"])
                channel_counts.append(audio_info["channels"])

                readable_file_count += 1

            except (
                wave.Error,
                EOFError,
                OSError,
                ValueError,
                ZeroDivisionError,
            ) as error:

                corrupted_files.append(
                    {
                        "split": split_name,
                        "class": class_name,
                        "file": str(wav_file),
                        "error": str(error),
                    }
                )

        # Calculate statistics only when at least one file was readable.
        if durations:
            minimum_duration = min(durations)
            maximum_duration = max(durations)
            average_duration = statistics.mean(durations)
        else:
            minimum_duration = 0
            maximum_duration = 0
            average_duration = 0

        # Create readable lists of unique sample rates and channel counts.
        unique_sample_rates = sorted(set(sample_rates))
        unique_channel_counts = sorted(set(channel_counts))

        summary = {
            "split": split_name,
            "class": class_name,
            "total_wav_files": len(wav_files),
            "readable_files": readable_file_count,
            "corrupted_files": len(wav_files) - readable_file_count,
            "minimum_duration_seconds": round(minimum_duration, 3),
            "maximum_duration_seconds": round(maximum_duration, 3),
            "average_duration_seconds": round(average_duration, 3),
            "sample_rates": ", ".join(
                str(rate) for rate in unique_sample_rates
            ),
            "channel_counts": ", ".join(
                str(count) for count in unique_channel_counts
            ),
        }

        class_summaries.append(summary)

        # Print one line for the current class.
        print(
            f"{class_name:<12} "
            f"Files: {len(wav_files):<5} "
            f"Readable: {readable_file_count:<5} "
            f"Average duration: {average_duration:.2f} seconds"
        )

    return class_summaries, corrupted_files


# ---------------------------------------------------------
# CSV SAVING FUNCTION
# ---------------------------------------------------------

def save_summary_to_csv(summary_rows: list[dict], output_file: Path) -> None:
    """
    Save dataset summary information to a CSV file.

    Parameters
    ----------
    summary_rows : list[dict]
        Dataset summary rows.

    output_file : Path
        Location where the CSV file will be saved.
    """

    if not summary_rows:
        print("No summary data was available to save.")
        return

    # Create the results folder if it does not already exist.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    column_names = [
        "split",
        "class",
        "total_wav_files",
        "readable_files",
        "corrupted_files",
        "minimum_duration_seconds",
        "maximum_duration_seconds",
        "average_duration_seconds",
        "sample_rates",
        "channel_counts",
    ]

    with output_file.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=column_names,
        )

        writer.writeheader()
        writer.writerows(summary_rows)


# ---------------------------------------------------------
# FINAL DATASET SUMMARY
# ---------------------------------------------------------

def print_overall_summary(
    summary_rows: list[dict],
    corrupted_files: list[dict],
) -> None:
    """
    Print an overall summary for the entire dataset.
    """

    print()
    print("=" * 70)
    print("OVERALL DATASET SUMMARY")
    print("=" * 70)

    total_classes = len(summary_rows)

    total_files = sum(
        row["total_wav_files"]
        for row in summary_rows
    )

    total_readable = sum(
        row["readable_files"]
        for row in summary_rows
    )

    total_corrupted = len(corrupted_files)

    print(f"Total class folders inspected: {total_classes}")
    print(f"Total WAV files found: {total_files}")
    print(f"Readable WAV files: {total_readable}")
    print(f"Corrupted or unreadable WAV files: {total_corrupted}")

    # Separate training and testing counts.
    train_files = sum(
        row["total_wav_files"]
        for row in summary_rows
        if row["split"] == "train"
    )

    test_files = sum(
        row["total_wav_files"]
        for row in summary_rows
        if row["split"] == "test"
    )

    print(f"Training files: {train_files}")
    print(f"Testing files: {test_files}")

    if total_corrupted == 0:
        print("No corrupted WAV files were detected.")
    else:
        print()
        print("Corrupted or unreadable files:")

        for file_information in corrupted_files:
            print(
                f"- Split: {file_information['split']}, "
                f"Class: {file_information['class']}, "
                f"File: {file_information['file']}, "
                f"Error: {file_information['error']}"
            )


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main() -> None:
    """
    Main function that controls the complete dataset exploration process.
    """

    print("=" * 70)
    print("GUITAR CHORD DATASET EXPLORER")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Training folder: {TRAIN_FOLDER}")
    print(f"Testing folder: {TEST_FOLDER}")

    # Explore the training dataset.
    train_summary, train_corrupted = explore_split(
        split_name="train",
        split_folder=TRAIN_FOLDER,
    )

    # Explore the testing dataset.
    test_summary, test_corrupted = explore_split(
        split_name="test",
        split_folder=TEST_FOLDER,
    )

    # Combine the results from both splits.
    complete_summary = train_summary + test_summary
    complete_corrupted_list = train_corrupted + test_corrupted

    # Print the final dataset totals.
    print_overall_summary(
        summary_rows=complete_summary,
        corrupted_files=complete_corrupted_list,
    )

    # Save the summary as a CSV file.
    save_summary_to_csv(
        summary_rows=complete_summary,
        output_file=SUMMARY_FILE,
    )

    if complete_summary:
        print()
        print(f"Dataset summary saved to:")
        print(SUMMARY_FILE)

    print()
    print("Dataset exploration complete.")


# This ensures that main() runs only when this file is executed directly.
if __name__ == "__main__":
    main()