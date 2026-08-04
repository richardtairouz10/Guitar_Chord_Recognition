"""
Audio Feature Extraction for Guitar Chord Recognition

This program:
1. Searches recursively for WAV audio files.
2. Loads and preprocesses each audio recording.
3. Extracts numerical audio features using Librosa.
4. Uses the parent folder name as the chord label.
5. Saves the extracted features into a CSV file.

Example:
    python3 src/extract_features.py \
        --input dataset_download \
        --output results/audio_features.csv
"""

import argparse
from pathlib import Path
import sys

import librosa
import numpy as np
import pandas as pd


# All audio files will be converted to this sample rate.
TARGET_SAMPLE_RATE = 22050

# Every recording will be normalized to this duration.
FIXED_DURATION_SECONDS = 5.0

# Number of MFCC coefficients to extract.
NUMBER_OF_MFCCS = 20


def load_and_preprocess_audio(audio_path):
    """
    Load and preprocess one audio file.

    Processing performed:
    - Converts stereo audio to mono.
    - Resamples the recording to 22,050 Hz.
    - Removes silence from the beginning and end.
    - Normalizes the amplitude.
    - Pads or trims the recording to exactly five seconds.

    Parameters
    ----------
    audio_path : pathlib.Path
        Path to the WAV audio file.

    Returns
    -------
    audio : numpy.ndarray
        Preprocessed audio signal.

    sample_rate : int
        Audio sample rate.
    """

    audio, sample_rate = librosa.load(
        audio_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True
    )

    # Verify that the audio file contains samples.
    if audio.size == 0:
        raise ValueError("The audio file is empty.")

    # Remove silence from the beginning and end.
    trimmed_audio, _ = librosa.effects.trim(
        audio,
        top_db=30
    )

    # If silence removal removed everything, use the original audio.
    if trimmed_audio.size > 0:
        audio = trimmed_audio

    # Normalize the amplitude to reduce volume differences.
    maximum_amplitude = np.max(np.abs(audio))

    if maximum_amplitude > 0:
        audio = audio / maximum_amplitude

    # Calculate the number of samples required for five seconds.
    target_length = int(FIXED_DURATION_SECONDS * sample_rate)

    # Pad shorter audio recordings with zeros.
    if len(audio) < target_length:
        audio = np.pad(
            audio,
            pad_width=(0, target_length - len(audio)),
            mode="constant"
        )

    # Trim longer audio recordings.
    elif len(audio) > target_length:
        audio = audio[:target_length]

    return audio, sample_rate


def add_summary_statistics(feature_dictionary, feature_name, values):
    """
    Add the mean and standard deviation of a feature to a dictionary.

    Many Librosa features produce values over several audio frames.
    A traditional machine-learning model needs a fixed number of values
    per recording, so the mean and standard deviation are calculated.

    Parameters
    ----------
    feature_dictionary : dict
        Dictionary containing the extracted feature values.

    feature_name : str
        Name used for the feature columns.

    values : numpy.ndarray
        Feature values calculated by Librosa.
    """

    feature_dictionary[f"{feature_name}_mean"] = float(np.mean(values))
    feature_dictionary[f"{feature_name}_std"] = float(np.std(values))


def extract_audio_features(audio, sample_rate):
    """
    Extract numerical features from a preprocessed audio signal.

    Features extracted:
    - MFCCs
    - Chroma
    - Spectral centroid
    - Spectral bandwidth
    - Spectral rolloff
    - Spectral contrast
    - Zero-crossing rate
    - RMS energy
    - Tonnetz
    - Tempo

    Parameters
    ----------
    audio : numpy.ndarray
        Preprocessed audio signal.

    sample_rate : int
        Audio sample rate.

    Returns
    -------
    features : dict
        Dictionary containing all numerical audio features.
    """

    features = {}

    # ---------------------------------------------------------
    # MFCCs
    # ---------------------------------------------------------
    # MFCCs describe the overall tone and timbre of the sound.
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=NUMBER_OF_MFCCS
    )

    for index in range(NUMBER_OF_MFCCS):
        features[f"mfcc_{index + 1}_mean"] = float(
            np.mean(mfccs[index])
        )
        features[f"mfcc_{index + 1}_std"] = float(
            np.std(mfccs[index])
        )

    # ---------------------------------------------------------
    # Chroma
    # ---------------------------------------------------------
    # Chroma represents the energy of the 12 musical pitch classes:
    # C, C#, D, D#, E, F, F#, G, G#, A, A#, and B.
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sample_rate
    )

    pitch_names = [
        "C", "C_sharp", "D", "D_sharp", "E", "F",
        "F_sharp", "G", "G_sharp", "A", "A_sharp", "B"
    ]

    for index, pitch_name in enumerate(pitch_names):
        features[f"chroma_{pitch_name}_mean"] = float(
            np.mean(chroma[index])
        )
        features[f"chroma_{pitch_name}_std"] = float(
            np.std(chroma[index])
        )

    # ---------------------------------------------------------
    # Spectral centroid
    # ---------------------------------------------------------
    # Measures the perceived brightness of the sound.
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    add_summary_statistics(
        features,
        "spectral_centroid",
        spectral_centroid
    )

    # ---------------------------------------------------------
    # Spectral bandwidth
    # ---------------------------------------------------------
    # Measures how widely the audio frequencies are distributed.
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate
    )

    add_summary_statistics(
        features,
        "spectral_bandwidth",
        spectral_bandwidth
    )

    # ---------------------------------------------------------
    # Spectral rolloff
    # ---------------------------------------------------------
    # Identifies the frequency below which most of the sound energy lies.
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate,
        roll_percent=0.85
    )

    add_summary_statistics(
        features,
        "spectral_rolloff",
        spectral_rolloff
    )

    # ---------------------------------------------------------
    # Spectral contrast
    # ---------------------------------------------------------
    # Measures the difference between frequency peaks and valleys.
    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sample_rate
    )

    for index in range(spectral_contrast.shape[0]):
        features[f"spectral_contrast_{index + 1}_mean"] = float(
            np.mean(spectral_contrast[index])
        )
        features[f"spectral_contrast_{index + 1}_std"] = float(
            np.std(spectral_contrast[index])
        )

    # ---------------------------------------------------------
    # Zero-crossing rate
    # ---------------------------------------------------------
    # Measures how frequently the audio signal changes direction.
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)

    add_summary_statistics(
        features,
        "zero_crossing_rate",
        zero_crossing_rate
    )

    # ---------------------------------------------------------
    # RMS energy
    # ---------------------------------------------------------
    # Measures the loudness or energy of the recording.
    rms_energy = librosa.feature.rms(y=audio)

    add_summary_statistics(
        features,
        "rms_energy",
        rms_energy
    )

    # ---------------------------------------------------------
    # Tonnetz
    # ---------------------------------------------------------
    # Describes harmonic relationships between musical pitches.
    harmonic_audio = librosa.effects.harmonic(audio)

    tonnetz = librosa.feature.tonnetz(
        y=harmonic_audio,
        sr=sample_rate
    )

    for index in range(tonnetz.shape[0]):
        features[f"tonnetz_{index + 1}_mean"] = float(
            np.mean(tonnetz[index])
        )
        features[f"tonnetz_{index + 1}_std"] = float(
            np.std(tonnetz[index])
        )

    # ---------------------------------------------------------
    # Tempo
    # ---------------------------------------------------------
    # Estimates the tempo of the audio.
    tempo, _ = librosa.beat.beat_track(
        y=audio,
        sr=sample_rate
    )

    # Different Librosa versions may return tempo as a scalar or array.
    features["tempo"] = float(np.asarray(tempo).flatten()[0])

    return features


def determine_label(audio_path, input_directory):
    """
    Determine the class label for an audio recording.

    The immediate parent folder is normally used as the label.

    Example:
        train/A_major/sample_01.wav

    Label:
        A_major

    Parameters
    ----------
    audio_path : pathlib.Path
        Path to the WAV file.

    input_directory : pathlib.Path
        Main dataset directory.

    Returns
    -------
    label : str
        Chord class label.
    """

    parent_name = audio_path.parent.name

    # Avoid treating general organizational folders as labels.
    ignored_folders = {
        "train",
        "training",
        "test",
        "testing",
        "validation",
        "val",
        "audio",
        "audios",
        "wav",
        "wavs"
    }

    if parent_name.lower() not in ignored_folders:
        return parent_name

    # If the immediate parent is an organizational folder,
    # use the next folder above it.
    try:
        relative_path = audio_path.relative_to(input_directory)
        folder_parts = relative_path.parts[:-1]

        for folder_name in reversed(folder_parts):
            if folder_name.lower() not in ignored_folders:
                return folder_name
    except ValueError:
        pass

    return "unknown"


def find_audio_files(input_directory):
    """
    Recursively find all WAV files inside the dataset directory.

    Both lowercase and uppercase extensions are supported.
    """

    lowercase_files = list(input_directory.rglob("*.wav"))
    uppercase_files = list(input_directory.rglob("*.WAV"))

    all_files = lowercase_files + uppercase_files

    # Remove duplicates and sort the paths.
    return sorted(set(all_files))


def process_dataset(input_directory, output_file):
    """
    Process every WAV file and save the extracted features to CSV.

    Parameters
    ----------
    input_directory : pathlib.Path
        Root directory containing the audio dataset.

    output_file : pathlib.Path
        Location where the features CSV will be saved.
    """

    audio_files = find_audio_files(input_directory)

    if not audio_files:
        raise FileNotFoundError(
            f"No WAV files were found inside: {input_directory}"
        )

    print("=" * 70)
    print("GUITAR CHORD AUDIO FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Input directory: {input_directory}")
    print(f"Audio files found: {len(audio_files)}")
    print(f"Output file: {output_file}")
    print("=" * 70)

    dataset_rows = []
    failed_files = []

    for file_number, audio_path in enumerate(audio_files, start=1):
        label = determine_label(audio_path, input_directory)

        print(
            f"[{file_number}/{len(audio_files)}] "
            f"Processing: {audio_path.name} | Label: {label}"
        )

        try:
            audio, sample_rate = load_and_preprocess_audio(audio_path)

            feature_values = extract_audio_features(
                audio,
                sample_rate
            )

            # Add identifying information.
            feature_values["file_name"] = audio_path.name
            feature_values["file_path"] = str(audio_path)
            feature_values["label"] = label

            dataset_rows.append(feature_values)

        except Exception as error:
            failed_files.append(
                {
                    "file_path": str(audio_path),
                    "error": str(error)
                }
            )

            print(f"    ERROR: {error}")

    if not dataset_rows:
        raise RuntimeError(
            "Feature extraction failed for every audio file."
        )

    # Convert all extracted rows into a Pandas DataFrame.
    features_dataframe = pd.DataFrame(dataset_rows)

    # Move identifying columns to the beginning.
    identifying_columns = [
        "file_name",
        "file_path",
        "label"
    ]

    numerical_columns = [
        column
        for column in features_dataframe.columns
        if column not in identifying_columns
    ]

    features_dataframe = features_dataframe[
        identifying_columns + numerical_columns
    ]

    # Create the output directory if it does not exist.
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the extracted features.
    features_dataframe.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Successfully processed: {len(dataset_rows)} files")
    print(f"Failed files: {len(failed_files)}")
    print(f"Number of columns: {len(features_dataframe.columns)}")
    print(f"Features saved to: {output_file}")

    # Display the number of recordings per class.
    print()
    print("Recordings per class:")
    print(features_dataframe["label"].value_counts().sort_index())

    # Save errors separately if any files failed.
    if failed_files:
        failed_output_file = (
            output_file.parent / "failed_audio_files.csv"
        )

        pd.DataFrame(failed_files).to_csv(
            failed_output_file,
            index=False
        )

        print()
        print(
            "Information about failed files was saved to: "
            f"{failed_output_file}"
        )


def parse_arguments():
    """
    Read command-line arguments supplied by the user.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Extract machine-learning features from guitar chord "
            "WAV recordings."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Root directory containing the WAV dataset."
    )

    parser.add_argument(
        "--output",
        default="results/audio_features.csv",
        help=(
            "CSV file where the extracted features will be saved. "
            "Default: results/audio_features.csv"
        )
    )

    return parser.parse_args()


def main():
    """
    Main program entry point.
    """

    arguments = parse_arguments()

    input_directory = Path(arguments.input).expanduser().resolve()
    output_file = Path(arguments.output).expanduser().resolve()

    if not input_directory.exists():
        print(
            f"Error: The input directory does not exist:\n"
            f"{input_directory}"
        )
        sys.exit(1)

    if not input_directory.is_dir():
        print(
            f"Error: The input path is not a directory:\n"
            f"{input_directory}"
        )
        sys.exit(1)

    try:
        process_dataset(
            input_directory=input_directory,
            output_file=output_file
        )

    except Exception as error:
        print(f"\nFeature extraction failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()