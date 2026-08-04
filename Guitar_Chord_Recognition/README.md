# Guitar Chord Recognition Using Traditional Machine Learning

## Overview

This project implements a machine learning system for recognizing isolated guitar chords from audio recordings. The system extracts numerical audio features from WAV files, compares multiple supervised machine learning algorithms, and uses the best-performing model to predict guitar chords.

The final classifier achieved **96.15% accuracy** on the held-out testing dataset and was further evaluated using independently recorded smartphone audio to assess its real-world performance.

---

## Features

- Automatic feature extraction from WAV audio files
- Extraction of 101 numerical audio features using Librosa
- Comparison of 23 machine learning classifiers using LazyPredict
- Automatic selection of the highest-performing classifier
- Prediction of 25 classes:
  - 12 Major chords
  - 12 Minor chords
  - Noise
- Performance evaluation using:
  - Accuracy
  - Balanced Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

---

## Project Structure

```
project/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── models/
│   ├── calibrated_classifier.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_columns.pkl
│
├── extracted_features/
│
├── prediction/
│
├── results/
│
├── main.py
├── feature_extraction.py
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10 or newer

Required Python libraries:

- librosa
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib
- LazyPredict

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Dataset

This project uses the **Isolated Guitar Chords** dataset available on Hugging Face.

The dataset contains:

- 780 WAV recordings
- 25 classes
  - 12 major chords
  - 12 minor chords
  - Noise

The recordings are divided into training and testing sets before feature extraction and model training.

---

## Running the Project

### 1. Extract Features

```bash
python feature_extraction.py
```

This script extracts 101 numerical audio features from every recording.

---

### 2. Train the Model

```bash
python train_model.py
```

This script:

- preprocesses the dataset
- compares multiple machine learning models
- selects the best classifier
- saves the trained model

---

### 3. Predict New Chords

```bash
python predict.py
```

Provide a WAV recording of an isolated guitar chord to receive a predicted chord label.

---

## Results

Final testing performance:

| Metric | Score |
|---------|-------|
| Accuracy | 96.15% |
| Balanced Accuracy | 96.29% |
| Macro Precision | 96.76% |
| Macro Recall | 96.29% |
| Macro F1 | 96.26% |
| Weighted F1 | 96.17% |

The model was also evaluated using twenty independently recorded smartphone recordings and achieved **45.00% accuracy** under real-world recording conditions.

---

## Methodology

The project follows the following workflow:

1. Dataset Preparation
2. Feature Extraction
3. Data Preprocessing
4. Model Comparison
5. Final Model Training
6. Performance Evaluation
7. Real-World Testing

---

## Future Improvements

Possible future enhancements include:

- Support for additional chord types
- Larger and more diverse datasets
- Audio data augmentation
- Deep learning approaches (CNNs, RNNs, Transformers)
- Real-time chord recognition
- Mobile application deployment

---

## Author

Richard Tairouz

Florida International University

CAI 4105 – Machine Learning

Summer 2026