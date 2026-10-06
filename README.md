# Guitar Chord Recognition

A machine learning project for classifying guitar chords from audio-derived features.

The model is designed to distinguish between:

- 12 major chords
- 12 minor chords
- Noise

The project compares multiple classification algorithms to determine which approach performs best for chord recognition.

## Overview

Recognizing musical chords from audio is a classification problem that combines signal processing and machine learning.

This project builds a pipeline that takes extracted audio features and trains machine learning models to predict the corresponding guitar chord.

The goal is to evaluate multiple algorithms and determine which models are most effective at distinguishing between similar chord classes.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Machine Learning
- Audio Feature Analysis

## Models Evaluated

The project evaluates several machine learning algorithms, including:

- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)

The models are compared using classification performance to determine which approach provides the most reliable chord recognition.

## Project Workflow

1. Collect or load audio-derived chord features.
2. Prepare and clean the dataset.
3. Separate features from target chord labels.
4. Split the dataset into training and testing sets.
5. Train multiple classification models.
6. Evaluate each model.
7. Compare model performance.
8. Identify the strongest approach for chord recognition.

## Classification Targets

The dataset includes 25 possible classes:

### Major Chords

- A
- A#
- B
- C
- C#
- D
- D#
- E
- F
- F#
- G
- G#

### Minor Chords

- Am
- A#m
- Bm
- Cm
- C#m
- Dm
- D#m
- Em
- Fm
- F#m
- Gm
- G#m

### Additional Class

- Noise

## Machine Learning Pipeline

The general modeling workflow is:

```text
Audio
  ↓
Feature Extraction
  ↓
Data Preparation
  ↓
Train/Test Split
  ↓
Machine Learning Models
  ↓
Model Evaluation
  ↓
Chord Prediction
