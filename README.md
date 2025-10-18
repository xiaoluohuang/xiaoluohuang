# Kyber Electromagnetic Signal Classifier

This repository contains a CNN-based electromagnetic signal classifier for analyzing Kyber post-quantum cryptographic implementations. The classifier can distinguish between known plaintext (M0) and random plaintext (M1) traces to detect potential side-channel vulnerabilities.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete analysis pipeline
python kyber_signal_classifier.py
```

## Features

- ✅ CNN-based binary classifier with batch normalization and SELU activation
- ✅ 80/20 train-test split with proper data preprocessing  
- ✅ Negative Log Likelihood (NLL) evaluation for enhanced detection
- ✅ Comprehensive visualizations (signals, PCA, frequency analysis, ROC curves)
- ✅ Model checkpointing and early stopping
- ✅ Full documentation and example usage scripts

## Documentation

See [CLASSIFIER_README.md](CLASSIFIER_README.md) for comprehensive documentation including:
- Installation instructions
- Usage examples
- Model architecture details
- Evaluation metrics explanation
- Troubleshooting guide

## Project Files

- `kyber_signal_classifier.py` - Main classifier implementation
- `test_classifier.py` - Test suite
- `example_usage.py` - Usage examples  
- `requirements.txt` - Python dependencies
- `200_GPT1_MV_4_class_m0.csv` - M0 (known plaintext) dataset
- `200_GPT1_MV_4_class_m1.csv` - M1 (random plaintext) dataset

## About

This project implements side-channel analysis for Kyber cryptographic implementations using deep learning to detect electromagnetic signal patterns that could indicate security vulnerabilities.
