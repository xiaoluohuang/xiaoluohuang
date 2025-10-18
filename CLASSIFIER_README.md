# Electromagnetic Signal Classifier for Kyber Cryptographic Analysis

This project implements a CNN-based binary classifier to distinguish between M0 (known plaintext) and M1 (random plaintext) electromagnetic traces in Kyber cryptographic implementations.

## Overview

The classifier analyzes electromagnetic signals captured during Kyber cryptographic operations to detect side-channel vulnerabilities. It uses deep learning to differentiate between traces generated with known plaintexts versus random plaintexts.

## Features

- **Data Processing**: Handles CSV datasets with 1000 features × 200 samples
- **80/20 Train-Test Split**: 160 training + 40 test samples per class
- **CNN Architecture**: Multi-layer convolutional network with batch normalization and SELU activation
- **Training Features**:
  - Binary crossentropy loss
  - Adam optimizer (learning rate: 0.001)
  - Model checkpointing for best validation model
  - Early stopping based on validation accuracy
  - Data normalization (divide by 256)
- **Evaluation Methods**:
  - Standard accuracy metrics
  - Negative Log Likelihood (NLL) comparison
  - Confusion matrix and classification reports
  - ROC curve analysis
- **Comprehensive Visualizations**:
  - Signal trace comparisons
  - PCA analysis
  - Frequency domain analysis
  - Training history plots
  - Evaluation results and confidence analysis

## Installation

### Requirements

Python 3.8 or higher is required. Install the dependencies:

```bash
pip install -r requirements.txt
```

The required packages are:
- numpy >= 1.24.0
- pandas >= 2.0.0
- tensorflow >= 2.13.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## Dataset Structure

The classifier expects two CSV files:
- `200_GPT1_MV_4_class_m0.csv`: M0 (known plaintext) traces
- `200_GPT1_MV_4_class_m1.csv`: M1 (random plaintext) traces

Each file should have:
- Rows: Features/time points (1000 rows)
- Columns: Traces/samples (200 columns)
- Format: CSV with numeric values

## Usage

### Basic Usage

Run the complete classification pipeline:

```bash
python kyber_signal_classifier.py
```

This will:
1. Load and preprocess the datasets
2. Generate data visualizations
3. Build the CNN model
4. Train the model with validation
5. Evaluate on test set
6. Perform NLL analysis
7. Save the trained model and all plots

### Using the Classifier as a Module

```python
from kyber_signal_classifier import KyberSignalClassifier

# Initialize classifier
classifier = KyberSignalClassifier(
    m0_path='200_GPT1_MV_4_class_m0.csv',
    m1_path='200_GPT1_MV_4_class_m1.csv'
)

# Load and preprocess data
classifier.load_and_preprocess_data()

# Visualize data
classifier.visualize_data()

# Build and train model
classifier.build_model()
classifier.train(epochs=100, batch_size=32, validation_split=0.2)

# Evaluate
eval_results = classifier.evaluate()
nll_results = classifier.nll_evaluation()

# Plot results
classifier.plot_training_history()
classifier.plot_evaluation_results(eval_results, nll_results)

# Save model
classifier.save_model('my_model.h5')
```

### Loading a Trained Model

```python
from kyber_signal_classifier import KyberSignalClassifier

classifier = KyberSignalClassifier(m0_path='...', m1_path='...')
classifier.load_and_preprocess_data()
classifier.load_model('kyber_classifier_model.h5')

# Evaluate loaded model
eval_results = classifier.evaluate()
```

## Model Architecture

The CNN architecture consists of:

1. **Convolutional Block 1**: 64 filters, kernel size 11, SELU activation, average pooling
2. **Convolutional Block 2**: 128 filters, kernel size 11, SELU activation, average pooling
3. **Convolutional Block 3**: 256 filters, kernel size 11, SELU activation, average pooling
4. **Convolutional Block 4**: 512 filters, kernel size 11, SELU activation, average pooling
5. **Fully Connected Layer 1**: 512 units, SELU activation, 50% dropout
6. **Fully Connected Layer 2**: 256 units, SELU activation, 30% dropout
7. **Output Layer**: 1 unit, sigmoid activation

## Output Files

After running the classifier, the following files are generated:

### Models
- `best_model.h5`: Best model checkpoint (highest validation accuracy)
- `kyber_classifier_model.h5`: Final trained model

### Visualizations (in `visualizations/` directory)
- `signal_comparison.png`: Sample M0 and M1 signal traces
- `average_signals.png`: Average signals with standard deviation
- `pca_analysis.png`: PCA visualization of training data
- `frequency_analysis.png`: Power spectrum and spectrogram
- `training_history.png`: Training/validation accuracy and loss curves
- `evaluation_results.png`: Confusion matrix, ROC curve, and LLR distribution
- `confidence_analysis.png`: Prediction probability and NLL distributions

## Evaluation Metrics

### Standard Metrics
- **Accuracy**: Percentage of correctly classified samples
- **Confusion Matrix**: True positives, false positives, true negatives, false negatives
- **Classification Report**: Precision, recall, F1-score for each class
- **ROC Curve**: True positive rate vs. false positive rate with AUC score

### NLL (Negative Log Likelihood) Method
The NLL method provides enhanced classification confidence:
- Calculates log-likelihood ratios: `LLR = log(P(M1|x) / P(M0|x))`
- Classifies based on LLR threshold (LLR > 0 → M1, LLR ≤ 0 → M0)
- Provides average NLL scores for each class
- Offers improved detection in ambiguous cases

## Reproducibility

The code sets random seeds for reproducibility:
- NumPy seed: 42
- TensorFlow seed: 42

Running the classifier multiple times should produce consistent results.

## Memory Considerations

For large datasets, the classifier:
- Loads data in chunks if needed
- Uses batch processing during training
- Implements efficient data preprocessing

Default batch size is 32, but can be adjusted based on available memory:

```python
classifier.train(epochs=100, batch_size=64, validation_split=0.2)
```

## Troubleshooting

### Out of Memory Errors
- Reduce batch size: `classifier.train(batch_size=16)`
- Use CPU instead of GPU if needed

### Poor Performance
- Increase training epochs
- Adjust learning rate
- Verify data quality and normalization
- Check for class imbalance

### Missing Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Side-Channel Analysis Context

This classifier is designed for side-channel analysis of the Kyber post-quantum cryptographic algorithm. It detects information leakage through electromagnetic emissions during cryptographic operations.

**Key Concepts:**
- **M0 (Known Plaintext)**: Traces with predictable input patterns
- **M1 (Random Plaintext)**: Traces with randomized input patterns
- **Side-Channel Attack**: Exploiting physical implementations rather than theoretical weaknesses
- **Electromagnetic Analysis**: Monitoring EM emissions during cryptographic operations

## Citation

If you use this classifier in your research, please cite the relevant papers on side-channel analysis and the Kyber cryptographic algorithm.

## License

This project is provided for research and educational purposes.

## Contact

For questions or issues, please open an issue in the repository.
