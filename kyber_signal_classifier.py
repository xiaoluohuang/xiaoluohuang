"""
Electromagnetic Signal Classifier for Kyber Cryptographic Implementation Analysis

This script implements a CNN-based binary classifier to distinguish between M0 (known plaintext)
and M1 (random plaintext) electromagnetic traces in Kyber cryptographic implementations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from scipy import signal as scipy_signal
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class KyberSignalClassifier:
    """
    CNN-based classifier for electromagnetic signal analysis in Kyber cryptographic implementations.
    """
    
    def __init__(self, m0_path, m1_path):
        """
        Initialize the classifier with paths to M0 and M1 datasets.
        
        Args:
            m0_path: Path to M0 (known plaintext) CSV file
            m1_path: Path to M1 (random plaintext) CSV file
        """
        self.m0_path = m0_path
        self.m1_path = m1_path
        self.model = None
        self.history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.input_shape = None
        
    def load_and_preprocess_data(self):
        """
        Load CSV datasets and prepare them for training.
        
        Data structure:
        - Each column represents one trace/sample
        - Each row represents a feature/time point
        - First row is the header (column indices)
        """
        print("Loading datasets...")
        
        # Load M0 dataset (known plaintext, label=0)
        m0_data = pd.read_csv(self.m0_path)
        print(f"M0 dataset shape: {m0_data.shape}")
        
        # Load M1 dataset (random plaintext, label=1)
        m1_data = pd.read_csv(self.m1_path)
        print(f"M1 dataset shape: {m1_data.shape}")
        
        # Transpose data: columns become samples, rows become features
        # Skip first row if it's just column indices
        m0_traces = m0_data.iloc[:, :].T.values  # Shape: (num_samples, num_features)
        m1_traces = m1_data.iloc[:, :].T.values
        
        print(f"M0 traces shape after transpose: {m0_traces.shape}")
        print(f"M1 traces shape after transpose: {m1_traces.shape}")
        
        # Normalize by dividing by 256
        m0_traces = m0_traces / 256.0
        m1_traces = m1_traces / 256.0
        
        # Create labels
        m0_labels = np.zeros(m0_traces.shape[0])
        m1_labels = np.ones(m1_traces.shape[0])
        
        # Implement 80/20 split: 160 training, 40 test per dataset
        num_train_per_class = 160
        num_test_per_class = 40
        
        # Split M0
        m0_train = m0_traces[:num_train_per_class]
        m0_test = m0_traces[num_train_per_class:num_train_per_class + num_test_per_class]
        m0_train_labels = m0_labels[:num_train_per_class]
        m0_test_labels = m0_labels[num_train_per_class:num_train_per_class + num_test_per_class]
        
        # Split M1
        m1_train = m1_traces[:num_train_per_class]
        m1_test = m1_traces[num_train_per_class:num_train_per_class + num_test_per_class]
        m1_train_labels = m1_labels[:num_train_per_class]
        m1_test_labels = m1_labels[num_train_per_class:num_train_per_class + num_test_per_class]
        
        # Combine training and test sets
        self.X_train = np.concatenate([m0_train, m1_train], axis=0)
        self.X_test = np.concatenate([m0_test, m1_test], axis=0)
        self.y_train = np.concatenate([m0_train_labels, m1_train_labels], axis=0)
        self.y_test = np.concatenate([m0_test_labels, m1_test_labels], axis=0)
        
        # Shuffle training data
        train_indices = np.random.permutation(len(self.X_train))
        self.X_train = self.X_train[train_indices]
        self.y_train = self.y_train[train_indices]
        
        # Shuffle test data
        test_indices = np.random.permutation(len(self.X_test))
        self.X_test = self.X_test[test_indices]
        self.y_test = self.y_test[test_indices]
        
        # Reshape for CNN input (add channel dimension)
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        
        self.input_shape = self.X_train.shape[1:]
        
        print(f"\nFinal dataset shapes:")
        print(f"Training set: {self.X_train.shape}, Labels: {self.y_train.shape}")
        print(f"Test set: {self.X_test.shape}, Labels: {self.y_test.shape}")
        print(f"Training label distribution: M0={np.sum(self.y_train == 0)}, M1={np.sum(self.y_train == 1)}")
        print(f"Test label distribution: M0={np.sum(self.y_test == 0)}, M1={np.sum(self.y_test == 1)}")
        
    def build_model(self):
        """
        Build CNN architecture for binary classification.
        
        Architecture:
        - Convolutional layers with batch normalization and SELU activation
        - Average pooling layers for dimensionality reduction
        - Fully connected layers for classification
        - Sigmoid output for binary classification
        """
        print("\nBuilding CNN model...")
        
        model = models.Sequential([
            # First convolutional block
            layers.Conv1D(64, kernel_size=11, strides=1, padding='same', 
                         input_shape=self.input_shape, name='conv1'),
            layers.BatchNormalization(name='bn1'),
            layers.Activation('selu', name='selu1'),
            layers.AveragePooling1D(pool_size=2, name='pool1'),
            
            # Second convolutional block
            layers.Conv1D(128, kernel_size=11, strides=1, padding='same', name='conv2'),
            layers.BatchNormalization(name='bn2'),
            layers.Activation('selu', name='selu2'),
            layers.AveragePooling1D(pool_size=2, name='pool2'),
            
            # Third convolutional block
            layers.Conv1D(256, kernel_size=11, strides=1, padding='same', name='conv3'),
            layers.BatchNormalization(name='bn3'),
            layers.Activation('selu', name='selu3'),
            layers.AveragePooling1D(pool_size=2, name='pool3'),
            
            # Fourth convolutional block
            layers.Conv1D(512, kernel_size=11, strides=1, padding='same', name='conv4'),
            layers.BatchNormalization(name='bn4'),
            layers.Activation('selu', name='selu4'),
            layers.AveragePooling1D(pool_size=2, name='pool4'),
            
            # Flatten and fully connected layers
            layers.Flatten(name='flatten'),
            layers.Dense(512, activation='selu', name='fc1'),
            layers.Dropout(0.5, name='dropout1'),
            layers.Dense(256, activation='selu', name='fc2'),
            layers.Dropout(0.3, name='dropout2'),
            layers.Dense(1, activation='sigmoid', name='output')
        ])
        
        # Compile model with binary crossentropy and Adam optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print(model.summary())
        
    def train(self, epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the model with early stopping and model checkpointing.
        
        Args:
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data to use for validation
        """
        print("\nTraining model...")
        
        # Create callbacks
        checkpoint_cb = callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )
        
        early_stopping_cb = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        
        # Train model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[checkpoint_cb, early_stopping_cb],
            verbose=1
        )
        
        print("\nTraining completed!")
        
    def evaluate(self):
        """
        Evaluate model on test set with standard metrics.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        print("\n" + "="*50)
        print("STANDARD EVALUATION")
        print("="*50)
        
        # Predict on test set
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate accuracy
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['M0 (Known)', 'M1 (Random)']))
        
        return {
            'accuracy': test_acc,
            'loss': test_loss,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def nll_evaluation(self):
        """
        Implement Negative Log Likelihood (NLL) comparison method for enhanced detection.
        
        This method calculates log-likelihood ratios to improve classification confidence.
        """
        print("\n" + "="*50)
        print("NLL (NEGATIVE LOG LIKELIHOOD) EVALUATION")
        print("="*50)
        
        # Get prediction probabilities
        y_pred_proba = self.model.predict(self.X_test).flatten()
        
        # Calculate log-likelihood ratios
        # LLR = log(P(M1|x) / P(M0|x))
        epsilon = 1e-10  # Small constant to avoid log(0)
        p_m1 = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        p_m0 = 1 - p_m1
        
        log_likelihood_ratio = np.log(p_m1 / p_m0)
        
        # Calculate NLL for each class
        nll_m0 = -np.log(p_m0)
        nll_m1 = -np.log(p_m1)
        
        # Make predictions based on NLL
        y_pred_nll = (log_likelihood_ratio > 0).astype(int)
        
        # Calculate accuracy
        nll_accuracy = np.mean(y_pred_nll == self.y_test)
        print(f"\nNLL-based Accuracy: {nll_accuracy:.4f}")
        
        # Confusion matrix for NLL method
        cm_nll = confusion_matrix(self.y_test, y_pred_nll)
        print("\nNLL Confusion Matrix:")
        print(cm_nll)
        
        # Classification report for NLL method
        print("\nNLL Classification Report:")
        print(classification_report(self.y_test, y_pred_nll,
                                   target_names=['M0 (Known)', 'M1 (Random)']))
        
        # Calculate average NLL for each class
        m0_indices = self.y_test == 0
        m1_indices = self.y_test == 1
        
        avg_nll_m0 = np.mean(nll_m0[m0_indices])
        avg_nll_m1 = np.mean(nll_m1[m1_indices])
        
        print(f"\nAverage NLL for M0 samples: {avg_nll_m0:.4f}")
        print(f"Average NLL for M1 samples: {avg_nll_m1:.4f}")
        
        return {
            'nll_accuracy': nll_accuracy,
            'log_likelihood_ratio': log_likelihood_ratio,
            'nll_m0': nll_m0,
            'nll_m1': nll_m1,
            'y_pred_nll': y_pred_nll
        }
    
    def visualize_data(self, output_dir='visualizations'):
        """
        Create comprehensive data visualizations.
        
        Args:
            output_dir: Directory to save visualization plots
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating visualizations in '{output_dir}' directory...")
        
        # 1. Signal Comparison: Sample traces from M0 and M1
        plt.figure(figsize=(15, 6))
        
        # Plot M0 samples
        plt.subplot(1, 2, 1)
        for i in range(min(5, np.sum(self.y_train == 0))):
            idx = np.where(self.y_train == 0)[0][i]
            plt.plot(self.X_train[idx, :, 0], alpha=0.6, label=f'M0 Sample {i+1}')
        plt.title('M0 (Known Plaintext) Signal Traces')
        plt.xlabel('Time Point')
        plt.ylabel('Normalized Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot M1 samples
        plt.subplot(1, 2, 2)
        for i in range(min(5, np.sum(self.y_train == 1))):
            idx = np.where(self.y_train == 1)[0][i]
            plt.plot(self.X_train[idx, :, 0], alpha=0.6, label=f'M1 Sample {i+1}')
        plt.title('M1 (Random Plaintext) Signal Traces')
        plt.xlabel('Time Point')
        plt.ylabel('Normalized Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/signal_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Signal comparison plot saved")
        
        # 2. Average Signal Comparison
        plt.figure(figsize=(12, 5))
        
        m0_indices = self.y_train == 0
        m1_indices = self.y_train == 1
        
        m0_mean = np.mean(self.X_train[m0_indices, :, 0], axis=0)
        m1_mean = np.mean(self.X_train[m1_indices, :, 0], axis=0)
        m0_std = np.std(self.X_train[m0_indices, :, 0], axis=0)
        m1_std = np.std(self.X_train[m1_indices, :, 0], axis=0)
        
        time_points = np.arange(len(m0_mean))
        
        plt.subplot(1, 2, 1)
        plt.plot(time_points, m0_mean, label='M0 Mean', linewidth=2)
        plt.fill_between(time_points, m0_mean - m0_std, m0_mean + m0_std, alpha=0.3)
        plt.plot(time_points, m1_mean, label='M1 Mean', linewidth=2)
        plt.fill_between(time_points, m1_mean - m1_std, m1_mean + m1_std, alpha=0.3)
        plt.title('Average Signal Traces with Standard Deviation')
        plt.xlabel('Time Point')
        plt.ylabel('Normalized Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        difference = np.abs(m0_mean - m1_mean)
        plt.plot(time_points, difference, color='red', linewidth=2)
        plt.title('Absolute Difference between M0 and M1 Averages')
        plt.xlabel('Time Point')
        plt.ylabel('Absolute Difference')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/average_signals.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Average signal comparison plot saved")
        
        # 3. PCA Analysis
        print("Performing PCA analysis...")
        X_train_2d = self.X_train.reshape(self.X_train.shape[0], -1)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train_2d)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y_train, 
                            cmap='coolwarm', alpha=0.6, edgecolors='k')
        plt.colorbar(scatter, label='Class (0=M0, 1=M1)')
        plt.title(f'PCA Visualization of Training Data\nExplained Variance: {sum(pca.explained_variance_ratio_):.2%}')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ PCA analysis plot saved")
        
        # 4. Frequency Domain Analysis
        print("Performing frequency domain analysis...")
        plt.figure(figsize=(12, 5))
        
        # FFT of M0 average
        m0_fft = np.fft.fft(m0_mean)
        m0_freq = np.fft.fftfreq(len(m0_mean))
        m0_power = np.abs(m0_fft) ** 2
        
        # FFT of M1 average
        m1_fft = np.fft.fft(m1_mean)
        m1_freq = np.fft.fftfreq(len(m1_mean))
        m1_power = np.abs(m1_fft) ** 2
        
        # Plot only positive frequencies
        positive_freqs = m0_freq > 0
        
        plt.subplot(1, 2, 1)
        plt.semilogy(m0_freq[positive_freqs], m0_power[positive_freqs], label='M0', alpha=0.7)
        plt.semilogy(m1_freq[positive_freqs], m1_power[positive_freqs], label='M1', alpha=0.7)
        plt.title('Power Spectrum (Frequency Domain)')
        plt.xlabel('Frequency')
        plt.ylabel('Power (log scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Spectrogram
        plt.subplot(1, 2, 2)
        sample_idx = np.where(self.y_train == 0)[0][0]
        f, t, Sxx = scipy_signal.spectrogram(self.X_train[sample_idx, :, 0], nperseg=128)
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        plt.colorbar(label='Power (dB)')
        plt.title('Spectrogram (M0 Sample)')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/frequency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Frequency domain analysis plot saved")
        
    def plot_training_history(self, output_dir='visualizations'):
        """
        Plot training history (loss and accuracy curves).
        
        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        print(f"\nPlotting training history...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy over Epochs', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot training & validation loss
        axes[1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss over Epochs', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Training history plot saved")
        
    def plot_evaluation_results(self, eval_results, nll_results, output_dir='visualizations'):
        """
        Plot evaluation results including confusion matrix and ROC curve.
        
        Args:
            eval_results: Dictionary from evaluate() method
            nll_results: Dictionary from nll_evaluation() method
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nPlotting evaluation results...")
        
        fig = plt.figure(figsize=(18, 6))
        
        # 1. Confusion Matrix (Standard)
        ax1 = plt.subplot(1, 3, 1)
        sns.heatmap(eval_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['M0 (Known)', 'M1 (Random)'],
                   yticklabels=['M0 (Known)', 'M1 (Random)'])
        plt.title('Confusion Matrix (Standard Method)', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=11)
        plt.xlabel('Predicted Label', fontsize=11)
        
        # 2. ROC Curve
        ax2 = plt.subplot(1, 3, 2)
        fpr, tpr, thresholds = roc_curve(self.y_test, eval_results['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=11)
        plt.ylabel('True Positive Rate', fontsize=11)
        plt.title('ROC Curve', fontsize=12, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 3. NLL Distribution
        ax3 = plt.subplot(1, 3, 3)
        m0_indices = self.y_test == 0
        m1_indices = self.y_test == 1
        
        llr = nll_results['log_likelihood_ratio']
        # Filter out infinite values for plotting
        llr_m0 = llr[m0_indices]
        llr_m1 = llr[m1_indices]
        llr_m0_finite = llr_m0[np.isfinite(llr_m0)]
        llr_m1_finite = llr_m1[np.isfinite(llr_m1)]
        
        if len(llr_m0_finite) > 0:
            plt.hist(llr_m0_finite, bins=30, alpha=0.6, label='M0 (Known)', color='blue', edgecolor='black')
        if len(llr_m1_finite) > 0:
            plt.hist(llr_m1_finite, bins=30, alpha=0.6, label='M1 (Random)', color='red', edgecolor='black')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
        plt.xlabel('Log-Likelihood Ratio', fontsize=11)
        plt.ylabel('Frequency', fontsize=11)
        plt.title('Log-Likelihood Ratio Distribution', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Evaluation results plot saved")
        
        # Additional plot: Prediction Confidence
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Prediction probabilities distribution
        proba_m0 = eval_results['y_pred_proba'][m0_indices].flatten()
        proba_m1 = eval_results['y_pred_proba'][m1_indices].flatten()
        axes[0].hist(proba_m0, bins=30, alpha=0.6, 
                    label='M0 (Known)', color='blue', edgecolor='black')
        axes[0].hist(proba_m1, bins=30, alpha=0.6,
                    label='M1 (Random)', color='red', edgecolor='black')
        axes[0].axvline(x=0.5, color='black', linestyle='--', linewidth=2, 
                       label='Decision Threshold')
        axes[0].set_xlabel('Prediction Probability (P(M1))', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # NLL values distribution - filter out infinite values
        nll_m0_vals = nll_results['nll_m0'][m0_indices]
        nll_m1_vals = nll_results['nll_m1'][m1_indices]
        nll_m0_finite = nll_m0_vals[np.isfinite(nll_m0_vals)]
        nll_m1_finite = nll_m1_vals[np.isfinite(nll_m1_vals)]
        
        if len(nll_m0_finite) > 0:
            axes[1].hist(nll_m0_finite, bins=30, alpha=0.6,
                        label='M0 NLL', color='blue', edgecolor='black')
        if len(nll_m1_finite) > 0:
            axes[1].hist(nll_m1_finite, bins=30, alpha=0.6,
                        label='M1 NLL', color='red', edgecolor='black')
        axes[1].set_xlabel('Negative Log-Likelihood', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('NLL Distribution by True Class', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Confidence analysis plot saved")
        
    def save_model(self, filepath='kyber_classifier_model.h5'):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            print("No model to save. Train the model first.")
            return
        
        self.model.save(filepath)
        print(f"\nModel saved to '{filepath}'")
        
    def load_model(self, filepath='kyber_classifier_model.h5'):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"\nModel loaded from '{filepath}'")


def main():
    """
    Main function to run the complete electromagnetic signal classification pipeline.
    """
    print("="*70)
    print("ELECTROMAGNETIC SIGNAL CLASSIFIER FOR KYBER CRYPTOGRAPHIC ANALYSIS")
    print("="*70)
    
    # Define file paths
    m0_path = '/home/runner/work/xiaoluohuang/xiaoluohuang/200_GPT1_MV_4_class_m0.csv'
    m1_path = '/home/runner/work/xiaoluohuang/xiaoluohuang/200_GPT1_MV_4_class_m1.csv'
    
    # Verify files exist
    if not os.path.exists(m0_path):
        raise FileNotFoundError(f"M0 dataset not found: {m0_path}")
    if not os.path.exists(m1_path):
        raise FileNotFoundError(f"M1 dataset not found: {m1_path}")
    
    # Initialize classifier
    classifier = KyberSignalClassifier(m0_path, m1_path)
    
    # Load and preprocess data
    classifier.load_and_preprocess_data()
    
    # Generate data visualizations
    classifier.visualize_data()
    
    # Build model
    classifier.build_model()
    
    # Train model
    classifier.train(epochs=100, batch_size=32, validation_split=0.2)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Evaluate model
    eval_results = classifier.evaluate()
    
    # NLL evaluation
    nll_results = classifier.nll_evaluation()
    
    # Plot evaluation results
    classifier.plot_evaluation_results(eval_results, nll_results)
    
    # Save model
    classifier.save_model('kyber_classifier_model.h5')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - best_model.h5: Best model checkpoint")
    print("  - kyber_classifier_model.h5: Final trained model")
    print("  - visualizations/: All visualization plots")
    print("\nAll plots have been saved in the 'visualizations' directory.")
    

if __name__ == "__main__":
    main()
