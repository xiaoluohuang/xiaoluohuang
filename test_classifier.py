"""
Test script to verify the classifier works correctly and can load saved models.
"""

import os
import numpy as np
from kyber_signal_classifier import KyberSignalClassifier

def test_load_and_predict():
    """Test loading a saved model and making predictions."""
    print("Testing model loading and prediction...")
    
    # Initialize classifier
    classifier = KyberSignalClassifier(
        '200_GPT1_MV_4_class_m0.csv',
        '200_GPT1_MV_4_class_m1.csv'
    )
    
    # Load data
    classifier.load_and_preprocess_data()
    
    # Load the trained model
    if os.path.exists('kyber_classifier_model.h5'):
        classifier.load_model('kyber_classifier_model.h5')
        print("✓ Model loaded successfully")
        
        # Make predictions on test set
        predictions = classifier.model.predict(classifier.X_test)
        print(f"✓ Made predictions on {len(predictions)} test samples")
        print(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # Evaluate
        test_loss, test_acc = classifier.model.evaluate(classifier.X_test, classifier.y_test, verbose=0)
        print(f"✓ Test accuracy: {test_acc:.4f}")
        print(f"✓ Test loss: {test_loss:.4f}")
        
        return True
    else:
        print("✗ Model file not found. Please run kyber_signal_classifier.py first.")
        return False

def test_data_shapes():
    """Test that data loading produces correct shapes."""
    print("\nTesting data shapes...")
    
    classifier = KyberSignalClassifier(
        '200_GPT1_MV_4_class_m0.csv',
        '200_GPT1_MV_4_class_m1.csv'
    )
    
    classifier.load_and_preprocess_data()
    
    # Check shapes
    assert classifier.X_train.shape[0] == 320, f"Expected 320 training samples, got {classifier.X_train.shape[0]}"
    assert classifier.X_test.shape[0] == 80, f"Expected 80 test samples, got {classifier.X_test.shape[0]}"
    assert classifier.X_train.shape[1] == 1000, f"Expected 1000 features, got {classifier.X_train.shape[1]}"
    assert classifier.X_train.shape[2] == 1, f"Expected 1 channel, got {classifier.X_train.shape[2]}"
    
    print("✓ Training set shape is correct:", classifier.X_train.shape)
    print("✓ Test set shape is correct:", classifier.X_test.shape)
    
    # Check label distribution
    train_m0 = np.sum(classifier.y_train == 0)
    train_m1 = np.sum(classifier.y_train == 1)
    test_m0 = np.sum(classifier.y_test == 0)
    test_m1 = np.sum(classifier.y_test == 1)
    
    assert train_m0 == 160, f"Expected 160 M0 training samples, got {train_m0}"
    assert train_m1 == 160, f"Expected 160 M1 training samples, got {train_m1}"
    assert test_m0 == 40, f"Expected 40 M0 test samples, got {test_m0}"
    assert test_m1 == 40, f"Expected 40 M1 test samples, got {test_m1}"
    
    print(f"✓ Training labels: M0={train_m0}, M1={train_m1}")
    print(f"✓ Test labels: M0={test_m0}, M1={test_m1}")
    
    return True

def main():
    """Run all tests."""
    print("="*70)
    print("TESTING KYBER SIGNAL CLASSIFIER")
    print("="*70)
    
    try:
        # Test data shapes
        test_data_shapes()
        
        # Test model loading and prediction
        test_load_and_predict()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
