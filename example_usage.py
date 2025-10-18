"""
Example Usage Script for Kyber Signal Classifier

This script demonstrates how to use the classifier for common tasks.
"""

from kyber_signal_classifier import KyberSignalClassifier


def example_1_quick_start():
    """
    Example 1: Quick start - train and evaluate a new model
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: QUICK START - TRAIN AND EVALUATE")
    print("="*70)
    
    # Initialize classifier with your data paths
    classifier = KyberSignalClassifier(
        m0_path='200_GPT1_MV_4_class_m0.csv',
        m1_path='200_GPT1_MV_4_class_m1.csv'
    )
    
    # Load and preprocess data
    print("\n1. Loading data...")
    classifier.load_and_preprocess_data()
    
    # Generate data visualizations
    print("\n2. Generating visualizations...")
    classifier.visualize_data()
    
    # Build model
    print("\n3. Building model...")
    classifier.build_model()
    
    # Train model (use fewer epochs for quick testing)
    print("\n4. Training model...")
    classifier.train(epochs=20, batch_size=32, validation_split=0.2)
    
    # Evaluate
    print("\n5. Evaluating model...")
    eval_results = classifier.evaluate()
    nll_results = classifier.nll_evaluation()
    
    # Plot results
    print("\n6. Plotting results...")
    classifier.plot_training_history()
    classifier.plot_evaluation_results(eval_results, nll_results)
    
    # Save model
    print("\n7. Saving model...")
    classifier.save_model('my_classifier_model.h5')
    
    print("\n✓ Example 1 complete!")


def example_2_load_and_evaluate():
    """
    Example 2: Load an existing model and evaluate it
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: LOAD EXISTING MODEL AND EVALUATE")
    print("="*70)
    
    # Initialize classifier
    classifier = KyberSignalClassifier(
        m0_path='200_GPT1_MV_4_class_m0.csv',
        m1_path='200_GPT1_MV_4_class_m1.csv'
    )
    
    # Load data
    print("\n1. Loading data...")
    classifier.load_and_preprocess_data()
    
    # Load pre-trained model
    print("\n2. Loading pre-trained model...")
    classifier.load_model('kyber_classifier_model.h5')
    
    # Evaluate
    print("\n3. Evaluating...")
    eval_results = classifier.evaluate()
    nll_results = classifier.nll_evaluation()
    
    print("\n✓ Example 2 complete!")


def example_3_custom_training():
    """
    Example 3: Custom training configuration
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: CUSTOM TRAINING CONFIGURATION")
    print("="*70)
    
    classifier = KyberSignalClassifier(
        m0_path='200_GPT1_MV_4_class_m0.csv',
        m1_path='200_GPT1_MV_4_class_m1.csv'
    )
    
    # Load data
    print("\n1. Loading data...")
    classifier.load_and_preprocess_data()
    
    # Build model
    print("\n2. Building model...")
    classifier.build_model()
    
    # Custom training parameters
    print("\n3. Training with custom parameters...")
    classifier.train(
        epochs=50,              # More epochs for better training
        batch_size=64,          # Larger batch size
        validation_split=0.25   # 25% validation split
    )
    
    # Evaluate
    print("\n4. Evaluating...")
    eval_results = classifier.evaluate()
    nll_results = classifier.nll_evaluation()
    
    # Save with custom name
    print("\n5. Saving model...")
    classifier.save_model('custom_model.h5')
    
    print("\n✓ Example 3 complete!")


def example_4_prediction_only():
    """
    Example 4: Load model and make predictions on test data
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: PREDICTION ONLY")
    print("="*70)
    
    import numpy as np
    
    classifier = KyberSignalClassifier(
        m0_path='200_GPT1_MV_4_class_m0.csv',
        m1_path='200_GPT1_MV_4_class_m1.csv'
    )
    
    # Load data
    print("\n1. Loading data...")
    classifier.load_and_preprocess_data()
    
    # Load model
    print("\n2. Loading model...")
    classifier.load_model('kyber_classifier_model.h5')
    
    # Make predictions
    print("\n3. Making predictions...")
    predictions = classifier.model.predict(classifier.X_test)
    
    # Show some predictions
    print("\nSample predictions:")
    print(f"{'Sample':<10} {'True Label':<15} {'Predicted Prob':<20} {'Predicted Class':<15}")
    print("-" * 60)
    
    for i in range(min(10, len(predictions))):
        true_label = "M0 (Known)" if classifier.y_test[i] == 0 else "M1 (Random)"
        pred_prob = predictions[i][0]
        pred_class = "M0 (Known)" if pred_prob < 0.5 else "M1 (Random)"
        print(f"{i+1:<10} {true_label:<15} {pred_prob:<20.4f} {pred_class:<15}")
    
    print("\n✓ Example 4 complete!")


def main():
    """
    Main function - select which example to run
    """
    print("\n" + "="*70)
    print("KYBER SIGNAL CLASSIFIER - EXAMPLE USAGE")
    print("="*70)
    
    print("\nAvailable examples:")
    print("  1. Quick start - train and evaluate a new model")
    print("  2. Load an existing model and evaluate it")
    print("  3. Custom training configuration")
    print("  4. Prediction only")
    print("\nTo run an example, call the corresponding function:")
    print("  - example_1_quick_start()")
    print("  - example_2_load_and_evaluate()")
    print("  - example_3_custom_training()")
    print("  - example_4_prediction_only()")
    
    # Uncomment to run a specific example:
    # example_1_quick_start()
    # example_2_load_and_evaluate()
    # example_3_custom_training()
    # example_4_prediction_only()
    
    print("\nFor a complete run, execute: python kyber_signal_classifier.py")


if __name__ == "__main__":
    main()
