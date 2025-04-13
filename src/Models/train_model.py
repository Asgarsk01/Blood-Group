#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training module for Blood Group Classification.

This module provides functions to train, evaluate, and fine-tune
blood group classification models using extracted features from blood sample images.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Import the blood classifier module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models.blood_classifier import (
    BloodGroupClassifier, create_blood_classifier, 
    BLOOD_GROUPS, DEFAULT_MODEL_DIR, SKLEARN_AVAILABLE
)

# Optional dependency imports with try-except
try:
    from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.feature_selection import SelectKBest, f_classif
except ImportError:
    print("Warning: scikit-learn not available. Training functionality limited.")

# Constants for training
DEFAULT_TRAIN_TEST_SPLIT = 0.2
DEFAULT_RANDOM_SEED = 42
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_results')
DEFAULT_FEATURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'features')


def load_features(features_path, label_col='blood_group'):
    """
    Load features from CSV or JSON file.
    
    Args:
        features_path (str): Path to the features file
        label_col (str): Name of the label/target column
        
    Returns:
        tuple: Features (X), labels (y), and feature names
    """
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    file_ext = os.path.splitext(features_path)[1].lower()
    
    if file_ext == '.csv':
        # Load from CSV
        df = pd.read_csv(features_path)
    elif file_ext in ['.json', '.jsonl']:
        # Load from JSON or JSONL
        df = pd.read_json(features_path, lines=(file_ext == '.jsonl'))
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in features file")
    
    # Extract features and labels
    y = df[label_col].values
    X = df.drop(label_col, axis=1)
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Convert to numpy array
    X = X.values
    
    return X, y, feature_names


def select_features(X, y, feature_names, k='all'):
    """
    Select the k best features using ANOVA F-value.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target labels
        feature_names (list): Names of features
        k (int or 'all'): Number of top features to select
        
    Returns:
        tuple: Selected features (X_selected), selector, selected feature names
    """
    if not SKLEARN_AVAILABLE:
        print("Warning: scikit-learn not available. Feature selection skipped.")
        return X, None, feature_names
    
    # If k is 'all' or greater than the number of features, use all features
    if k == 'all' or k >= X.shape[1]:
        return X, None, feature_names
    
    # Select k best features
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_indices]
    
    print(f"Selected {len(selected_feature_names)} features")
    return X_selected, selector, selected_feature_names


def split_data(X, y, test_size=DEFAULT_TRAIN_TEST_SPLIT, random_state=DEFAULT_RANDOM_SEED, stratify=True):
    """
    Split data into training and testing sets.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target labels
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        stratify (bool): Whether to stratify the splits based on the labels
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if not SKLEARN_AVAILABLE:
        # Simple split without stratification
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        
        indices = np.random.RandomState(random_state).permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    # Use scikit-learn's train_test_split with stratification if requested
    if stratify:
        return train_test_split(X, y, test_size=test_size, 
                               random_state=random_state, stratify=y)
    else:
        return train_test_split(X, y, test_size=test_size, 
                               random_state=random_state)


def train_model(X_train, y_train, model_type='ensemble', n_estimators=100, verbose=True):
    """
    Train a blood group classifier.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        model_type (str): Type of classifier to use
        n_estimators (int): Number of estimators for ensemble methods
        verbose (bool): Whether to print training progress
        
    Returns:
        BloodGroupClassifier: Trained classifier
    """
    if verbose:
        print(f"Training {model_type} classifier with {X_train.shape[0]} samples and {X_train.shape[1]} features")
    
    # Create the classifier
    classifier = create_blood_classifier(model_type=model_type, n_estimators=n_estimators)
    
    # Train the classifier
    classifier.fit(X_train, y_train)
    
    return classifier


def evaluate_model(classifier, X_test, y_test, feature_names=None, 
                  save_results=False, results_dir=DEFAULT_RESULTS_DIR, model_name=None):
    """
    Evaluate a trained classifier and generate performance reports.
    
    Args:
        classifier (BloodGroupClassifier): Trained classifier
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        feature_names (list): Names of features for feature importance visualization
        save_results (bool): Whether to save evaluation results
        results_dir (str): Directory to save results
        model_name (str): Name of the model for saving results
        
    Returns:
        dict: Evaluation metrics
    """
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model_name is None:
        model_name = f"{classifier.model_type}_{timestamp}"
    
    # Evaluate the classifier
    metrics = classifier.evaluate(X_test, y_test)
    
    # Print evaluation metrics
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    for label, stats in metrics['classification_report'].items():
        if isinstance(stats, dict):
            print(f"  {label}: precision={stats['precision']:.4f}, recall={stats['recall']:.4f}, f1-score={stats['f1-score']:.4f}")
    
    # Save results if requested
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(results_dir, f"{model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
        
        # Save confusion matrix visualization
        cm_path = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")
        classifier.visualize_confusion_matrix(X_test, y_test, save_path=cm_path)
        
        # Save feature importance visualization if applicable
        if feature_names is not None and hasattr(classifier, 'visualize_feature_importance'):
            fi_path = os.path.join(results_dir, f"{model_name}_feature_importance.png")
            classifier.visualize_feature_importance(
                feature_names=feature_names, 
                top_n=min(20, len(feature_names)),
                save_path=fi_path
            )
    
    return metrics


def cross_validate(X, y, n_splits=5, model_type='ensemble', n_estimators=100, 
                  random_state=DEFAULT_RANDOM_SEED):
    """
    Perform cross-validation to evaluate model stability.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target labels
        n_splits (int): Number of cross-validation folds
        model_type (str): Type of classifier to use
        n_estimators (int): Number of estimators for ensemble methods
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Cross-validation results
    """
    if not SKLEARN_AVAILABLE:
        print("Warning: scikit-learn not available. Cross-validation skipped.")
        return None
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Track metrics for each fold
    fold_accuracies = []
    fold_metrics = []
    
    # Perform cross-validation
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {i+1}/{n_splits}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        classifier = train_model(
            X_train, y_train, 
            model_type=model_type, 
            n_estimators=n_estimators,
            verbose=False
        )
        
        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)
        
        # Store results
        fold_accuracies.append(metrics['accuracy'])
        fold_metrics.append(metrics)
        
        print(f"Fold {i+1} accuracy: {metrics['accuracy']:.4f}")
    
    # Calculate average metrics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    print(f"\nCross-validation results ({n_splits} folds):")
    print(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    results = {
        'fold_metrics': fold_metrics,
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'n_splits': n_splits
    }
    
    return results


def optimize_hyperparameters(X, y, model_type='rf', param_grid=None, 
                           cv=5, random_state=DEFAULT_RANDOM_SEED):
    """
    Optimize model hyperparameters using grid search.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target labels
        model_type (str): Type of classifier to use
        param_grid (dict): Parameter grid for grid search
        cv (int): Number of cross-validation folds
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: Best parameters, best score, and grid search results
    """
    if not SKLEARN_AVAILABLE:
        print("Warning: scikit-learn not available. Hyperparameter optimization skipped.")
        return None, None, None
    
    # Create a base classifier
    base_classifier = create_blood_classifier(model_type=model_type)
    
    # Default parameter grids if not provided
    if param_grid is None:
        if model_type == 'rf':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'svm':
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
        elif model_type == 'xgb':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        elif model_type == 'mlp':
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        else:
            print(f"No default param_grid for model_type '{model_type}'. Using limited grid.")
            param_grid = {'n_estimators': [50, 100, 200]}
    
    # Set up grid search
    grid_search = GridSearchCV(
        base_classifier.model if hasattr(base_classifier, 'model') else base_classifier,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    # Run grid search
    print(f"Running grid search for {model_type} model...")
    grid_search.fit(X, y)
    
    # Get results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")
    
    return best_params, best_score, grid_search


def save_training_pipeline(
    classifier, feature_names=None, selector=None, 
    model_dir=DEFAULT_MODEL_DIR, model_name=None
):
    """
    Save the trained model and associated preprocessing components.
    
    Args:
        classifier (BloodGroupClassifier): Trained classifier
        feature_names (list): Names of features
        selector (SelectKBest): Feature selector used during training
        model_dir (str): Directory to save the model
        model_name (str): Name of the model file
        
    Returns:
        str: Path to the saved model
    """
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"blood_group_model_{timestamp}.pkl"
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Full path to the model file
    model_path = os.path.join(model_dir, model_name)
    
    # Save the classifier
    save_path = classifier.save_model(model_path)
    
    # Save metadata and feature information
    metadata = {
        "model_type": classifier.model_type if hasattr(classifier, 'model_type') else 'unknown',
        "timestamp": datetime.now().isoformat(),
        "feature_names": feature_names,
        "has_feature_selector": selector is not None
    }
    
    metadata_path = os.path.splitext(model_path)[0] + "_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save feature selector if provided
    if selector is not None:
        selector_path = os.path.splitext(model_path)[0] + "_selector.pkl"
        with open(selector_path, 'wb') as f:
            pickle.dump(selector, f)
        print(f"Feature selector saved to: {selector_path}")
    
    return save_path


def main(features_path=None, model_type='ensemble', n_estimators=100, 
        use_feature_selection=True, k_features=20, test_size=DEFAULT_TRAIN_TEST_SPLIT,
        save_model=True, cross_val=True, optimize_params=False, 
        random_state=DEFAULT_RANDOM_SEED):
    """
    Main function to train and evaluate a blood group classifier.
    
    Args:
        features_path (str): Path to the features file
        model_type (str): Type of classifier to use
        n_estimators (int): Number of estimators for ensemble methods
        use_feature_selection (bool): Whether to use feature selection
        k_features (int): Number of top features to select
        test_size (float): Proportion of data to use for testing
        save_model (bool): Whether to save the trained model
        cross_val (bool): Whether to perform cross-validation
        optimize_params (bool): Whether to optimize hyperparameters
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: Trained classifier and evaluation metrics
    """
    # Check if scikit-learn is available
    if not SKLEARN_AVAILABLE:
        print("Warning: scikit-learn not available. Limited functionality.")
    
    # If no features path provided, look for default features file
    if features_path is None:
        default_features_path = os.path.join(DEFAULT_FEATURES_DIR, 'blood_group_features.csv')
        if os.path.exists(default_features_path):
            features_path = default_features_path
        else:
            raise ValueError("No features_path provided and default features file not found.")
    
    print(f"Loading features from: {features_path}")
    X, y, feature_names = load_features(features_path)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    
    # Display class distribution
    class_counts = np.bincount([BLOOD_GROUPS.index(label) if label in BLOOD_GROUPS else -1 
                              for label in y])
    print("\nClass distribution:")
    for i, count in enumerate(class_counts):
        if i < len(BLOOD_GROUPS):
            print(f"  {BLOOD_GROUPS[i]}: {count} samples")
    
    # Feature selection if requested
    selector = None
    if use_feature_selection and k_features < X.shape[1]:
        print(f"\nSelecting top {k_features} features...")
        X, selector, feature_names = select_features(X, y, feature_names, k=k_features)
    
    # Hyperparameter optimization if requested
    if optimize_params:
        print("\nOptimizing hyperparameters...")
        best_params, best_score, _ = optimize_hyperparameters(
            X, y, model_type=model_type, cv=5, random_state=random_state
        )
    
    # Cross-validation if requested
    if cross_val:
        print("\nPerforming cross-validation...")
        cv_results = cross_validate(
            X, y, n_splits=5, model_type=model_type, 
            n_estimators=n_estimators, random_state=random_state
        )
    
    # Split data into training and testing sets
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train the final model
    print("\nTraining final model...")
    classifier = train_model(
        X_train, y_train, model_type=model_type, 
        n_estimators=n_estimators, verbose=True
    )
    
    # Evaluate the final model
    print("\nEvaluating final model...")
    metrics = evaluate_model(
        classifier, X_test, y_test, feature_names=feature_names,
        save_results=True, model_name=f"{model_type}_final"
    )
    
    # Save the model if requested
    if save_model:
        print("\nSaving model...")
        save_training_pipeline(
            classifier, feature_names=feature_names, 
            selector=selector, model_name=f"{model_type}_final.pkl"
        )
    
    return classifier, metrics


if __name__ == "__main__":
    print("Blood Group Classifier Training Module")
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a blood group classifier")
    parser.add_argument("--features", help="Path to features file (CSV or JSON)")
    parser.add_argument("--model-type", default="ensemble", 
                      choices=["rf", "svm", "xgb", "mlp", "ensemble"],
                      help="Type of classifier to use")
    parser.add_argument("--n-estimators", type=int, default=100, 
                      help="Number of estimators for ensemble methods")
    parser.add_argument("--no-feature-selection", action="store_true", 
                      help="Disable feature selection")
    parser.add_argument("--k-features", type=int, default=20, 
                      help="Number of top features to select")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TRAIN_TEST_SPLIT, 
                      help="Proportion of data to use for testing")
    parser.add_argument("--no-save", action="store_true", 
                      help="Don't save the trained model")
    parser.add_argument("--no-cross-val", action="store_true", 
                      help="Don't perform cross-validation")
    parser.add_argument("--optimize", action="store_true", 
                      help="Optimize hyperparameters")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, 
                      help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    try:
        main(
            features_path=args.features,
            model_type=args.model_type,
            n_estimators=args.n_estimators,
            use_feature_selection=not args.no_feature_selection,
            k_features=args.k_features,
            test_size=args.test_size,
            save_model=not args.no_save,
            cross_val=not args.no_cross_val,
            optimize_params=args.optimize,
            random_state=args.random_seed
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 