#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Blood Group Classification Model using machine learning techniques.
Implements different classifiers for blood group detection from processed blood samples.
"""

import os
import numpy as np
import pickle
import json
from collections import Counter
import matplotlib.pyplot as plt

# Optional ML libraries imports with try-except to avoid dependency errors
try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Limited functionality.")

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Constants for the blood group classifier
BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
DEFAULT_MODEL_NAME = 'blood_group_model.pkl'


class BloodGroupClassifier:
    """
    Classifier for blood group detection from processed blood sample features.
    Supports multiple ML algorithms and ensemble approaches.
    """
    
    def __init__(self, model_type='ensemble', n_estimators=DEFAULT_N_ESTIMATORS):
        """
        Initialize the blood group classifier.
        
        Args:
            model_type (str): Type of classifier to use 
                              ('rf', 'svm', 'xgb', 'mlp', or 'ensemble')
            n_estimators (int): Number of estimators for ensemble methods
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for blood group classification")
            
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.scaler = StandardScaler()
        self.models = {}
        self.ensemble_weights = {}
        self.feature_importance = None
        
        # Initialize the classifier based on model_type
        self._init_model()
        
    def _init_model(self):
        """Initialize the machine learning model(s) based on model_type."""
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                C=10.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            )
        elif self.model_type == 'xgb':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is required for model_type='xgb'")
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        elif self.model_type == 'ensemble':
            # Create multiple models for ensemble voting
            self.models = {
                'rf': RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=10,
                    random_state=42
                ),
                'svm': SVC(
                    C=10.0,
                    kernel='rbf',
                    gamma='scale',
                    probability=True,
                    random_state=42
                ),
                'mlp': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    random_state=42
                )
            }
            
            if XGBOOST_AVAILABLE:
                self.models['xgb'] = xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                
            # Default equal weights for ensemble
            total_models = len(self.models)
            self.ensemble_weights = {model: 1.0/total_models for model in self.models}
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def preprocess_features(self, X, fit=False):
        """
        Preprocess features using standardization.
        
        Args:
            X (numpy.ndarray): Feature matrix
            fit (bool): Whether to fit the scaler or just transform
            
        Returns:
            numpy.ndarray: Preprocessed features
        """
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
    
    def fit(self, X, y):
        """
        Train the blood group classifier.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target blood group labels
            
        Returns:
            self: Trained classifier
        """
        # Preprocess the features
        X_scaled = self.preprocess_features(X, fit=True)
        
        if self.model_type == 'ensemble':
            # Train each model in the ensemble
            for name, model in self.models.items():
                print(f"Training {name} model...")
                model.fit(X_scaled, y)
                
            # Evaluate and update weights based on performance
            self._update_ensemble_weights(X_scaled, y)
        else:
            # Train the single model
            self.model.fit(X_scaled, y)
            
            # Calculate feature importance for tree-based models
            if self.model_type in ['rf', 'xgb']:
                self.feature_importance = self.model.feature_importances_
                
        return self
    
    def _update_ensemble_weights(self, X, y):
        """
        Update ensemble weights based on model performance.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target blood group labels
        """
        scores = {}
        total_score = 0.0
        
        # Calculate cross-validation score for each model
        for name, model in self.models.items():
            score = np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy'))
            scores[name] = score
            total_score += score
        
        # Update weights based on relative performance
        if total_score > 0:
            for name in self.models.keys():
                self.ensemble_weights[name] = scores[name] / total_score
                
        print("Ensemble weights:", self.ensemble_weights)
    
    def predict(self, X):
        """
        Predict blood group for input features.
        
        Args:
            X (numpy.ndarray): Feature matrix
            
        Returns:
            numpy.ndarray: Predicted blood group labels
        """
        # Preprocess the features
        X_scaled = self.preprocess_features(X)
        
        if self.model_type == 'ensemble':
            # Get predictions from each model
            predictions = []
            for name, model in self.models.items():
                pred = model.predict(X_scaled)
                predictions.append(pred)
            
            # Weighted voting
            final_predictions = []
            for i in range(len(X)):
                votes = {}
                for name, preds in zip(self.models.keys(), predictions):
                    pred = preds[i]
                    if pred in votes:
                        votes[pred] += self.ensemble_weights[name]
                    else:
                        votes[pred] = self.ensemble_weights[name]
                
                # Select the class with the highest weighted vote
                final_predictions.append(max(votes.items(), key=lambda x: x[1])[0])
            
            return np.array(final_predictions)
        else:
            # Single model prediction
            return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for input features.
        
        Args:
            X (numpy.ndarray): Feature matrix
            
        Returns:
            numpy.ndarray: Class probabilities
        """
        # Preprocess the features
        X_scaled = self.preprocess_features(X)
        
        if self.model_type == 'ensemble':
            # Get probability predictions from each model
            all_probs = []
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_scaled)
                    all_probs.append((name, probs))
            
            # Weighted average of probabilities
            if not all_probs:
                raise ValueError("No models with predict_proba available in ensemble")
                
            # Initialize with zeros
            final_probs = np.zeros((X.shape[0], len(BLOOD_GROUPS)))
            
            # Weight and sum probabilities
            weight_sum = 0.0
            for name, probs in all_probs:
                weight = self.ensemble_weights[name]
                final_probs += weight * probs
                weight_sum += weight
                
            # Normalize
            if weight_sum > 0:
                final_probs /= weight_sum
                
            return final_probs
        else:
            # Single model probability prediction
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X_scaled)
            else:
                raise ValueError(f"Model {self.model_type} does not support predict_proba")
    
    def evaluate(self, X, y):
        """
        Evaluate the classifier performance.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): True blood group labels
            
        Returns:
            dict: Performance metrics
        """
        # Preprocess the features
        X_scaled = self.preprocess_features(X)
        
        # Make predictions
        y_pred = self.predict(X_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Compile results
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        return metrics
    
    def save_model(self, model_path=None):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str, optional): Path to save the model
            
        Returns:
            str: Path where the model was saved
        """
        if model_path is None:
            os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
            model_path = os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the full classifier object
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
            
        print(f"Model saved to: {model_path}")
        return model_path
    
    @classmethod
    def load_model(cls, model_path=None):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str, optional): Path to the saved model
            
        Returns:
            BloodGroupClassifier: Loaded classifier
        """
        if model_path is None:
            model_path = os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME)
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load the full classifier object
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        print(f"Model loaded from: {model_path}")
        return model
    
    def visualize_feature_importance(self, feature_names=None, top_n=10, save_path=None):
        """
        Visualize feature importance for tree-based models.
        
        Args:
            feature_names (list, optional): Names of features
            top_n (int): Number of top features to show
            save_path (str, optional): Path to save the visualization
        """
        if self.model_type not in ['rf', 'xgb'] and 'rf' not in self.models:
            print(f"Feature importance not available for model type: {self.model_type}")
            return
        
        if self.model_type == 'ensemble':
            if 'rf' in self.models:
                importance = self.models['rf'].feature_importances_
            elif 'xgb' in self.models and XGBOOST_AVAILABLE:
                importance = self.models['xgb'].feature_importances_
            else:
                print("No tree-based model in ensemble for feature importance")
                return
        else:
            importance = self.feature_importance
            
        if importance is None:
            print("No feature importance data available. Model may not be trained.")
            return
            
        # Use default feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importance))]
            
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        
        # Plot top N features
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance for Blood Group Classification")
        plt.bar(
            range(min(top_n, len(indices))),
            importance[indices[:top_n]],
            align="center"
        )
        plt.xticks(
            range(min(top_n, len(indices))),
            [feature_names[i] for i in indices[:top_n]],
            rotation=45,
            ha="right"
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Feature importance visualization saved to: {save_path}")
            
        plt.show()
    
    def visualize_confusion_matrix(self, X, y, save_path=None):
        """
        Visualize confusion matrix for model evaluation.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): True blood group labels
            save_path (str, optional): Path to save the visualization
        """
        # Preprocess features
        X_scaled = self.preprocess_features(X)
        
        # Make predictions
        y_pred = self.predict(X_scaled)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Create labels if needed
        labels = sorted(set(y))
        if len(labels) <= len(BLOOD_GROUPS):
            labels = [bg for bg in BLOOD_GROUPS if bg in labels]
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix for Blood Group Classification")
        plt.colorbar()
        
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black"
                )
                
        plt.tight_layout()
        plt.ylabel('True Blood Group')
        plt.xlabel('Predicted Blood Group')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix visualization saved to: {save_path}")
            
        plt.show()


def create_blood_classifier(model_type='ensemble', n_estimators=DEFAULT_N_ESTIMATORS):
    """
    Factory function to create a blood group classifier.
    
    Args:
        model_type (str): Type of classifier to use
        n_estimators (int): Number of estimators for ensemble methods
        
    Returns:
        BloodGroupClassifier: Initialized classifier
    """
    if not SKLEARN_AVAILABLE:
        return DummyBloodGroupClassifier()
        
    return BloodGroupClassifier(model_type=model_type, n_estimators=n_estimators)


class DummyBloodGroupClassifier:
    """
    Dummy classifier for demonstration when scikit-learn is not available.
    """
    def __init__(self):
        self.blood_groups = BLOOD_GROUPS
        
    def predict(self, features):
        """Return random predictions as a fallback"""
        return np.random.choice(self.blood_groups, size=len(features))
        
    def predict_proba(self, features):
        """Return random probabilities as a fallback"""
        n_samples = len(features)
        n_classes = len(self.blood_groups)
        probs = np.random.random((n_samples, n_classes))
        # Normalize to make each row sum to 1
        return probs / probs.sum(axis=1, keepdims=True)
        
    def save_model(self, model_path=None):
        """Dummy save function"""
        print("Warning: Using dummy classifier. Model saving not supported.")
        return None
        
    @classmethod
    def load_model(cls, model_path=None):
        """Dummy load function"""
        print("Warning: Using dummy classifier. Model loading not supported.")
        return DummyBloodGroupClassifier()


if __name__ == "__main__":
    # Test the module if run directly
    print("Blood Group Classifier module loaded successfully")
    print(f"scikit-learn available: {SKLEARN_AVAILABLE}")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    
    # Create a classifier
    classifier = create_blood_classifier()
    print(f"Created classifier of type: {type(classifier).__name__}")
    
    # Example of generating a dummy dataset
    if SKLEARN_AVAILABLE:
        # Generate random features
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        X = np.random.rand(n_samples, n_features)
        
        # Generate random labels
        y = np.random.choice(BLOOD_GROUPS, size=n_samples)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train the model
        print("\nTraining with random data (for demonstration only)...")
        classifier.fit(X_train, y_train)
        
        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)
        print(f"\nAccuracy on test set: {metrics['accuracy']:.4f}")
        
        # Make a prediction
        sample = X_test[0:1]
        pred = classifier.predict(sample)[0]
        print(f"\nSample prediction: {pred}")
        
        if hasattr(classifier, 'predict_proba'):
            probs = classifier.predict_proba(sample)[0]
            for i, blood_group in enumerate(BLOOD_GROUPS):
                if i < len(probs):
                    print(f"  {blood_group}: {probs[i]:.4f}") 