#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prediction module for Blood Group Classification.

This module provides functions to apply trained blood group classification models
to new blood sample images for prediction and visualization.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the blood classifier and preprocessing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from Models.blood_classifier import (
        BloodGroupClassifier, create_blood_classifier, 
        BLOOD_GROUPS, DEFAULT_MODEL_DIR
    )
    from Preprocessing.preprocess import (
        load_image, preprocess_image, extract_features, 
        visualize_preprocessing, DEFAULT_SIZE
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing required modules: {str(e)}")
    MODULES_AVAILABLE = False

# Constants for prediction
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, 'blood_group_model.pkl')
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')


class BloodGroupPredictor:
    """
    Predictor for blood group detection using a trained model.
    Handles loading models, preprocessing images, and making predictions.
    """
    
    def __init__(self, model_path=None, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        """
        Initialize the blood group predictor.
        
        Args:
            model_path (str, optional): Path to the trained model file
            confidence_threshold (float): Minimum confidence threshold for predictions
        """
        self.model = None
        self.model_path = model_path if model_path else DEFAULT_MODEL_PATH
        self.confidence_threshold = confidence_threshold
        self.metadata = None
        self.feature_selector = None
        
        # Load the model
        self._load_model()
        
    def _load_model(self):
        """Load the trained model and associated components."""
        if not MODULES_AVAILABLE:
            logger.error("Required modules not available. Cannot load model.")
            return
            
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return
                
            # Load the model
            logger.info(f"Loading model from: {self.model_path}")
            self.model = BloodGroupClassifier.load_model(self.model_path)
            
            # Load metadata if available
            metadata_path = os.path.splitext(self.model_path)[0] + "_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded model metadata from: {metadata_path}")
                
            # Load feature selector if available
            selector_path = os.path.splitext(self.model_path)[0] + "_selector.pkl"
            if os.path.exists(selector_path):
                import pickle
                with open(selector_path, 'rb') as f:
                    self.feature_selector = pickle.load(f)
                logger.info(f"Loaded feature selector from: {selector_path}")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def preprocess_sample(self, image_path, visualize=False, save_vis_path=None):
        """
        Preprocess a blood sample image for prediction.
        
        Args:
            image_path (str): Path to the blood sample image
            visualize (bool): Whether to visualize preprocessing steps
            save_vis_path (str, optional): Path to save visualization
            
        Returns:
            tuple: Preprocessed image and extracted features
        """
        if not MODULES_AVAILABLE:
            logger.error("Required modules not available. Cannot preprocess sample.")
            return None, None
            
        try:
            # Visualize preprocessing if requested
            if visualize:
                vis_result = visualize_preprocessing(image_path, save_path=save_vis_path)
                if vis_result is False:
                    logger.warning("Visualization failed")
            
            # Load and preprocess the image
            logger.info(f"Preprocessing image: {image_path}")
            preprocessed_img = preprocess_image(
                image_path, 
                target_size=DEFAULT_SIZE,
                normalize=True,
                segment=True,
                use_enhanced=True
            )
            
            # Extract features from the preprocessed image
            logger.info("Extracting features")
            features = extract_features(preprocessed_img)
            
            # Apply feature selection if available
            if self.feature_selector is not None and self.metadata is not None:
                logger.info("Applying feature selection")
                features = self.feature_selector.transform(features.reshape(1, -1))
            
            return preprocessed_img, features
            
        except Exception as e:
            logger.error(f"Error preprocessing sample: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def predict(self, image_path, visualize=False, save_results=False, results_dir=DEFAULT_RESULTS_DIR):
        """
        Predict blood group for a blood sample image.
        
        Args:
            image_path (str): Path to the blood sample image
            visualize (bool): Whether to visualize results
            save_results (bool): Whether to save prediction results
            results_dir (str): Directory to save results
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            logger.error("No model loaded. Cannot make predictions.")
            return None
            
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Preprocess the image
        vis_path = None
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            vis_path = os.path.join(results_dir, f"preprocess_{timestamp}.png")
            
        img, features = self.preprocess_sample(
            image_path, 
            visualize=visualize,
            save_vis_path=vis_path if save_results else None
        )
        
        if img is None or features is None:
            logger.error("Failed to preprocess image or extract features")
            return None
            
        try:
            # Make prediction
            logger.info("Making prediction")
            
            # Reshape features if needed
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
                
            # Get predicted class
            predicted_class = self.model.predict(features)[0]
            
            # Get class probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                
            # Create prediction result
            result = {
                'image_path': image_path,
                'predicted_blood_group': predicted_class,
                'confidence': float(max(probabilities) if probabilities is not None else 1.0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add class probabilities if available
            if probabilities is not None:
                result['class_probabilities'] = {
                    bg: float(prob) for bg, prob in zip(BLOOD_GROUPS, probabilities)
                    if bg in BLOOD_GROUPS and prob >= 0.01  # Only include non-negligible probabilities
                }
            
            # Check if prediction confidence exceeds threshold
            if result['confidence'] < self.confidence_threshold:
                logger.warning(f"Low confidence prediction: {result['confidence']:.4f} < {self.confidence_threshold}")
                result['low_confidence'] = True
            
            # Visualize results if requested
            if visualize:
                self._visualize_prediction(img, result)
                
            # Save results if requested
            if save_results:
                self._save_prediction_results(img, result, results_dir, timestamp)
                
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_batch(self, image_paths, save_results=False, results_dir=DEFAULT_RESULTS_DIR):
        """
        Make predictions for multiple blood sample images.
        
        Args:
            image_paths (list): List of paths to blood sample images
            save_results (bool): Whether to save prediction results
            results_dir (str): Directory to save results
            
        Returns:
            list: Prediction results for each image
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            result = self.predict(image_path, visualize=False, save_results=save_results, results_dir=results_dir)
            if result is not None:
                results.append(result)
        
        # Save batch results if requested
        if save_results and results:
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_results_path = os.path.join(results_dir, f"batch_results_{timestamp}.json")
            with open(batch_results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Batch results saved to: {batch_results_path}")
        
        return results
    
    def _visualize_prediction(self, img, result):
        """
        Visualize prediction results.
        
        Args:
            img (numpy.ndarray): Preprocessed image
            result (dict): Prediction result
        """
        plt.figure(figsize=(12, 8))
        
        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Preprocessed Blood Sample")
        plt.axis('off')
        
        # Display prediction results
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.text(0.5, 0.9, "Blood Group Prediction", 
                fontsize=14, fontweight='bold', ha='center')
        
        plt.text(0.5, 0.8, f"Predicted: {result['predicted_blood_group']}", 
                fontsize=18, fontweight='bold', ha='center')
        
        plt.text(0.5, 0.7, f"Confidence: {result['confidence']:.2f}", 
                fontsize=12, ha='center', 
                color='red' if result.get('low_confidence', False) else 'black')
        
        # Display class probabilities if available
        if 'class_probabilities' in result:
            y_pos = 0.6
            plt.text(0.5, y_pos, "Class Probabilities:", fontsize=10, ha='center')
            y_pos -= 0.05
            
            # Sort probabilities by value in descending order
            sorted_probs = sorted(
                result['class_probabilities'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for blood_group, prob in sorted_probs:
                y_pos -= 0.04
                plt.text(0.5, y_pos, f"{blood_group}: {prob:.4f}", 
                        fontsize=9, ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def _save_prediction_results(self, img, result, results_dir, timestamp):
        """
        Save prediction results.
        
        Args:
            img (numpy.ndarray): Preprocessed image
            result (dict): Prediction result
            results_dir (str): Directory to save results
            timestamp (str): Timestamp for unique filenames
        """
        os.makedirs(results_dir, exist_ok=True)
        
        # Save prediction result as JSON
        result_path = os.path.join(results_dir, f"prediction_{timestamp}.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Prediction result saved to: {result_path}")
        
        # Create and save visualization
        fig = plt.figure(figsize=(12, 8))
        
        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Preprocessed Blood Sample")
        plt.axis('off')
        
        # Display prediction results
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.text(0.5, 0.9, "Blood Group Prediction", 
                fontsize=14, fontweight='bold', ha='center')
        
        plt.text(0.5, 0.8, f"Predicted: {result['predicted_blood_group']}", 
                fontsize=18, fontweight='bold', ha='center')
        
        plt.text(0.5, 0.7, f"Confidence: {result['confidence']:.2f}", 
                fontsize=12, ha='center', 
                color='red' if result.get('low_confidence', False) else 'black')
        
        # Display class probabilities if available
        if 'class_probabilities' in result:
            y_pos = 0.6
            plt.text(0.5, y_pos, "Class Probabilities:", fontsize=10, ha='center')
            y_pos -= 0.05
            
            # Sort probabilities by value in descending order
            sorted_probs = sorted(
                result['class_probabilities'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for blood_group, prob in sorted_probs:
                y_pos -= 0.04
                plt.text(0.5, y_pos, f"{blood_group}: {prob:.4f}", 
                        fontsize=9, ha='center')
        
        plt.tight_layout()
        
        # Save visualization
        vis_path = os.path.join(results_dir, f"prediction_vis_{timestamp}.png")
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Prediction visualization saved to: {vis_path}")


def predict_sample(image_path, model_path=None, visualize=True, save_results=False,
                  confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Convenience function to predict blood group for a single sample.
    
    Args:
        image_path (str): Path to the blood sample image
        model_path (str, optional): Path to the trained model file
        visualize (bool): Whether to visualize results
        save_results (bool): Whether to save prediction results
        confidence_threshold (float): Minimum confidence threshold for predictions
        
    Returns:
        dict: Prediction result
    """
    predictor = BloodGroupPredictor(
        model_path=model_path,
        confidence_threshold=confidence_threshold
    )
    
    return predictor.predict(
        image_path,
        visualize=visualize,
        save_results=save_results
    )


def predict_directory(directory_path, model_path=None, pattern="*.jpg", 
                     recursive=False, save_results=True):
    """
    Predict blood groups for all blood sample images in a directory.
    
    Args:
        directory_path (str): Path to directory containing blood sample images
        model_path (str, optional): Path to the trained model file
        pattern (str): Glob pattern for image files
        recursive (bool): Whether to search for images recursively
        save_results (bool): Whether to save prediction results
        
    Returns:
        dict: Summary of prediction results
    """
    # Find image files matching the pattern
    if recursive:
        image_paths = list(Path(directory_path).rglob(pattern))
    else:
        image_paths = list(Path(directory_path).glob(pattern))
    
    image_paths = [str(p) for p in image_paths]
    
    if not image_paths:
        logger.warning(f"No images found in {directory_path} matching pattern {pattern}")
        return None
    
    logger.info(f"Found {len(image_paths)} images in {directory_path}")
    
    # Create predictor
    predictor = BloodGroupPredictor(model_path=model_path)
    
    # Process all images
    results = predictor.predict_batch(image_paths, save_results=save_results)
    
    # Create summary
    if results:
        summary = {
            'total_samples': len(results),
            'directory': directory_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Count predictions by blood group
        blood_group_counts = {}
        for result in results:
            bg = result['predicted_blood_group']
            blood_group_counts[bg] = blood_group_counts.get(bg, 0) + 1
        
        summary['blood_group_counts'] = blood_group_counts
        
        # Count low confidence predictions
        low_confidence_count = sum(1 for result in results if result.get('low_confidence', False))
        summary['low_confidence_count'] = low_confidence_count
        
        logger.info(f"Processed {summary['total_samples']} images")
        logger.info(f"Blood group distribution: {blood_group_counts}")
        
        return summary
    
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict blood groups from blood sample images")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Predict single sample
    sample_parser = subparsers.add_parser('sample', help='Predict a single blood sample')
    sample_parser.add_argument('image_path', help='Path to blood sample image')
    sample_parser.add_argument('--model', help='Path to trained model file')
    sample_parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    sample_parser.add_argument('--save', action='store_true', help='Save prediction results')
    sample_parser.add_argument('--threshold', type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                             help='Confidence threshold for predictions')
    
    # Predict directory of samples
    dir_parser = subparsers.add_parser('directory', help='Predict blood samples in a directory')
    dir_parser.add_argument('directory_path', help='Path to directory containing blood sample images')
    dir_parser.add_argument('--model', help='Path to trained model file')
    dir_parser.add_argument('--pattern', default="*.jpg", help='Glob pattern for image files')
    dir_parser.add_argument('--recursive', action='store_true', help='Search for images recursively')
    dir_parser.add_argument('--no-save', action='store_true', help='Do not save prediction results')
    
    args = parser.parse_args()
    
    if args.command == 'sample':
        # Predict single sample
        result = predict_sample(
            args.image_path,
            model_path=args.model,
            visualize=not args.no_vis,
            save_results=args.save,
            confidence_threshold=args.threshold
        )
        
        if result:
            print(f"Predicted blood group: {result['predicted_blood_group']}")
            print(f"Confidence: {result['confidence']:.4f}")
            
    elif args.command == 'directory':
        # Predict directory of samples
        summary = predict_directory(
            args.directory_path,
            model_path=args.model,
            pattern=args.pattern,
            recursive=args.recursive,
            save_results=not args.no_save
        )
        
        if summary:
            print(f"Processed {summary['total_samples']} images")
            print("Blood group distribution:")
            for bg, count in summary['blood_group_counts'].items():
                print(f"  {bg}: {count} samples")
            
            if summary['low_confidence_count'] > 0:
                print(f"Low confidence predictions: {summary['low_confidence_count']}")
    else:
        parser.print_help() 