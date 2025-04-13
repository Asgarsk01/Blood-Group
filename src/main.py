#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Blood Group Classification System.

This script provides a comprehensive command-line interface to access
all the functionalities of the blood group classification system:
preprocessing, training, prediction, and visualization.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to path for imports
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import modules with try-except to handle potential import errors
try:
    # Import preprocessing module
    from Preprocessing.preprocess import (
        load_image, preprocess_image, extract_features, 
        segment_blood_cells, visualize_preprocessing,
        augment_samples, DEFAULT_SIZE
    )
    
    # Import model modules
    from Models.blood_classifier import (
        create_blood_classifier, BloodGroupClassifier, 
        BLOOD_GROUPS, DEFAULT_MODEL_DIR
    )
    
    from Models.train_model import (
        load_features, select_features, train_model, evaluate_model, 
        cross_validate, optimize_hyperparameters, save_training_pipeline,
        main as train_main
    )
    
    from Models.predict import (
        predict_sample, predict_directory, BloodGroupPredictor
    )
    
    from Models.visualize import (
        visualize_confusion_matrix, visualize_feature_importance,
        visualize_prediction_distribution, visualize_confidence_distribution,
        visualize_metrics_by_blood_group, visualize_results_dashboard
    )
    
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing modules: {str(e)}")
    MODULES_AVAILABLE = False

# Default paths
DEFAULT_DATA_DIR = os.path.join(src_dir, 'data')
DEFAULT_FEATURES_DIR = os.path.join(DEFAULT_DATA_DIR, 'features')
DEFAULT_IMAGES_DIR = os.path.join(DEFAULT_DATA_DIR, 'images')
DEFAULT_RESULTS_DIR = os.path.join(src_dir, 'results')


def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
    os.makedirs(DEFAULT_FEATURES_DIR, exist_ok=True)
    os.makedirs(DEFAULT_IMAGES_DIR, exist_ok=True)
    os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
    os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)


def check_system_status():
    """Check system status and dependencies."""
    status = {
        "system_status": "operational",
        "dependencies": {}
    }
    
    # Check required dependencies
    try:
        import numpy
        status["dependencies"]["numpy"] = True
    except ImportError:
        status["dependencies"]["numpy"] = False
        status["system_status"] = "degraded"
    
    try:
        import pandas
        status["dependencies"]["pandas"] = True
    except ImportError:
        status["dependencies"]["pandas"] = False
        status["system_status"] = "degraded"
    
    try:
        import matplotlib
        status["dependencies"]["matplotlib"] = True
    except ImportError:
        status["dependencies"]["matplotlib"] = False
        status["system_status"] = "degraded"
    
    try:
        import sklearn
        status["dependencies"]["scikit-learn"] = True
    except ImportError:
        status["dependencies"]["scikit-learn"] = False
        status["system_status"] = "degraded"
    
    try:
        import cv2
        status["dependencies"]["opencv"] = True
    except ImportError:
        status["dependencies"]["opencv"] = False
        status["system_status"] = "degraded"
    
    try:
        import tensorflow
        status["dependencies"]["tensorflow"] = True
    except ImportError:
        status["dependencies"]["tensorflow"] = False
    
    try:
        import xgboost
        status["dependencies"]["xgboost"] = True
    except ImportError:
        status["dependencies"]["xgboost"] = False
    
    # Check module imports
    status["modules_available"] = MODULES_AVAILABLE
    
    # Check directories
    create_directories()
    status["data_dir_exists"] = os.path.exists(DEFAULT_DATA_DIR)
    status["model_dir_exists"] = os.path.exists(DEFAULT_MODEL_DIR)
    
    # Check for trained models
    status["trained_models"] = []
    if os.path.exists(DEFAULT_MODEL_DIR):
        model_files = list(Path(DEFAULT_MODEL_DIR).glob("*.pkl"))
        status["trained_models"] = [os.path.basename(str(f)) for f in model_files]
    
    return status


def print_status_report(status):
    """Print a formatted status report."""
    print("\n===== BLOOD GROUP CLASSIFICATION SYSTEM STATUS =====")
    print(f"System Status: {status['system_status'].upper()}")
    print("\nDependencies:")
    for dep, installed in status["dependencies"].items():
        print(f"  {dep}: {'✓' if installed else '✗'}")
    
    print("\nSystem Checks:")
    print(f"  Modules Available: {'✓' if status['modules_available'] else '✗'}")
    print(f"  Data Directory: {'✓' if status['data_dir_exists'] else '✗'}")
    print(f"  Model Directory: {'✓' if status['model_dir_exists'] else '✗'}")
    
    print("\nTrained Models:")
    if status["trained_models"]:
        for model in status["trained_models"]:
            print(f"  - {model}")
    else:
        print("  No trained models found")
    
    print("\n=================================================")


def preprocess_cmd(args):
    """Handle preprocessing command."""
    if not MODULES_AVAILABLE:
        logger.error("Required modules not available. Cannot perform preprocessing.")
        return
    
    if args.visualize:
        # Visualize preprocessing steps
        logger.info(f"Visualizing preprocessing steps for: {args.image_path}")
        save_path = None
        if args.output:
            save_path = args.output
        visualize_preprocessing(args.image_path, save_path=save_path)
        return
    
    # Process the image
    logger.info(f"Preprocessing image: {args.image_path}")
    preprocessed = preprocess_image(
        args.image_path,
        target_size=(args.width, args.height) if args.width and args.height else DEFAULT_SIZE,
        normalize=not args.no_normalize,
        segment=args.segment,
        use_enhanced=not args.no_enhance
    )
    
    # Extract features if requested
    if args.extract_features:
        features = extract_features(preprocessed)
        logger.info(f"Extracted {len(features)} features")
        
        # Save features if output path provided
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(features, f, indent=2)
            logger.info(f"Features saved to: {args.output}")
    
    logger.info("Preprocessing completed successfully")


def train_cmd(args):
    """Handle training command."""
    if not MODULES_AVAILABLE:
        logger.error("Required modules not available. Cannot perform training.")
        return
    
    # Call the train_main function from train_model module
    train_args = {
        "features_path": args.features,
        "model_type": args.model_type,
        "n_estimators": args.n_estimators,
        "use_feature_selection": not args.no_feature_selection,
        "k_features": args.k_features,
        "test_size": args.test_size,
        "save_model": not args.no_save,
        "cross_val": not args.no_cross_val,
        "optimize_params": args.optimize,
        "random_state": args.random_seed
    }
    
    logger.info("Starting model training with the following parameters:")
    for key, value in train_args.items():
        logger.info(f"  {key}: {value}")
    
    try:
        classifier, metrics = train_main(**train_args)
        logger.info(f"Training completed. Final model accuracy: {metrics['accuracy']:.4f}")
        
        # Save visualization of results if requested
        if args.visualize and metrics:
            os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_dir = os.path.join(DEFAULT_RESULTS_DIR, f"training_vis_{timestamp}")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Save confusion matrix
            conf_matrix = metrics.get('confusion_matrix')
            if conf_matrix:
                cm_path = os.path.join(vis_dir, "confusion_matrix.png")
                visualize_confusion_matrix(
                    np.array(conf_matrix),
                    title=f"Confusion Matrix - {args.model_type.upper()} Model",
                    save_path=cm_path
                )
            
            # Save metrics by blood group
            if 'classification_report' in metrics:
                metrics_path = os.path.join(vis_dir, "metrics_by_blood_group.png")
                visualize_metrics_by_blood_group(
                    metrics['classification_report'],
                    title=f"Performance Metrics by Blood Group - {args.model_type.upper()} Model",
                    save_path=metrics_path
                )
            
            logger.info(f"Training visualizations saved to: {vis_dir}")
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()


def predict_cmd(args):
    """Handle prediction command."""
    if not MODULES_AVAILABLE:
        logger.error("Required modules not available. Cannot perform prediction.")
        return
    
    # Set model path
    model_path = args.model
    
    # Process a single sample
    if args.command == 'sample':
        result = predict_sample(
            args.image_path,
            model_path=model_path,
            visualize=not args.no_vis,
            save_results=args.save,
            confidence_threshold=args.threshold
        )
        
        if result:
            print(f"\nPredicted blood group: {result['predicted_blood_group']}")
            print(f"Confidence: {result['confidence']:.4f}")
            
            # Print confidence scores for each blood group if available
            if 'class_probabilities' in result:
                print("\nClass probabilities:")
                for bg, prob in sorted(result['class_probabilities'].items(), 
                                      key=lambda x: x[1], reverse=True):
                    print(f"  {bg}: {prob:.4f}")
    
    # Process a directory of images
    elif args.command == 'directory':
        summary = predict_directory(
            args.directory_path,
            model_path=model_path,
            pattern=args.pattern,
            recursive=args.recursive,
            save_results=not args.no_save
        )
        
        if summary:
            print(f"\nProcessed {summary['total_samples']} images")
            print("\nBlood group distribution:")
            for bg, count in sorted(summary['blood_group_counts'].items()):
                print(f"  {bg}: {count} samples")
            
            if summary.get('low_confidence_count', 0) > 0:
                print(f"\nLow confidence predictions: {summary['low_confidence_count']}")
                
            # Visualize distribution if requested
            if args.visualize:
                os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                distribution_path = os.path.join(DEFAULT_RESULTS_DIR, f"distribution_{timestamp}.png")
                
                # Create list of predictions for visualization
                prediction_list = []
                for _ in range(summary['total_samples']):
                    for bg, count in summary['blood_group_counts'].items():
                        prediction_list.extend([bg] * count)
                
                visualize_prediction_distribution(
                    prediction_list,
                    title="Blood Group Distribution in Dataset",
                    save_path=distribution_path
                )


def visualize_cmd(args):
    """Handle visualization command."""
    if not MODULES_AVAILABLE:
        logger.error("Required modules not available. Cannot create visualizations.")
        return
    
    if args.dashboard:
        # Create a comprehensive dashboard from metrics file
        if not os.path.exists(args.dashboard):
            logger.error(f"Metrics file not found: {args.dashboard}")
            return
        
        output_dir = args.output if args.output else None
        dashboard_dir = visualize_results_dashboard(args.dashboard, output_dir)
        logger.info(f"Dashboard created in: {dashboard_dir}")
    
    elif args.confusion_matrix:
        # Visualize a confusion matrix
        try:
            import json
            with open(args.confusion_matrix, 'r') as f:
                data = json.load(f)
            
            if 'confusion_matrix' in data:
                cm = np.array(data['confusion_matrix'])
                save_path = args.output if args.output else None
                
                visualize_confusion_matrix(
                    cm,
                    title=args.title or "Confusion Matrix",
                    normalize=args.normalize,
                    save_path=save_path
                )
            else:
                logger.error("No confusion matrix found in the provided file")
        except Exception as e:
            logger.error(f"Error visualizing confusion matrix: {str(e)}")
    
    elif args.feature_importance:
        # Visualize feature importance
        try:
            import json
            with open(args.feature_importance, 'r') as f:
                data = json.load(f)
            
            if 'feature_importance' in data and 'feature_names' in data:
                importance = np.array(data['feature_importance'])
                names = data['feature_names']
                save_path = args.output if args.output else None
                
                visualize_feature_importance(
                    importance,
                    feature_names=names,
                    top_n=args.top_n or 20,
                    title=args.title or "Feature Importance",
                    save_path=save_path
                )
            else:
                logger.error("Feature importance or feature names not found in the provided file")
        except Exception as e:
            logger.error(f"Error visualizing feature importance: {str(e)}")
    
    else:
        logger.error("No visualization type specified. Use --dashboard, --confusion-matrix, or --feature-importance.")


def setup_parser():
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Blood Group Classification System")
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check system status')
    
    # Preprocessing command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess blood sample images')
    preprocess_parser.add_argument('image_path', help='Path to the image file')
    preprocess_parser.add_argument('--width', type=int, help='Target width for resizing')
    preprocess_parser.add_argument('--height', type=int, help='Target height for resizing')
    preprocess_parser.add_argument('--no-normalize', action='store_true', help='Disable normalization')
    preprocess_parser.add_argument('--segment', action='store_true', help='Perform blood cell segmentation')
    preprocess_parser.add_argument('--no-enhance', action='store_true', help='Disable image enhancement')
    preprocess_parser.add_argument('--extract-features', action='store_true', help='Extract features from the preprocessed image')
    preprocess_parser.add_argument('--visualize', action='store_true', help='Visualize preprocessing steps')
    preprocess_parser.add_argument('--output', help='Output path for features or visualization')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a blood group classifier')
    train_parser.add_argument('--features', help='Path to features file (CSV or JSON)')
    train_parser.add_argument('--model-type', default='ensemble', 
                            choices=['rf', 'svm', 'xgb', 'mlp', 'ensemble'],
                            help='Type of classifier to use')
    train_parser.add_argument('--n-estimators', type=int, default=100,
                            help='Number of estimators for ensemble methods')
    train_parser.add_argument('--no-feature-selection', action='store_true',
                            help='Disable feature selection')
    train_parser.add_argument('--k-features', type=int, default=20,
                            help='Number of top features to select')
    train_parser.add_argument('--test-size', type=float, default=0.2,
                            help='Proportion of data to use for testing')
    train_parser.add_argument('--no-save', action='store_true',
                            help="Don't save the trained model")
    train_parser.add_argument('--no-cross-val', action='store_true',
                            help="Don't perform cross-validation")
    train_parser.add_argument('--optimize', action='store_true',
                            help='Optimize hyperparameters')
    train_parser.add_argument('--random-seed', type=int, default=42,
                            help='Random seed for reproducibility')
    train_parser.add_argument('--visualize', action='store_true',
                            help='Visualize training results')
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Predict blood groups from samples')
    predict_subparsers = predict_parser.add_subparsers(dest='command', help='Prediction command')
    
    # Single sample prediction
    sample_parser = predict_subparsers.add_parser('sample', help='Predict a single blood sample')
    sample_parser.add_argument('image_path', help='Path to blood sample image')
    sample_parser.add_argument('--model', help='Path to trained model file')
    sample_parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    sample_parser.add_argument('--save', action='store_true', help='Save prediction results')
    sample_parser.add_argument('--threshold', type=float, default=0.6,
                             help='Confidence threshold for predictions')
    
    # Directory prediction
    dir_parser = predict_subparsers.add_parser('directory', help='Predict blood samples in a directory')
    dir_parser.add_argument('directory_path', help='Path to directory containing blood sample images')
    dir_parser.add_argument('--model', help='Path to trained model file')
    dir_parser.add_argument('--pattern', default='*.jpg', help='Glob pattern for image files')
    dir_parser.add_argument('--recursive', action='store_true', help='Search for images recursively')
    dir_parser.add_argument('--no-save', action='store_true', help='Do not save prediction results')
    dir_parser.add_argument('--visualize', action='store_true', help='Visualize prediction distribution')
    
    # Visualization command
    visualize_parser = subparsers.add_parser('visualize', help='Create visualizations')
    visualize_parser.add_argument('--dashboard', help='Path to metrics JSON file for dashboard creation')
    visualize_parser.add_argument('--confusion-matrix', help='Path to metrics JSON file with confusion matrix')
    visualize_parser.add_argument('--normalize', action='store_true', help='Normalize confusion matrix')
    visualize_parser.add_argument('--feature-importance', help='Path to JSON file with feature importance data')
    visualize_parser.add_argument('--top-n', type=int, help='Number of top features to display')
    visualize_parser.add_argument('--title', help='Title for the visualization')
    visualize_parser.add_argument('--output', help='Output path for the visualization')
    
    return parser


def main():
    """Main entry point for the script."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Create necessary directories
    create_directories()
    
    # Handle commands
    if args.action == 'status':
        status = check_system_status()
        print_status_report(status)
    
    elif args.action == 'preprocess':
        preprocess_cmd(args)
    
    elif args.action == 'train':
        train_cmd(args)
    
    elif args.action == 'predict':
        predict_cmd(args)
    
    elif args.action == 'visualize':
        visualize_cmd(args)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 