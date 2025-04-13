#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization module for Blood Group Classification results.

This module provides functions to visualize model performance, predictions,
and blood sample features for better understanding and analysis.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from datetime import datetime

# Set better visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')

# Import the blood classifier module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from Models.blood_classifier import BLOOD_GROUPS
except ImportError:
    BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Constants
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualization_results')


def create_results_directory(subdir=None):
    """
    Create a directory for saving visualization results.
    
    Args:
        subdir (str, optional): Subdirectory name
        
    Returns:
        str: Path to the created directory
    """
    if subdir:
        results_dir = os.path.join(DEFAULT_RESULTS_DIR, subdir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(DEFAULT_RESULTS_DIR, f"viz_{timestamp}")
    
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def visualize_confusion_matrix(confusion_matrix, class_names=None, title="Confusion Matrix",
                              normalize=False, save_path=None, figsize=(10, 8)):
    """
    Visualize a confusion matrix with annotations.
    
    Args:
        confusion_matrix (numpy.ndarray): Confusion matrix to visualize
        class_names (list, optional): Names of the classes
        title (str): Title for the plot
        normalize (bool): Whether to normalize the confusion matrix
        save_path (str, optional): Path to save the visualization
        figsize (tuple): Figure size (width, height)
    """
    if class_names is None:
        # Use blood groups that appear in the matrix
        class_names = BLOOD_GROUPS[:len(confusion_matrix)]
    
    if normalize:
        # Normalize by row (true labels)
        row_sums = confusion_matrix.sum(axis=1)
        confusion_matrix = confusion_matrix / row_sums[:, np.newaxis]
        fmt = '.2f'
        vmax = 1.0
    else:
        fmt = 'd'
        vmax = np.max(confusion_matrix) * 0.9
    
    plt.figure(figsize=figsize)
    
    # Create the heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt=fmt,
               xticklabels=class_names, yticklabels=class_names,
               cmap='Blues', vmin=0, vmax=vmax)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix visualization saved to: {save_path}")
    
    plt.show()


def visualize_feature_importance(feature_importance, feature_names=None, top_n=20,
                                title="Feature Importance", save_path=None, figsize=(12, 8)):
    """
    Visualize feature importance from a trained model.
    
    Args:
        feature_importance (numpy.ndarray): Feature importance values
        feature_names (list, optional): Names of the features
        top_n (int): Number of top features to display
        title (str): Title for the plot
        save_path (str, optional): Path to save the visualization
        figsize (tuple): Figure size (width, height)
    """
    # Create default feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
    
    # Sort features by importance
    indices = np.argsort(feature_importance)[::-1]
    
    # Get top N features
    top_indices = indices[:min(top_n, len(indices))]
    top_importance = feature_importance[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=figsize)
    
    # Create the bar plot with a colormap
    bars = plt.barh(range(len(top_indices)), top_importance, align='center',
                  color=plt.cm.viridis(np.linspace(0, 1, len(top_indices))))
    
    # Add feature names and values
    for i, (val, name) in enumerate(zip(top_importance, top_names)):
        plt.text(val * 1.02, i, f"{val:.4f}", va='center', fontsize=9)
    
    plt.yticks(range(len(top_indices)), top_names)
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()  # Display the highest importance at the top
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance visualization saved to: {save_path}")
    
    plt.show()


def visualize_prediction_distribution(predictions, title="Blood Group Distribution",
                                     save_path=None, figsize=(10, 6)):
    """
    Visualize the distribution of predicted blood groups.
    
    Args:
        predictions (list): List of prediction results or blood group labels
        title (str): Title for the plot
        save_path (str, optional): Path to save the visualization
        figsize (tuple): Figure size (width, height)
    """
    # Count blood groups
    if isinstance(predictions[0], dict):
        # Extract blood group from prediction results
        blood_groups = [p.get('predicted_blood_group') for p in predictions]
    else:
        # Assume list of blood group labels
        blood_groups = predictions
    
    # Count occurrences
    counts = Counter(blood_groups)
    
    # Sort by blood group
    sorted_items = sorted(counts.items(), key=lambda x: BLOOD_GROUPS.index(x[0]) 
                         if x[0] in BLOOD_GROUPS else 999)
    labels, values = zip(*sorted_items) if sorted_items else ([], [])
    
    plt.figure(figsize=figsize)
    
    # Create the bar plot
    bars = plt.bar(labels, values, color=plt.cm.tab10(np.linspace(0, 1, len(labels))))
    
    # Add count annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.0f}', ha='center', va='bottom')
    
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction distribution visualization saved to: {save_path}")
    
    plt.show()


def visualize_confidence_distribution(predictions, bins=10, title="Prediction Confidence Distribution",
                                     save_path=None, figsize=(10, 6)):
    """
    Visualize the distribution of prediction confidence scores.
    
    Args:
        predictions (list): List of prediction results
        bins (int): Number of bins for the histogram
        title (str): Title for the plot
        save_path (str, optional): Path to save the visualization
        figsize (tuple): Figure size (width, height)
    """
    # Extract confidence values
    confidence_values = [p.get('confidence', 0) for p in predictions]
    
    plt.figure(figsize=figsize)
    
    # Create the histogram
    n, bins, patches = plt.hist(confidence_values, bins=bins, alpha=0.7, 
                               color='skyblue', edgecolor='black')
    
    # Add mean and median lines
    mean_conf = np.mean(confidence_values)
    median_conf = np.median(confidence_values)
    
    plt.axvline(mean_conf, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_conf:.2f}')
    plt.axvline(median_conf, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_conf:.2f}')
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence distribution visualization saved to: {save_path}")
    
    plt.show()


def visualize_model_comparison(model_metrics, metric='accuracy', title="Model Performance Comparison",
                              save_path=None, figsize=(12, 6)):
    """
    Visualize performance comparison between different models.
    
    Args:
        model_metrics (dict): Dictionary mapping model names to their metrics
        metric (str): Metric to compare (e.g., 'accuracy', 'f1_score')
        title (str): Title for the plot
        save_path (str, optional): Path to save the visualization
        figsize (tuple): Figure size (width, height)
    """
    # Extract the specified metric from each model
    models = []
    values = []
    
    for model_name, metrics in model_metrics.items():
        models.append(model_name)
        values.append(metrics.get(metric, 0))
    
    # Sort by performance
    sorted_indices = np.argsort(values)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    
    plt.figure(figsize=figsize)
    
    # Create the bar plot
    bars = plt.bar(sorted_models, sorted_values, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
    
    # Add value annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.ylim(0, max(values) * 1.1)
    plt.ylabel(metric.capitalize())
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison visualization saved to: {save_path}")
    
    plt.show()


def visualize_cross_validation_results(cv_results, title="Cross-Validation Performance",
                                      save_path=None, figsize=(10, 6)):
    """
    Visualize cross-validation results.
    
    Args:
        cv_results (dict): Dictionary with cross-validation results
        title (str): Title for the plot
        save_path (str, optional): Path to save the visualization
        figsize (tuple): Figure size (width, height)
    """
    # Extract accuracy for each fold
    fold_accuracies = cv_results.get('fold_accuracies', [])
    mean_accuracy = cv_results.get('mean_accuracy', 0)
    std_accuracy = cv_results.get('std_accuracy', 0)
    
    plt.figure(figsize=figsize)
    
    # Create the line plot
    folds = range(1, len(fold_accuracies) + 1)
    plt.plot(folds, fold_accuracies, 'o-', color='royalblue', label='Fold Accuracy')
    
    # Add mean accuracy line
    plt.axhline(mean_accuracy, color='red', linestyle='dashed', linewidth=2, 
               label=f'Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}')
    
    # Add confidence interval
    plt.fill_between(folds, mean_accuracy - std_accuracy, mean_accuracy + std_accuracy,
                    alpha=0.2, color='red')
    
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(folds)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cross-validation results visualization saved to: {save_path}")
    
    plt.show()


def visualize_learning_curve(train_sizes, train_scores, test_scores, title="Learning Curve",
                            save_path=None, figsize=(10, 6)):
    """
    Visualize learning curve showing model performance versus training set size.
    
    Args:
        train_sizes (list): Training set sizes
        train_scores (list): Scores on training sets
        test_scores (list): Scores on test sets
        title (str): Title for the plot
        save_path (str, optional): Path to save the visualization
        figsize (tuple): Figure size (width, height)
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=figsize)
    
    # Plot mean accuracy
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation score')
    
    # Plot standard deviation bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve visualization saved to: {save_path}")
    
    plt.show()


def visualize_sample_predictions(images, predictions, max_samples=16, title="Sample Predictions",
                               save_path=None, figsize=(15, 12)):
    """
    Visualize a grid of sample images with their predicted blood groups.
    
    Args:
        images (list): List of sample images as numpy arrays
        predictions (list): List of prediction results
        max_samples (int): Maximum number of samples to display
        title (str): Title for the plot
        save_path (str, optional): Path to save the visualization
        figsize (tuple): Figure size (width, height)
    """
    # Limit the number of samples
    n_samples = min(len(images), len(predictions), max_samples)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16)
    
    for i in range(n_samples):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # Display the image
        plt.imshow(images[i])
        plt.axis('off')
        
        # Get prediction info
        pred = predictions[i]
        blood_group = pred.get('predicted_blood_group', 'Unknown')
        confidence = pred.get('confidence', 0)
        
        # Set color based on confidence
        color = 'green' if confidence >= 0.7 else ('orange' if confidence >= 0.5 else 'red')
        
        plt.title(f"{blood_group} ({confidence:.2f})", color=color, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust to make room for the title
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions visualization saved to: {save_path}")
    
    plt.show()


def visualize_feature_correlation(features_df, target_col=None, top_n=20, title="Feature Correlation",
                                 save_path=None, figsize=(12, 10)):
    """
    Visualize correlation between features.
    
    Args:
        features_df (pandas.DataFrame): DataFrame containing features
        target_col (str, optional): Target column name to highlight
        top_n (int): Number of top correlated features to display
        title (str): Title for the plot
        save_path (str, optional): Path to save the visualization
        figsize (tuple): Figure size (width, height)
    """
    # Calculate correlation matrix
    corr = features_df.corr()
    
    # If target column specified, sort features by correlation with target
    if target_col and target_col in corr.columns:
        # Get absolute correlation with target
        target_corr = corr[target_col].abs()
        # Select top N features (excluding target itself)
        top_features = target_corr.sort_values(ascending=False)[1:top_n+1].index.tolist()
        # Add target column
        selected_columns = [target_col] + top_features
        corr_subset = corr.loc[selected_columns, selected_columns]
    else:
        # Just use top_n features
        corr_subset = corr.iloc[:top_n, :top_n]
    
    plt.figure(figsize=figsize)
    
    # Create the heatmap
    mask = np.triu(np.ones_like(corr_subset, dtype=bool))
    heatmap = sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm',
                         mask=mask, vmin=-1, vmax=1, center=0,
                         square=True, linewidths=.5, cbar_kws={'shrink': .5})
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature correlation visualization saved to: {save_path}")
    
    plt.show()


def visualize_metrics_by_blood_group(classification_report, title="Performance by Blood Group",
                                    save_path=None, figsize=(12, 8)):
    """
    Visualize classification metrics for each blood group.
    
    Args:
        classification_report (dict): Classification report as dictionary
        title (str): Title for the plot
        save_path (str, optional): Path to save the visualization
        figsize (tuple): Figure size (width, height)
    """
    # Extract metrics for each class
    blood_groups = []
    precision = []
    recall = []
    f1_score = []
    
    for class_name, metrics in classification_report.items():
        if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            blood_groups.append(class_name)
            precision.append(metrics.get('precision', 0))
            recall.append(metrics.get('recall', 0))
            f1_score.append(metrics.get('f1-score', 0))
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Blood Group': blood_groups,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    })
    
    # Sort by blood group if possible
    try:
        df['sort_key'] = df['Blood Group'].apply(lambda x: BLOOD_GROUPS.index(x) if x in BLOOD_GROUPS else 999)
        df = df.sort_values('sort_key').drop('sort_key', axis=1)
    except:
        pass
    
    # Melt the DataFrame for easier plotting
    df_melted = pd.melt(df, id_vars=['Blood Group'], var_name='Metric', value_name='Score')
    
    plt.figure(figsize=figsize)
    
    # Create the grouped bar plot
    sns.barplot(x='Blood Group', y='Score', hue='Metric', data=df_melted)
    
    plt.ylim(0, 1)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics by blood group visualization saved to: {save_path}")
    
    plt.show()


def visualize_results_dashboard(metrics_file, save_dir=None):
    """
    Create a comprehensive dashboard of model evaluation results.
    
    Args:
        metrics_file (str): Path to metrics JSON file
        save_dir (str, optional): Directory to save visualizations
    """
    # Load metrics from file
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Create save directory if needed
    if save_dir is None:
        save_dir = create_results_directory("dashboard")
    else:
        os.makedirs(save_dir, exist_ok=True)
    
    # Extract data from metrics
    accuracy = metrics.get('accuracy', 0)
    classification_report = metrics.get('classification_report', {})
    confusion_matrix = np.array(metrics.get('confusion_matrix', []))
    
    # Create a report summary
    summary = f"Model Evaluation Results\n\n"
    summary += f"Accuracy: {accuracy:.4f}\n\n"
    
    # Add class-specific metrics
    summary += "Metrics by Blood Group:\n"
    for class_name, class_metrics in classification_report.items():
        if isinstance(class_metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            summary += f"{class_name}: Precision={class_metrics.get('precision', 0):.4f}, "
            summary += f"Recall={class_metrics.get('recall', 0):.4f}, "
            summary += f"F1-Score={class_metrics.get('f1-score', 0):.4f}\n"
    
    # Write summary to file
    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    # Create visualizations
    
    # 1. Confusion Matrix
    if confusion_matrix.size > 0:
        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        visualize_confusion_matrix(
            confusion_matrix, 
            title="Confusion Matrix",
            save_path=cm_path
        )
        
        # Also create a normalized version
        cm_norm_path = os.path.join(save_dir, "confusion_matrix_normalized.png")
        visualize_confusion_matrix(
            confusion_matrix, 
            title="Normalized Confusion Matrix",
            normalize=True,
            save_path=cm_norm_path
        )
    
    # 2. Metrics by Blood Group
    if classification_report:
        metrics_path = os.path.join(save_dir, "metrics_by_blood_group.png")
        visualize_metrics_by_blood_group(
            classification_report,
            title="Performance Metrics by Blood Group",
            save_path=metrics_path
        )
    
    print(f"Dashboard created in: {save_dir}")
    return save_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize blood group classification results")
    parser.add_argument("metrics_file", help="Path to metrics JSON file")
    parser.add_argument("--output", help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    if os.path.exists(args.metrics_file):
        visualize_results_dashboard(args.metrics_file, args.output)
    else:
        print(f"Error: Metrics file not found: {args.metrics_file}")
        sys.exit(1) 