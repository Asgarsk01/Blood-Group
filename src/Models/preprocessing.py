#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocessing module for blood group detection images.
Handles image loading, normalization, segmentation, augmentation,
and feature extraction for optimal model performance.
"""

import os
import cv2
import numpy as np
from skimage import exposure, filters, morphology, segmentation, measure
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter

# Constants for preprocessing
DEFAULT_SIZE = (224, 224)
CHANNEL_MEANS = [0.485, 0.456, 0.406]  # ImageNet means
CHANNEL_STDS = [0.229, 0.224, 0.225]   # ImageNet stds

# Augmentation settings
ROTATION_RANGE = 15
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)
BLUR_SIGMA_RANGE = (0.0, 0.5)
NOISE_LEVEL = 0.01
HORIZONTAL_FLIP_PROB = 0.5


def load_image(image_path, grayscale=False):
    """
    Load an image from path.
    
    Args:
        image_path (str): Path to the image file
        grayscale (bool): Whether to load as grayscale
        
    Returns:
        numpy.ndarray: Loaded image array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if grayscale:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        # OpenCV loads as BGR, convert to RGB
        img = cv2.imread(image_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(image, target_size=DEFAULT_SIZE):
    """
    Resize image to target size.
    
    Args:
        image (numpy.ndarray): Input image array
        target_size (tuple): Target size as (width, height)
        
    Returns:
        numpy.ndarray: Resized image
    """
    if image is None:
        raise ValueError("Input image is None")
    
    # Check if grayscale and convert to 3 channels if needed
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image, method='standard'):
    """
    Normalize image values.
    
    Args:
        image (numpy.ndarray): Input image array
        method (str): Normalization method ('standard', 'imagenet', 'minmax', 'adaptive')
        
    Returns:
        numpy.ndarray: Normalized image
    """
    if method == 'standard':
        # Simple [0, 1] normalization
        return image.astype(np.float32) / 255.0
    
    elif method == 'imagenet':
        # ImageNet normalization
        img = image.astype(np.float32) / 255.0
        # For RGB images
        if len(img.shape) == 3 and img.shape[2] == 3:
            for i in range(3):
                img[:, :, i] = (img[:, :, i] - CHANNEL_MEANS[i]) / CHANNEL_STDS[i]
        return img
    
    elif method == 'minmax':
        # Min-max scaling to [0, 1]
        img_min = np.min(image)
        img_max = np.max(image)
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        return image.astype(np.float32) / 255.0
    
    elif method == 'adaptive':
        # Adaptive histogram equalization for enhanced contrast
        if len(image.shape) == 3:
            # Convert to LAB and apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab_planes = list(cv2.split(lab))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return enhanced.astype(np.float32) / 255.0
        else:
            # Apply CLAHE directly to grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            return enhanced.astype(np.float32) / 255.0
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def segment_blood_cells(image):
    """
    Segment blood cells in the image using image processing techniques.
    
    Args:
        image (numpy.ndarray): Input blood sample image
        
    Returns:
        tuple: (segmented_image, mask, contours)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold (Otsu's method)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Opening operation to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Marker-based watershed
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Find unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Create markers for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    if len(image.shape) == 3:
        markers = cv2.watershed(image, markers)
    else:
        markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), markers)
    
    # Create a mask for the identified regions
    mask = np.zeros_like(gray)
    mask[markers > 1] = 255
    
    # Find contours of the segmented cells
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Apply the mask to the original image
    if len(image.shape) == 3:
        segmented = image.copy()
        segmented[markers <= 1] = [0, 0, 0]
    else:
        segmented = gray.copy()
        segmented[markers <= 1] = 0
    
    return segmented, mask, contours


def extract_features(image, mask=None):
    """
    Extract features from blood sample images.
    
    Args:
        image (numpy.ndarray): Input blood sample image
        mask (numpy.ndarray, optional): Binary mask of segmented cells
        
    Returns:
        dict: Dictionary of extracted features
    """
    features = {}
    
    # Convert to grayscale for texture analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Color features (if RGB)
    if len(image.shape) == 3:
        # Apply mask if provided
        if mask is not None:
            masked_img = image.copy()
            for c in range(3):
                masked_img[:, :, c] = image[:, :, c] * (mask > 0)
            
            # Extract color histograms from masked image
            color_means = [np.mean(masked_img[:, :, c][mask > 0]) for c in range(3)]
            color_stds = [np.std(masked_img[:, :, c][mask > 0]) for c in range(3)]
        else:
            # Extract global color features
            color_means = [np.mean(image[:, :, c]) for c in range(3)]
            color_stds = [np.std(image[:, :, c]) for c in range(3)]
        
        features['color_means'] = color_means
        features['color_stds'] = color_stds
        
        # RGB ratios (important for blood type analysis)
        features['r_g_ratio'] = color_means[0] / max(color_means[1], 1e-5)
        features['r_b_ratio'] = color_means[0] / max(color_means[2], 1e-5)
        features['g_b_ratio'] = color_means[1] / max(color_means[2], 1e-5)
    
    # Texture features
    # GLCM (Gray-Level Co-occurrence Matrix) features
    from skimage.feature import graycomatrix, graycoprops
    
    # Quantize the image to fewer gray levels
    nbins = 32
    gray_quantized = np.digitize(gray, np.linspace(0, 255, nbins+1)) - 1
    
    # Calculate GLCM
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray_quantized, distances, angles, levels=nbins, symmetric=True, normed=True)
    
    # Calculate GLCM properties
    glcm_features = {}
    glcm_features['contrast'] = graycoprops(glcm, 'contrast').mean()
    glcm_features['dissimilarity'] = graycoprops(glcm, 'dissimilarity').mean()
    glcm_features['homogeneity'] = graycoprops(glcm, 'homogeneity').mean()
    glcm_features['energy'] = graycoprops(glcm, 'energy').mean()
    glcm_features['correlation'] = graycoprops(glcm, 'correlation').mean()
    
    features.update(glcm_features)
    
    # Morphological features if mask is provided
    if mask is not None:
        # Label individual cells
        labeled_mask = measure.label(mask)
        props = measure.regionprops(labeled_mask)
        
        if props:
            # Cell count
            features['cell_count'] = len(props)
            
            # Extract size and shape features
            areas = [prop.area for prop in props]
            perimeters = [prop.perimeter for prop in props]
            eccentricities = [prop.eccentricity for prop in props]
            
            # Calculate statistics of these features
            features['mean_cell_area'] = np.mean(areas)
            features['std_cell_area'] = np.std(areas)
            features['mean_cell_perimeter'] = np.mean(perimeters)
            features['mean_cell_eccentricity'] = np.mean(eccentricities)
            
            # Circularity (4*pi*area/perimeter^2)
            circularities = [4 * np.pi * area / max(perim**2, 1e-5) for area, perim in zip(areas, perimeters)]
            features['mean_cell_circularity'] = np.mean(circularities)
    
    return features


def augment_samples(image_path, num_samples=5, target_size=DEFAULT_SIZE):
    """
    Generate augmented versions of an image.
    
    Args:
        image_path (str): Path to the original image
        num_samples (int): Number of augmented samples to generate
        target_size (tuple): Target size as (width, height)
        
    Returns:
        list: List of augmented images as numpy arrays
    """
    if isinstance(image_path, str):
        # Load the image
        img = Image.open(image_path)
    else:
        # Assume already a numpy array
        img = Image.fromarray((image_path * 255).astype(np.uint8))
    
    # Resize to target size
    img = img.resize(target_size)
    
    # List to store augmented samples
    augmented_images = []
    
    # Add original image
    augmented_images.append(np.array(img).astype(np.float32) / 255.0)
    
    # Generate additional augmented samples
    for _ in range(num_samples - 1):
        # Copy the original image
        aug_img = img.copy()
        
        # Random rotation
        if np.random.random() < 0.7:
            angle = np.random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
            aug_img = aug_img.rotate(angle, resample=Image.BILINEAR, expand=False)
        
        # Random horizontal flip
        if np.random.random() < HORIZONTAL_FLIP_PROB:
            aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random brightness adjustment
        if np.random.random() < 0.5:
            brightness_factor = np.random.uniform(*BRIGHTNESS_RANGE)
            aug_img = ImageEnhance.Brightness(aug_img).enhance(brightness_factor)
        
        # Random contrast adjustment
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(*CONTRAST_RANGE)
            aug_img = ImageEnhance.Contrast(aug_img).enhance(contrast_factor)
        
        # Random slight blur
        if np.random.random() < 0.3:
            sigma = np.random.uniform(*BLUR_SIGMA_RANGE)
            if sigma > 0:
                aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        # Convert to numpy array and normalize
        aug_array = np.array(aug_img).astype(np.float32) / 255.0
        
        # Add random noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, NOISE_LEVEL, aug_array.shape)
            aug_array = np.clip(aug_array + noise, 0, 1)
        
        augmented_images.append(aug_array)
    
    return augmented_images


def preprocess_image(image_path, target_size=DEFAULT_SIZE, normalize=True, segment=False, use_enhanced=True):
    """
    Preprocess an image for blood group detection.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size as (width, height)
        normalize (bool): Whether to normalize pixel values
        segment (bool): Whether to perform cell segmentation
        use_enhanced (bool): Whether to use enhanced version of the image
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Load the image
    image = load_image(image_path)
    
    # Resize to target size
    image = resize_image(image, target_size)
    
    # Apply segmentation if requested
    if segment:
        segmented, mask, _ = segment_blood_cells(image)
        # Use segmented image
        processed = segmented
    else:
        processed = image
    
    # Apply enhancement
    if use_enhanced:
        # Use adaptive histogram equalization
        processed = normalize_image(processed, method='adaptive')
    elif normalize:
        # Standard normalization
        processed = normalize_image(processed, method='standard')
    
    return processed


def visualize_preprocessing(image_path, save_path=None):
    """
    Visualize the preprocessing steps.
    
    Args:
        image_path (str): Path to the input image
        save_path (str, optional): Path to save the visualization
    """
    # Load original image
    original = load_image(image_path)
    
    # Resize
    resized = resize_image(original)
    
    # Normalize with different methods
    norm_standard = normalize_image(resized, 'standard')
    norm_adaptive = normalize_image(resized, 'adaptive')
    
    # Segment
    segmented, mask, _ = segment_blood_cells(resized)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot original
    axs[0, 0].imshow(original)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    # Plot resized
    axs[0, 1].imshow(resized)
    axs[0, 1].set_title('Resized Image')
    axs[0, 1].axis('off')
    
    # Plot standard normalized
    axs[0, 2].imshow(norm_standard)
    axs[0, 2].set_title('Standard Normalization')
    axs[0, 2].axis('off')
    
    # Plot adaptive normalized
    axs[1, 0].imshow(norm_adaptive)
    axs[1, 0].set_title('Adaptive Normalization')
    axs[1, 0].axis('off')
    
    # Plot segmented
    axs[1, 1].imshow(segmented)
    axs[1, 1].set_title('Segmented Image')
    axs[1, 1].axis('off')
    
    # Plot mask
    axs[1, 2].imshow(mask, cmap='gray')
    axs[1, 2].set_title('Segmentation Mask')
    axs[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test the preprocessing module if run directly
    print("Preprocessing module loaded successfully")
    
    # Example usage
    # visualize_preprocessing("path/to/test/image.jpg", "preprocessing_viz.png") 