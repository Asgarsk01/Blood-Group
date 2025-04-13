#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BloodDetector: Core blood type detection model using convolutional neural networks
and ensemble techniques to accurately classify blood groups from slide images.

This module implements the primary detection algorithms for the Blood Vision system.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from .preprocessing import preprocess_image, augment_samples
from .utils import load_model_weights, get_available_gpus

# Configure TensorFlow to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Blood group classification constants
BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
NUM_CLASSES = len(BLOOD_GROUPS)
INPUT_SHAPE = (224, 224, 3)  # Standard input size for our model

# Model architecture parameters
FILTERS = [64, 128, 256, 512]
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DROPOUT_RATE = 0.5

class BloodGroupDetector:
    """Blood group detection model for slide images using CNN architecture."""
    
    def __init__(self, model_path=None, use_gpu=True, confidence_threshold=0.75):
        """
        Initialize the blood group detector.
        
        Args:
            model_path (str, optional): Path to pre-trained model weights
            use_gpu (bool): Whether to use GPU acceleration if available
            confidence_threshold (float): Minimum confidence for a valid prediction
        """
        self.model = None
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Configure GPU usage
        if use_gpu and get_available_gpus():
            print(f"Using GPU(s): {get_available_gpus()}")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("Using CPU for inference")
        
        # Build and load model
        self._build_model()
        if model_path:
            self._load_weights()
    
    def _build_model(self):
        """
        Build the CNN architecture for blood group detection.
        Implements a standard convolutional network with batch normalization.
        """
        inputs = Input(shape=INPUT_SHAPE)
        x = inputs
        
        # Convolutional blocks
        for filters in FILTERS:
            x = Conv2D(filters, KERNEL_SIZE, activation='relu', padding='same')(x)
            x = Conv2D(filters, KERNEL_SIZE, activation='relu', padding='same')(x)
            x = MaxPooling2D(pool_size=POOL_SIZE)(x)
        
        # Classification head
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(DROPOUT_RATE)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(DROPOUT_RATE/2)(x)
        outputs = Dense(NUM_CLASSES, activation='softmax')(x)
        
        # Compile model
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def _load_weights(self):
        """Load pre-trained weights if available."""
        try:
            self.model.load_weights(self.model_path)
            print(f"Model weights loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Using randomly initialized weights")
    
    def predict(self, image_path, preprocess=True):
        """
        Predict blood group from an image.
        
        Args:
            image_path (str): Path to image file
            preprocess (bool): Whether to preprocess the image
        
        Returns:
            dict: Prediction results with blood group and confidence
        """
        # Ensure model is ready
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Load and preprocess image
        if preprocess:
            img = preprocess_image(image_path, target_size=INPUT_SHAPE[:2])
        else:
            # Load image without preprocessing (already preprocessed)
            from tensorflow.keras.preprocessing.image import load_img, img_to_array
            img = img_to_array(load_img(image_path, target_size=INPUT_SHAPE[:2]))
            img = img / 255.0
        
        # Expand dimensions to match model input requirements
        img = np.expand_dims(img, axis=0)
        
        # Get predictions
        predictions = self.model.predict(img)[0]
        idx = np.argmax(predictions)
        confidence = predictions[idx]
        
        # Return results
        result = {
            'blood_group': BLOOD_GROUPS[idx],
            'confidence': float(confidence),
            'predictions': {group: float(pred) for group, pred in zip(BLOOD_GROUPS, predictions)},
            'is_confident': confidence >= self.confidence_threshold
        }
        
        return result
    
    def ensemble_predict(self, image_path, num_augmentations=5):
        """
        Use ensemble prediction with augmentations for more robust results.
        
        Args:
            image_path (str): Path to image file
            num_augmentations (int): Number of augmented samples to generate
        
        Returns:
            dict: Ensemble prediction results
        """
        # Preprocess and augment image
        augmented_images = augment_samples(image_path, num_samples=num_augmentations)
        
        # Predict on all augmented images
        all_predictions = []
        for img in augmented_images:
            pred = self.model.predict(np.expand_dims(img, axis=0))[0]
            all_predictions.append(pred)
        
        # Average predictions
        avg_predictions = np.mean(all_predictions, axis=0)
        idx = np.argmax(avg_predictions)
        confidence = avg_predictions[idx]
        
        # Return results
        result = {
            'blood_group': BLOOD_GROUPS[idx],
            'confidence': float(confidence),
            'predictions': {group: float(pred) for group, pred in zip(BLOOD_GROUPS, avg_predictions)},
            'is_confident': confidence >= self.confidence_threshold
        }
        
        return result

    def train(self, train_data, validation_data, epochs=50, batch_size=32, callbacks=None):
        """
        Train the blood group detection model.
        
        Args:
            train_data: Training data generator or tuple (x_train, y_train)
            validation_data: Validation data generator or tuple (x_val, y_val)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            callbacks (list): List of Keras callbacks
        
        Returns:
            History object
        """
        return self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

    def evaluate(self, test_data, batch_size=32):
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test data generator or tuple (x_test, y_test)
            batch_size (int): Batch size for evaluation
        
        Returns:
            tuple: (loss, accuracy)
        """
        return self.model.evaluate(test_data, batch_size=batch_size)

    def save_model(self, save_path):
        """
        Save the model weights to a file.
        
        Args:
            save_path (str): Path to save the model weights
        """
        self.model.save_weights(save_path)
        print(f"Model saved to {save_path}")


def create_detector(model_path=None, use_gpu=True):
    """Factory function to create a blood group detector."""
    return BloodGroupDetector(model_path=model_path, use_gpu=use_gpu)


if __name__ == "__main__":
    # Simple test if run directly
    detector = BloodGroupDetector()
    print("BloodDetector module loaded successfully")
    print(f"Model summary: {detector.model.summary()}") 