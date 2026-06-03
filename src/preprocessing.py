"""
Image preprocessing and data augmentation for HandScript AI.
Save this file in: HandScriptAI/src/preprocessing.py
"""

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImagePreprocessor:
    def __init__(self, target_size=(28, 28)):
        self.target_size = target_size
    
    def denoise(self, image):
        """Apply noise reduction to image."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur for noise reduction
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Apply bilateral filter for edge-preserving smoothing
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        return denoised
    
    def binarize(self, image):
        """Convert to binary image using adaptive thresholding."""
        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        return binary
    
    def normalize(self, image):
        """Normalize pixel values to 0-1 range."""
        return image.astype(np.float32) / 255.0
    
    def resize(self, image):
        """Resize image to target size."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def preprocess(self, image):
        """Full preprocessing pipeline."""
        # Denoise
        processed = self.denoise(image)
        
        # Binarize
        processed = self.binarize(processed)
        
        # Resize
        processed = self.resize(processed)
        
        # Normalize
        processed = self.normalize(processed)
        
        return processed


def create_data_generator(augment=True):
    """Create data generator with augmentation for training."""
    if augment:
        return ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest',
            rescale=1./255
        )
    else:
        return ImageDataGenerator(rescale=1./255)


def load_and_preprocess_data(train_dir, test_dir, target_size=(28, 28), batch_size=32):
    """Load data from directories with preprocessing."""
    train_datagen = create_data_generator(augment=True)
    test_datagen = create_data_generator(augment=False)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, test_generator
