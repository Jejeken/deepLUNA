"""
Image Transforms for CXR Classification
=======================================

Advanced image transformation pipeline optimized for medical X-ray images.
"""

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AdvancedXRayTransforms:
    """
    Advanced transforms for X-ray image preprocessing and augmentation
    
    Optimized for medical imaging with appropriate augmentations that preserve
    clinical relevance while improving model robustness.
    """
    
    def __init__(self, image_size=224, is_training=False, normalization_type='standard'):
        """
        Initialize transforms pipeline
        
        Args:
            image_size: Target image size (assumes square images)
            is_training: Whether to apply training augmentations
            normalization_type: Type of normalization ('standard', 'imagenet', 'custom')
        """
        self.image_size = image_size
        self.is_training = is_training
        self.normalization_type = normalization_type
        
        # Set normalization parameters
        self.mean, self.std = self._get_normalization_params()
        
        # Build transformation pipeline
        self.transform = self._build_transform_pipeline()
    
    def _get_normalization_params(self):
        """Get normalization parameters based on type"""
        if self.normalization_type == 'imagenet':
            return [0.485], [0.229]  # ImageNet grayscale equivalent
        elif self.normalization_type == 'custom':
            return [0.449], [0.226]  # Custom medical imaging normalization
        else:  # standard
            return [0.5], [0.5]  # Standard [-1, 1] normalization
    
    def _build_transform_pipeline(self):
        """Build the complete transformation pipeline"""
        if self.is_training:
            return self._build_training_transforms()
        else:
            return self._build_inference_transforms()
    
    def _build_training_transforms(self):
        """Build training transforms with medical-appropriate augmentations"""
        return A.Compose([
            # Geometric transformations (conservative for medical images)
            A.Rotate(
                limit=10,  # Conservative rotation for medical images
                border_mode=cv2.BORDER_CONSTANT, 
                value=0, 
                p=0.7
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.7
            ),
            A.HorizontalFlip(p=0.5),  # Only horizontal flip for chest X-rays
            
            # Intensity transformations (preserve clinical information)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.8
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.5
            ),
            
            # Noise and blur (subtle to preserve clinical details)
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(1, 3), p=1.0),
            ], p=0.3),
            
            # CLAHE enhancement (beneficial for X-rays)
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=0.5
            ),
            
            # Elastic transformations (very subtle for medical images)
            A.ElasticTransform(
                alpha=50,
                sigma=5,
                alpha_affine=0,
                p=0.2
            ),
            
            # Final resize and normalization
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])
    
    def _build_inference_transforms(self):
        """Build inference transforms (minimal processing)"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])
    
    def __call__(self, image):
        """
        Apply transforms to input image
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Transformed image tensor
        """
        # Convert to numpy array if needed
        if hasattr(image, 'numpy'):  # PIL Image
            image = np.array(image)
        
        # Ensure grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 3:  # RGB to grayscale
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 1:  # Remove channel dimension
                image = image[:, :, 0]
        
        # Apply preprocessing if not in training mode
        if not self.is_training:
            image = self._apply_preprocessing(image)
        
        # Apply albumentations transform
        try:
            transformed = self.transform(image=image)
            return transformed['image']
        except Exception as e:
            # Fallback: basic processing if albumentations fails
            print(f"Warning: Transform failed ({e}), using fallback")
            return self._fallback_transform(image)
    
    def _apply_preprocessing(self, image):
        """Apply standard preprocessing for inference"""
        # CLAHE enhancement (improves contrast in medical images)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        return image
    
    def _fallback_transform(self, image):
        """Fallback transform if main pipeline fails"""
        # Basic resize and normalize
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        
        # Apply normalization
        image = (image - self.mean[0]) / self.std[0]
        
        # Convert to tensor format
        import torch
        return torch.from_numpy(image).unsqueeze(0).float()
    
    def get_transform_info(self):
        """Get information about current transform configuration"""
        return {
            'image_size': self.image_size,
            'is_training': self.is_training,
            'normalization_type': self.normalization_type,
            'mean': self.mean,
            'std': self.std,
            'num_transforms': len(self.transform.transforms) if hasattr(self.transform, 'transforms') else 1
        }


class XRayAugmentationPipeline:
    """
    Specialized augmentation pipeline for X-ray images during training
    
    Provides more control over augmentation intensity and medical-specific
    transformations.
    """
    
    def __init__(self, image_size=224, intensity='medium'):
        """
        Initialize augmentation pipeline
        
        Args:
            image_size: Target image size
            intensity: Augmentation intensity ('light', 'medium', 'heavy')
        """
        self.image_size = image_size
        self.intensity = intensity
        self.transform = self._build_pipeline()
    
    def _build_pipeline(self):
        """Build augmentation pipeline based on intensity"""
        if self.intensity == 'light':
            return self._light_augmentations()
        elif self.intensity == 'heavy':
            return self._heavy_augmentations()
        else:
            return self._medium_augmentations()
    
    def _light_augmentations(self):
        """Light augmentations for sensitive medical data"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    
    def _medium_augmentations(self):
        """Medium augmentations (balanced approach)"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    
    def _heavy_augmentations(self):
        """Heavy augmentations for robust training"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.8),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
                A.GaussianBlur(blur_limit=(1, 5), p=1.0),
            ], p=0.4),
            A.ElasticTransform(alpha=100, sigma=10, p=0.3),
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    
    def __call__(self, image):
        """Apply augmentation pipeline"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        transformed = self.transform(image=image)
        return transformed['image']


def create_test_time_augmentation(image, num_augmentations=5):
    """
    Create multiple augmented versions for test-time augmentation
    
    Args:
        image: Input image
        num_augmentations: Number of augmented versions to create
        
    Returns:
        List of augmented images
    """
    # Light augmentation for TTA
    tta_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=5, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Resize(224, 224),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])
    
    augmented_images = []
    for _ in range(num_augmentations):
        augmented = tta_transform(image=image)['image']
        augmented_images.append(augmented)
    
    return augmented_images