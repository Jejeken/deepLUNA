"""
Model Loader for CXR Analysis v3
================================

Enhanced model loading with support for multiple model types and better error handling.
"""

import os
import torch
import tensorflow as tf
from tensorflow import keras
from typing import Optional, Dict, Any, Tuple
from .model_config import ModelInfo, ModelConfig

# Import model architectures
from models import OptimizedMultiInputClassifier, EnhancedGradCAM
from models.losses import (
    dice_coefficient, dice_loss, iou_coefficient, iou_loss,
    combined_loss, precision, recall, specificity
)


class ModelLoader:
    """Enhanced model loader with dynamic model selection"""

    def __init__(self, device: torch.device = None):
        """Initialize model loader"""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = ModelConfig()

        # Loaded models cache
        self.loaded_models = {
            'segmentation': {},
            'classification': {}
        }

        # Custom objects for TensorFlow models
        self.custom_objects = {
            'combined_loss': combined_loss,
            'dice_coefficient': dice_coefficient,
            'dice_loss': dice_loss,
            'iou_coefficient': iou_coefficient,
            'iou_loss': iou_loss,
            'precision': precision,
            'recall': recall,
            'specificity': specificity
        }

    def get_available_models(self, model_type: str) -> Dict[str, str]:
        """Get available models as name-path mapping for dropdown"""
        models = self.config.get_available_models(model_type)
        return {model.name: model.path for model in models}

    def load_segmentation_model(self, model_name: str) -> Tuple[Optional[Any], str]:
        """Load segmentation model by name"""
        try:
            # Check if already loaded
            if model_name in self.loaded_models['segmentation']:
                return self.loaded_models['segmentation'][model_name], f"✓ {model_name} (cached)"

            # Get model info
            model_info = self.config.get_model_by_name('segmentation', model_name)
            if not model_info:
                return None, f"✗ Model '{model_name}' not found"

            if not os.path.exists(model_info.path):
                return None, f"✗ Model file not found: {model_info.path}"

            # Load based on file extension
            if model_info.path.endswith(('.h5', '.keras')):
                model = self._load_tensorflow_model(model_info)
            else:
                return None, f"✗ Unsupported model format: {model_info.path}"

            if model is not None:
                # Cache the model
                self.loaded_models['segmentation'][model_name] = model
                return model, f"✓ {model_name} loaded successfully"
            else:
                return None, f"✗ Failed to load {model_name}"

        except Exception as e:
            error_msg = f"✗ Error loading {model_name}: {str(e)}"
            print(error_msg)
            return None, error_msg

    def load_classification_model(self, model_name: str) -> Tuple[Optional[Any], Optional[EnhancedGradCAM], str]:
        """Load classification model by name and return model + GradCAM"""
        try:
            # Check if already loaded
            cache_key = model_name
            if cache_key in self.loaded_models['classification']:
                cached_data = self.loaded_models['classification'][cache_key]
                return cached_data['model'], cached_data['gradcam'], f"✓ {model_name} (cached)"

            # Get model info
            model_info = self.config.get_model_by_name('classification', model_name)
            if not model_info:
                return None, None, f"✗ Model '{model_name}' not found"

            if not os.path.exists(model_info.path):
                return None, None, f"✗ Model file not found: {model_info.path}"

            # Load based on file extension and architecture
            if model_info.path.endswith('.pth'):
                model, gradcam = self._load_pytorch_model(model_info)
            elif model_info.path.endswith('.h5'):
                model = self._load_tensorflow_classification_model(model_info)
                gradcam = None  # TensorFlow models don't support GradCAM in this implementation
            else:
                return None, None, f"✗ Unsupported model format: {model_info.path}"

            if model is not None:
                # Cache the model and gradcam
                self.loaded_models['classification'][cache_key] = {
                    'model': model,
                    'gradcam': gradcam
                }
                return model, gradcam, f"✓ {model_name} loaded successfully"
            else:
                return None, None, f"✗ Failed to load {model_name}"

        except Exception as e:
            error_msg = f"✗ Error loading {model_name}: {str(e)}"
            print(error_msg)
            return None, None, error_msg

    def _load_tensorflow_model(self, model_info: ModelInfo) -> Optional[Any]:
        """Load TensorFlow/Keras segmentation model"""
        try:
            # Update custom objects with actual functions
            if model_info.custom_objects:
                for key in model_info.custom_objects:
                    if key in self.custom_objects:
                        model_info.custom_objects[key] = self.custom_objects[key]

            # Load model
            model = keras.models.load_model(
                model_info.path,
                custom_objects=self.custom_objects
            )

            print(f"Successfully loaded TensorFlow model: {model_info.name}")
            return model

        except Exception as e:
            print(f"Failed to load TensorFlow model {model_info.name}: {e}")
            return None

    def _load_tensorflow_classification_model(self, model_info: ModelInfo) -> Optional[Any]:
        """Load TensorFlow/Keras classification model"""
        try:
            model = keras.models.load_model(model_info.path)
            print(f"Successfully loaded TensorFlow classification model: {model_info.name}")
            return model

        except Exception as e:
            print(f"Failed to load TensorFlow classification model {model_info.name}: {e}")
            return None

    def _load_pytorch_model(self, model_info: ModelInfo) -> Tuple[Optional[Any], Optional[EnhancedGradCAM]]:
        """Load PyTorch classification model"""
        try:
            # Determine model architecture and create instance
            if 'efficientnet' in model_info.architecture.lower():
                model = OptimizedMultiInputClassifier(dropout=0.3).to(self.device)
            else:
                # For other architectures, would need specific implementations
                print(f"Architecture {model_info.architecture} not yet implemented")
                return None, None

            # Load checkpoint
            checkpoint = torch.load(model_info.path, map_location=self.device, weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    # Store additional info if available
                    self.model_config = checkpoint.get('model_config', {})
                    self.performance_metrics = checkpoint.get('performance_metrics', {})
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume the checkpoint is the state dict
                    model.load_state_dict(checkpoint)
            else:
                # Direct state dict
                model.load_state_dict(checkpoint)

            model.eval()

            # Create GradCAM instance for PyTorch models
            gradcam = EnhancedGradCAM(model)

            print(f"Successfully loaded PyTorch model: {model_info.name}")
            return model, gradcam

        except Exception as e:
            print(f"Failed to load PyTorch model {model_info.name}: {e}")
            return None, None

    def unload_model(self, model_type: str, model_name: str) -> bool:
        """Unload a specific model from cache"""
        try:
            if model_name in self.loaded_models[model_type]:
                del self.loaded_models[model_type][model_name]
                # Force garbage collection
                import gc
                gc.collect()

                # Clear GPU cache if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return True
            return False

        except Exception as e:
            print(f"Error unloading model {model_name}: {e}")
            return False

    def unload_all_models(self):
        """Unload all cached models"""
        try:
            self.loaded_models = {
                'segmentation': {},
                'classification': {}
            }

            # Force cleanup
            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("All models unloaded successfully")

        except Exception as e:
            print(f"Error unloading all models: {e}")

    def get_model_info(self, model_type: str, model_name: str) -> Optional[ModelInfo]:
        """Get detailed information about a model"""
        return self.config.get_model_by_name(model_type, model_name)

    def validate_model_pair(self, seg_model_name: str, clf_model_name: str) -> Dict[str, Any]:
        """Validate compatibility between selected models"""
        seg_model_info = self.config.get_model_by_name('segmentation', seg_model_name)
        clf_model_info = self.config.get_model_by_name('classification', clf_model_name)

        if not seg_model_info or not clf_model_info:
            return {
                'valid': False,
                'error': 'One or both models not found'
            }

        # Check file existence
        if not os.path.exists(seg_model_info.path):
            return {
                'valid': False,
                'error': f'Segmentation model file not found: {seg_model_info.path}'
            }

        if not os.path.exists(clf_model_info.path):
            return {
                'valid': False,
                'error': f'Classification model file not found: {clf_model_info.path}'
            }

        # Check compatibility
        compatibility = self.config.validate_model_compatibility(seg_model_info, clf_model_info)

        return {
            'valid': compatibility['compatible'],
            'warnings': compatibility['warnings'],
            'segmentation_info': {
                'name': seg_model_info.name,
                'architecture': seg_model_info.architecture,
                'input_size': seg_model_info.input_size
            },
            'classification_info': {
                'name': clf_model_info.name,
                'architecture': clf_model_info.architecture,
                'input_size': clf_model_info.input_size
            }
        }

    def get_loading_status(self) -> Dict[str, Any]:
        """Get current loading status of all models"""
        status = {
            'segmentation': {},
            'classification': {},
            'total_loaded': 0
        }

        for model_type in ['segmentation', 'classification']:
            available_models = self.config.get_model_names(model_type)
            for model_name in available_models:
                is_loaded = model_name in self.loaded_models[model_type]
                status[model_type][model_name] = {
                    'loaded': is_loaded,
                    'path': self.config.get_model_by_name(model_type, model_name).path if self.config.get_model_by_name(model_type, model_name) else 'Unknown'
                }
                if is_loaded:
                    status['total_loaded'] += 1

        return status

    def preload_default_models(self) -> Dict[str, str]:
        """Preload default models for faster startup"""
        status = {}

        # Load default segmentation model
        default_seg = self.config.get_default_model('segmentation')
        if default_seg:
            _, seg_status = self.load_segmentation_model(default_seg.name)
            status['segmentation'] = seg_status

        # Load default classification model
        default_clf = self.config.get_default_model('classification')
        if default_clf:
            _, _, clf_status = self.load_classification_model(default_clf.name)
            status['classification'] = clf_status

        return status

    def refresh_available_models(self):
        """Refresh the list of available models"""
        self.config.refresh_models()

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        import psutil
        import torch

        memory_info = {
            'system_ram': {
                'total': psutil.virtual_memory().total / (1024**3),  # GB
                'available': psutil.virtual_memory().available / (1024**3),  # GB
                'used': psutil.virtual_memory().used / (1024**3)  # GB
            },
            'loaded_models': {
                'segmentation': len(self.loaded_models['segmentation']),
                'classification': len(self.loaded_models['classification'])
            }
        }

        if torch.cuda.is_available():
            memory_info['gpu'] = {
                'total': torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
                'allocated': torch.cuda.memory_allocated(0) / (1024**3),  # GB
                'cached': torch.cuda.memory_reserved(0) / (1024**3)  # GB
            }

        return memory_info