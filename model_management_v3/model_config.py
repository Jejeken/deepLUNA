"""
Model Configuration for CXR Analysis v3
=======================================

Configuration management for different model types and their specifications.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class ModelInfo:
    """Information about a specific model"""
    name: str
    path: str
    model_type: str  # 'segmentation' or 'classification'
    architecture: str
    description: str
    input_size: tuple
    output_classes: Optional[int] = None
    preprocessing_required: bool = True
    custom_objects: Optional[Dict[str, Any]] = None


class ModelConfig:
    """Configuration manager for available models"""

    def __init__(self, base_path: str = None):
        """Initialize model configuration"""
        if base_path is None:
            base_path = "J:/Portofolio Diri/BRIN/aplikasi"

        self.base_path = base_path
        self.seg_model_path = os.path.join(base_path, "models/segmentation")
        self.clf_model_path = os.path.join(base_path, "models/clf")

        # Initialize model configurations
        self.available_models = {
            'segmentation': [],
            'classification': []
        }

        self._scan_available_models()

    def _scan_available_models(self):
        """Scan for available models in the designated directories"""
        # Scan segmentation models
        self._scan_segmentation_models()

        # Scan classification models
        self._scan_classification_models()

    def _scan_segmentation_models(self):
        """Scan for available segmentation models"""
        print(f"DEBUG: Scanning segmentation models in: {self.seg_model_path}")
        if not os.path.exists(self.seg_model_path):
            print(f"DEBUG: Segmentation model directory not found: {self.seg_model_path}")
            return

        # Define known segmentation models
        segmentation_configs = {
            'best_resunet_lung_segmentation.h5': ModelInfo(
                name="ResUNet - Best Performance",
                path=os.path.join(self.seg_model_path, 'best_resunet_lung_segmentation.h5'),
                model_type="segmentation",
                architecture="ResUNet",
                description="Best performing ResUNet model for lung segmentation",
                input_size=(256, 256, 1),
                preprocessing_required=True,
                custom_objects={
                    'combined_loss': None,  # Will be filled during import
                    'dice_coefficient': None,
                    'iou_coefficient': None,
                    'precision': None,
                    'recall': None,
                    'specificity': None
                }
            ),
            'best_resunet_lung_segmentation_backup.h5': ModelInfo(
                name="ResUNet - Backup Model",
                path=os.path.join(self.seg_model_path, 'best_resunet_lung_segmentation_backup.h5'),
                model_type="segmentation",
                architecture="ResUNet",
                description="Backup ResUNet model for lung segmentation",
                input_size=(256, 256, 1),
                preprocessing_required=True,
                custom_objects={
                    'combined_loss': None,
                    'dice_coefficient': None,
                    'iou_coefficient': None,
                    'precision': None,
                    'recall': None,
                    'specificity': None
                }
            ),
            'final_resunet_lung_segmentation.h5': ModelInfo(
                name="ResUNet - Final Version",
                path=os.path.join(self.seg_model_path, 'final_resunet_lung_segmentation.h5'),
                model_type="segmentation",
                architecture="ResUNet",
                description="Final version of ResUNet model",
                input_size=(256, 256, 1),
                preprocessing_required=True,
                custom_objects={
                    'combined_loss': None,
                    'dice_coefficient': None,
                    'iou_coefficient': None,
                    'precision': None,
                    'recall': None,
                    'specificity': None
                }
            ),
            'final_resunet_lung_segmentation.keras': ModelInfo(
                name="ResUNet - Keras Format",
                path=os.path.join(self.seg_model_path, 'final_resunet_lung_segmentation.keras'),
                model_type="segmentation",
                architecture="ResUNet",
                description="ResUNet model in Keras format",
                input_size=(256, 256, 1),
                preprocessing_required=True,
                custom_objects={
                    'combined_loss': None,
                    'dice_coefficient': None,
                    'iou_coefficient': None,
                    'precision': None,
                    'recall': None,
                    'specificity': None
                }
            )
        }

        # Add existing models
        for filename, config in segmentation_configs.items():
            print(f"DEBUG: Checking segmentation model: {filename} at {config.path}")
            if os.path.exists(config.path):
                print(f"DEBUG: Found segmentation model: {filename}")
                self.available_models['segmentation'].append(config)
            else:
                print(f"DEBUG: Segmentation model not found: {config.path}")

        print(f"DEBUG: Total segmentation models found: {len(self.available_models['segmentation'])}")

    def _scan_classification_models(self):
        """Scan for available classification models"""
        print(f"DEBUG: Scanning classification models in: {self.clf_model_path}")
        if not os.path.exists(self.clf_model_path):
            print(f"DEBUG: Classification model directory not found: {self.clf_model_path}")
            return

        # Define known classification models
        classification_configs = {
            'optimized_classifier_efficientB2.pth': ModelInfo(
                name="EfficientNet-B2 - Optimized",
                path=os.path.join(self.clf_model_path, 'optimized_classifier_efficientB2.pth'),
                model_type="classification",
                architecture="EfficientNet-B2",
                description="Optimized EfficientNet-B2 multi-input classifier",
                input_size=(224, 224, 3),
                output_classes=2,
                preprocessing_required=True
            ),
            'best_model_efficientnet.h5': ModelInfo(
                name="EfficientNet - Keras Version",
                path=os.path.join(self.clf_model_path, 'best_model_efficientnet.h5'),
                model_type="classification",
                architecture="EfficientNet",
                description="EfficientNet model in Keras format",
                input_size=(224, 224, 3),
                output_classes=2,
                preprocessing_required=True
            ),
            'torchxray_densenet_attention.pth': ModelInfo(
                name="DenseNet with Attention",
                path=os.path.join(self.clf_model_path, 'torchxray_densenet_attention.pth'),
                model_type="classification",
                architecture="DenseNet + Attention",
                description="DenseNet with attention mechanism for X-ray analysis",
                input_size=(224, 224, 3),
                output_classes=2,
                preprocessing_required=True
            )
        }

        # Add existing models
        for filename, config in classification_configs.items():
            print(f"DEBUG: Checking classification model: {filename} at {config.path}")
            if os.path.exists(config.path):
                print(f"DEBUG: Found classification model: {filename}")
                self.available_models['classification'].append(config)
            else:
                print(f"DEBUG: Classification model not found: {config.path}")

        print(f"DEBUG: Total classification models found: {len(self.available_models['classification'])}")

    def get_available_models(self, model_type: str) -> List[ModelInfo]:
        """Get list of available models for specified type"""
        return self.available_models.get(model_type, [])

    def get_model_names(self, model_type: str) -> List[str]:
        """Get list of model names for dropdown display"""
        models = self.get_available_models(model_type)
        return [model.name for model in models]

    def get_model_by_name(self, model_type: str, name: str) -> Optional[ModelInfo]:
        """Get model info by name"""
        models = self.get_available_models(model_type)
        for model in models:
            if model.name == name:
                return model
        return None

    def get_default_model(self, model_type: str) -> Optional[ModelInfo]:
        """Get default model for specified type"""
        models = self.get_available_models(model_type)
        if not models:
            return None

        # Default selection logic
        if model_type == 'segmentation':
            # Prefer the best performance model
            for model in models:
                if 'best' in model.name.lower():
                    return model
            return models[0]  # Fallback to first available

        elif model_type == 'classification':
            # Prefer the optimized EfficientNet model
            for model in models:
                if 'optimized' in model.name.lower() and 'efficientnet' in model.name.lower():
                    return model
            return models[0]  # Fallback to first available

        return models[0] if models else None

    def validate_model_compatibility(self, seg_model: ModelInfo, clf_model: ModelInfo) -> Dict[str, Any]:
        """Validate compatibility between segmentation and classification models"""
        issues = []
        warnings = []

        # Check input size compatibility
        if seg_model.input_size[:2] != (256, 256):
            warnings.append("Segmentation model input size is not 256x256")

        if clf_model.input_size[:2] != (224, 224):
            warnings.append("Classification model input size is not 224x224")

        # Check architecture compatibility
        if clf_model.architecture == "EfficientNet-B2" and seg_model.architecture != "ResUNet":
            warnings.append("Mixed architectures may affect performance")

        return {
            'compatible': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all available models"""
        summary = {
            'segmentation_models': len(self.available_models['segmentation']),
            'classification_models': len(self.available_models['classification']),
            'total_models': len(self.available_models['segmentation']) + len(self.available_models['classification']),
            'segmentation_architectures': list(set(m.architecture for m in self.available_models['segmentation'])),
            'classification_architectures': list(set(m.architecture for m in self.available_models['classification']))
        }
        return summary

    def refresh_models(self):
        """Refresh the list of available models"""
        self.available_models = {
            'segmentation': [],
            'classification': []
        }
        self._scan_available_models()

    def add_custom_model(self, model_info: ModelInfo):
        """Add a custom model configuration"""
        if model_info.model_type in self.available_models:
            # Check if model already exists
            existing_names = [m.name for m in self.available_models[model_info.model_type]]
            if model_info.name not in existing_names:
                self.available_models[model_info.model_type].append(model_info)
            else:
                raise ValueError(f"Model with name '{model_info.name}' already exists")
        else:
            raise ValueError(f"Invalid model type: {model_info.model_type}")

    def remove_model(self, model_type: str, name: str):
        """Remove a model from available models"""
        models = self.available_models.get(model_type, [])
        self.available_models[model_type] = [m for m in models if m.name != name]

    def export_config(self) -> Dict[str, Any]:
        """Export current configuration"""
        config = {
            'base_path': self.base_path,
            'segmentation_path': self.seg_model_path,
            'classification_path': self.clf_model_path,
            'available_models': {}
        }

        for model_type, models in self.available_models.items():
            config['available_models'][model_type] = [
                {
                    'name': m.name,
                    'path': m.path,
                    'architecture': m.architecture,
                    'description': m.description,
                    'input_size': m.input_size,
                    'output_classes': m.output_classes
                }
                for m in models
            ]

        return config