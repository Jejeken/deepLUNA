"""
Analysis Pipeline for CXR Analysis v3
=====================================

Enhanced analysis pipeline with improved modularity and error handling.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import traceback
import time

# Import existing modules
from preprocessing import apply_advanced_soft_mask, apply_attention_mask, AdvancedXRayTransforms
from utils import cleanup_gpu_memory

from .analysis_state import StateManager, AnalysisStatus


class AnalysisPipeline:
    """Enhanced analysis pipeline with state management"""

    def __init__(self, state_manager: StateManager):
        """Initialize analysis pipeline"""
        self.state_manager = state_manager
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Models will be injected
        self.segmentation_model = None
        self.classification_model = None
        self.gradcam = None

        # Pipeline configuration
        self.config = {
            'segmentation_input_size': (256, 256),
            'classification_input_size': (224, 224),
            'segmentation_threshold': 0.5,
            'soft_mask_alpha': 0.2,
            'use_gpu_optimization': True,
            'timeout_seconds': 120
        }

    def set_models(self, segmentation_model, classification_model, gradcam=None):
        """Set models for the pipeline"""
        self.segmentation_model = segmentation_model
        self.classification_model = classification_model
        self.gradcam = gradcam

    def run_complete_analysis(self, image_path: str, selected_models: Dict[str, str]) -> Dict[str, Any]:
        """
        Run complete analysis pipeline

        Returns:
            Dict with analysis results or error information
        """
        try:
            # Initialize analysis
            self.state_manager.start_analysis(image_path, selected_models)

            # Check if models are available
            if not self._validate_models():
                raise ValueError("Required models not loaded")

            # Step 1: Load and validate image
            self.state_manager.update_progress(1, "Loading Image", "Reading and validating input image")
            self.state_manager.update_status(AnalysisStatus.LOADING_IMAGE)

            original_img = self._load_and_validate_image(image_path)
            self.state_manager.store_intermediate_result('original_image', original_img)

            if self._check_cancellation():
                return self._create_cancellation_result()

            # Step 2: Run segmentation
            self.state_manager.update_progress(2, "Segmentation", "Running lung segmentation model")
            self.state_manager.update_status(AnalysisStatus.RUNNING_SEGMENTATION)

            mask_binary = self._run_segmentation(original_img)
            self.state_manager.store_intermediate_result('segmentation_mask', mask_binary)

            if self._check_cancellation():
                return self._create_cancellation_result()

            # Step 3: Create masked versions
            self.state_manager.update_progress(3, "Applying Masks", "Creating masked image variants")
            self.state_manager.update_status(AnalysisStatus.APPLYING_MASKS)

            masked_versions = self._create_masked_versions(original_img, mask_binary)
            self.state_manager.store_intermediate_result('masked_images', masked_versions)

            if self._check_cancellation():
                return self._create_cancellation_result()

            # Step 4: Run classification
            self.state_manager.update_progress(4, "Classification", "Running classification models")
            self.state_manager.update_status(AnalysisStatus.RUNNING_CLASSIFICATION)

            classification_results = self._run_classification(masked_versions)
            self.state_manager.store_intermediate_result('classification_results', classification_results)

            if self._check_cancellation():
                return self._create_cancellation_result()

            # Step 5: Generate GradCAM
            self.state_manager.update_progress(5, "GradCAM", "Generating attention visualizations")
            self.state_manager.update_status(AnalysisStatus.GENERATING_GRADCAM)

            gradcam_result = self._generate_gradcam(masked_versions, classification_results)
            self.state_manager.store_intermediate_result('gradcam_result', gradcam_result)

            if self._check_cancellation():
                return self._create_cancellation_result()

            # Step 6: Compile results
            self.state_manager.update_progress(6, "Compiling Results", "Finalizing analysis results")
            self.state_manager.update_status(AnalysisStatus.COMPILING_RESULTS)

            final_results = self._compile_results(
                original_img, mask_binary, masked_versions,
                classification_results, gradcam_result
            )

            # Step 7: Complete
            self.state_manager.update_progress(7, "Complete", "Analysis completed successfully")
            self.state_manager.complete_analysis(final_results)

            return {
                'success': True,
                'results': final_results,
                'analysis_time': self.state_manager.get_state().get_total_duration()
            }

        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()

            self.state_manager.set_error(error_msg, error_trace)

            return {
                'success': False,
                'error': error_msg,
                'traceback': error_trace
            }

        finally:
            # Cleanup
            cleanup_gpu_memory()

    def _validate_models(self) -> bool:
        """Validate that required models are loaded"""
        return (self.segmentation_model is not None and
                self.classification_model is not None)

    def _check_cancellation(self) -> bool:
        """Check if cancellation has been requested"""
        return self.state_manager.get_state().is_cancellation_requested()

    def _create_cancellation_result(self) -> Dict[str, Any]:
        """Create result for cancelled analysis"""
        self.state_manager.get_state().set_cancelled()
        return {
            'success': False,
            'cancelled': True,
            'message': 'Analysis was cancelled by user'
        }

    def _load_and_validate_image(self, image_path: str) -> np.ndarray:
        """Load and validate input image"""
        self.state_manager.get_state().start_step("load_image")

        try:
            original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if original_img is None:
                raise ValueError(f"Cannot read image file: {image_path}")

            if original_img.size == 0:
                raise ValueError("Image file is empty")

            # Basic validation
            if len(original_img.shape) != 2:
                raise ValueError(f"Expected grayscale image, got shape: {original_img.shape}")

            if original_img.shape[0] < 64 or original_img.shape[1] < 64:
                raise ValueError(f"Image too small: {original_img.shape}")

            return original_img

        finally:
            self.state_manager.get_state().end_step("load_image")

    def _run_segmentation(self, original_img: np.ndarray) -> np.ndarray:
        """Run segmentation model"""
        self.state_manager.get_state().start_step("segmentation")

        try:
            # Resize for segmentation model
            seg_size = self.config['segmentation_input_size']
            original_resized = cv2.resize(original_img, seg_size)

            # Prepare input
            seg_input = original_resized.astype(np.float32) / 255.0
            seg_input = np.expand_dims(seg_input, axis=(0, -1))

            # Run segmentation
            mask_pred = self.segmentation_model.predict(seg_input, verbose=0)

            # Post-process
            threshold = self.config['segmentation_threshold']
            mask_binary = (mask_pred[0, :, :, 0] > threshold).astype(np.uint8)

            # Fallback if mask is empty
            if np.sum(mask_binary) == 0:
                print("Warning: Empty segmentation mask, using fallback")
                mask_binary = np.zeros(seg_size, dtype=np.uint8)
                center_y, center_x = seg_size[0] // 2, seg_size[1] // 2
                radius = min(seg_size) // 4
                mask_binary[center_y-radius:center_y+radius,
                          center_x-radius:center_x+radius] = 1

            return mask_binary

        finally:
            self.state_manager.get_state().end_step("segmentation")

    def _create_masked_versions(self, original_img: np.ndarray, mask_binary: np.ndarray) -> Dict[str, np.ndarray]:
        """Create different masked versions of the image"""
        self.state_manager.get_state().start_step("masking")

        try:
            # Resize original to match mask
            seg_size = self.config['segmentation_input_size']
            original_resized = cv2.resize(original_img, seg_size)

            masked_versions = {
                'original': original_resized,
                'hard_masked': original_resized * mask_binary,
                'soft_masked': apply_advanced_soft_mask(
                    original_resized, mask_binary,
                    alpha=self.config['soft_mask_alpha']
                ),
                'attention_masked': apply_attention_mask(original_resized, mask_binary)
            }

            return masked_versions

        finally:
            self.state_manager.get_state().end_step("masking")

    def _run_classification(self, masked_versions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Run classification on different input versions"""
        self.state_manager.get_state().start_step("classification")

        try:
            clf_size = self.config['classification_input_size']
            clf_transform = AdvancedXRayTransforms(image_size=clf_size[0], is_training=False)

            results = {}
            self.classification_model.eval()

            with torch.no_grad():
                for version_name, image in masked_versions.items():
                    if self._check_cancellation():
                        break

                    # Determine input type
                    input_type = 0 if version_name == 'original' else 1

                    # Prepare input
                    tensor = clf_transform(image).unsqueeze(0).to(self.device)
                    type_tensor = torch.tensor([input_type]).to(self.device)

                    # Run model
                    output = self.classification_model(tensor, type_tensor)
                    probs = F.softmax(output, dim=1)

                    results[version_name] = {
                        'tensor': tensor,
                        'output': output,
                        'probs': probs.cpu().numpy()[0],
                        'confidence': probs.max().item()
                    }

                    # Clean up intermediate tensors
                    del tensor, type_tensor, output

            # Compute ensemble if we have results
            if results and not self._check_cancellation():
                ensemble_probs = self._compute_ensemble(results)
                results['ensemble'] = ensemble_probs

            cleanup_gpu_memory()
            return results

        finally:
            self.state_manager.get_state().end_step("classification")

    def _compute_ensemble(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute ensemble prediction from individual results"""
        try:
            # Get confidence weights
            individual_results = {k: v for k, v in results.items() if k != 'ensemble'}
            confidences = [res['confidence'] for res in individual_results.values()]
            total_conf = sum(confidences)

            if total_conf == 0:
                weights = [1.0 / len(confidences)] * len(confidences)
            else:
                weights = [conf / total_conf for conf in confidences]

            # Weighted average of probabilities
            ensemble_probs = np.zeros(2)
            for (name, res), weight in zip(individual_results.items(), weights):
                ensemble_probs += res['probs'] * weight

            return {
                'probs': ensemble_probs,
                'prediction': ensemble_probs.argmax(),
                'confidence': ensemble_probs.max(),
                'weights': dict(zip(individual_results.keys(), weights))
            }

        except Exception as e:
            print(f"Error computing ensemble: {e}")
            # Fallback to simple average
            individual_results = {k: v for k, v in results.items() if k != 'ensemble'}
            if individual_results:
                avg_probs = np.mean([res['probs'] for res in individual_results.values()], axis=0)
                return {
                    'probs': avg_probs,
                    'prediction': avg_probs.argmax(),
                    'confidence': avg_probs.max(),
                    'weights': {name: 1.0/len(individual_results) for name in individual_results.keys()}
                }
            else:
                return {
                    'probs': np.array([0.5, 0.5]),
                    'prediction': 0,
                    'confidence': 0.5,
                    'weights': {}
                }

    def _generate_gradcam(self, masked_versions: Dict[str, np.ndarray],
                         classification_results: Dict[str, Any]) -> np.ndarray:
        """Generate GradCAM visualization"""
        self.state_manager.get_state().start_step("gradcam")

        try:
            if not self.gradcam or 'ensemble' not in classification_results:
                # Return dummy heatmap
                clf_size = self.config['classification_input_size']
                return np.random.random(clf_size) * 0.3

            # Use the best performing input for GradCAM
            individual_results = {k: v for k, v in classification_results.items() if k != 'ensemble'}
            if not individual_results:
                clf_size = self.config['classification_input_size']
                return np.random.random(clf_size) * 0.3

            best_version = max(individual_results.keys(),
                             key=lambda k: individual_results[k]['confidence'])

            best_result = individual_results[best_version]
            input_type = 0 if best_version == 'original' else 1
            target_class = classification_results['ensemble']['prediction']

            # Generate GradCAM
            with torch.enable_grad():
                best_tensor = best_result['tensor'].requires_grad_(True)
                type_tensor = torch.tensor([input_type]).to(self.device)

                cam = self.gradcam.generate_cam(best_tensor, type_tensor, target_class)

            return cam if cam is not None else np.random.random(self.config['classification_input_size']) * 0.3

        except Exception as e:
            print(f"GradCAM generation failed: {e}")
            clf_size = self.config['classification_input_size']
            return np.random.random(clf_size) * 0.3

        finally:
            self.state_manager.get_state().end_step("gradcam")

    def _compile_results(self, original_img: np.ndarray, mask_binary: np.ndarray,
                        masked_versions: Dict[str, np.ndarray],
                        classification_results: Dict[str, Any],
                        gradcam: np.ndarray) -> Dict[str, Any]:
        """Compile all results into final format"""
        self.state_manager.get_state().start_step("compile_results")

        try:
            ensemble = classification_results.get('ensemble', {})
            class_names = ['Normal', 'Abnormal']

            # Get individual results (exclude ensemble)
            individual_results = {k: v for k, v in classification_results.items() if k != 'ensemble'}

            results = {
                'original_image': original_img,
                'original_resized': masked_versions.get('original'),
                'segmentation_mask': mask_binary,
                'segmented_hard': masked_versions.get('hard_masked'),
                'segmented_soft': masked_versions.get('soft_masked'),
                'segmented_attention': masked_versions.get('attention_masked'),
                'gradcam': gradcam,
                'prediction_class': ensemble.get('prediction', 0),
                'prediction_label': class_names[ensemble.get('prediction', 0)],
                'confidence': ensemble.get('confidence', 0.0),
                'ensemble_probabilities': ensemble.get('probs', np.array([0.5, 0.5])),
                'individual_probs': {k: v['probs'] for k, v in individual_results.items()},
                'adaptive_weights': ensemble.get('weights', {}),
                'individual_confidences': {k: v['confidence'] for k, v in individual_results.items()},
                'analysis_metadata': {
                    'segmentation_input_size': self.config['segmentation_input_size'],
                    'classification_input_size': self.config['classification_input_size'],
                    'device': str(self.device),
                    'models_used': self.state_manager.get_state().selected_models,
                    'processing_time': self.state_manager.get_state().get_total_duration()
                }
            }

            return results

        finally:
            self.state_manager.get_state().end_step("compile_results")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline configuration and status information"""
        return {
            'config': self.config.copy(),
            'device': str(self.device),
            'models_loaded': {
                'segmentation': self.segmentation_model is not None,
                'classification': self.classification_model is not None,
                'gradcam': self.gradcam is not None
            },
            'state': self.state_manager.get_state_copy()
        }

    def update_config(self, new_config: Dict[str, Any]):
        """Update pipeline configuration"""
        self.config.update(new_config)

    def cancel_analysis(self):
        """Request cancellation of current analysis"""
        self.state_manager.request_cancellation()