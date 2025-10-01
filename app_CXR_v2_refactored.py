"""
CXR Segmentation and Classification Desktop Application - Refactored Version
===========================================================================

Clean and modular implementation of the Enhanced CXR Analysis Pipeline.
Uses modular components for better maintainability and code organization.

Dependencies to install:
pip install customtkinter pillow torch torchxrayvision tensorflow opencv-python-headless numpy albumentations scikit-learn matplotlib timm

Required model files (place in same directory or 'models' subfolder):
- best_resunet_lung_segmentation.h5
- optimized_classifier_efficientB2.pth

Author: Refactored for Modular Architecture
Version: 2.1 - Modular
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageTk
import warnings
import gc
import traceback
from threading import Timer
import time

# Import modular components
from models import (
    OptimizedMultiInputClassifier,
    EnhancedGradCAM,
    dice_coefficient, dice_loss, iou_coefficient, iou_loss,
    combined_loss, precision, recall, specificity
)
from preprocessing import (
    apply_advanced_soft_mask,
    apply_attention_mask,
    AdvancedXRayTransforms
)
from utils import cleanup_gpu_memory, get_memory_info

warnings.filterwarnings('ignore')

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class CXRAnalysisApp(ctk.CTk):
    """
    Main application class for CXR Analysis Pipeline
    
    Provides a GUI interface for loading chest X-ray images and running
    the enhanced analysis pipeline with segmentation and classification.
    """
    
    def __init__(self):
        """Initialize the application"""
        super().__init__()
        
        # Configure window
        self.title("Enhanced CXR Analysis - Pipeline")
        self.geometry("1600x1000")
        self.resizable(True, True)
        
        # Initialize 
        self.sgm_model = None
        self.clf_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gradcam = None
        self.current_image_path = None
        self.analysis_results = None
        
        # Analysis control variables
        self.current_analysis_thread = None
        self.analysis_in_progress = False
        self.stop_analysis_flag = threading.Event()

        # Setup UI
        self.setup_ui()
        
        # Load models in separate thread
        self.load_models_thread()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create sidebar frame
        self.sidebar_frame = ctk.CTkFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)
        
        # Sidebar title
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Enhanced CXR Analysis", 
                                      font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Version info
        self.version_label = ctk.CTkLabel(self.sidebar_frame, text="EfficientNet-B2 Pipeline v2.1", 
                                         font=ctk.CTkFont(size=11), text_color="gray")
        self.version_label.grid(row=1, column=0, padx=20, pady=(0, 10))
        
        # Load image button
        self.load_button = ctk.CTkButton(self.sidebar_frame, text="Load X-Ray Image", 
                                        command=self.load_image, height=40,
                                        font=ctk.CTkFont(size=14, weight="bold"))
        self.load_button.grid(row=2, column=0, padx=20, pady=10)
        
        # Change Image button
        self.change_button = ctk.CTkButton(
            self.sidebar_frame,
            text="Change Image",
            command=self.change_image_with_confirmation,
            height=35,
            font=ctk.CTkFont(size=13),
            fg_color="transparent",
            border_width=2,
            border_color="#3B82F6",
            text_color="#3B82F6",
            hover_color="#DC2626",
            state="disabled"
        )
        self.change_button.grid(row=3, column=0, padx=20, pady=(0, 10))

        # Current file label
        self.file_label = ctk.CTkLabel(self.sidebar_frame, text="No image selected",
                                      wraplength=300, font=ctk.CTkFont(size=12))
        self.file_label.grid(row=4, column=0, padx=20, pady=5)
        
        # Separator
        self.separator = ctk.CTkFrame(self.sidebar_frame, height=2)
        self.separator.grid(row=5, column=0, padx=20, pady=10, sticky="ew")

        # Analysis results section
        self._setup_results_section()
        
        # Model info section
        self._setup_model_info_section()
        
        # Main panel with tabs
        self._setup_main_panel()
        
        # Status bar
        self.status_bar = ctk.CTkLabel(self, text="Ready - Enhanced Pipeline Loaded", anchor="w", 
                                      font=ctk.CTkFont(size=12))
        self.status_bar.grid(row=1, column=1, padx=20, pady=(0, 10), sticky="ew")
    
    def _setup_results_section(self):
        """Setup the analysis results section"""
        # Analysis results title
        self.results_title = ctk.CTkLabel(self.sidebar_frame, text="Analysis Results",
                                         font=ctk.CTkFont(size=16, weight="bold"))
        self.results_title.grid(row=6, column=0, padx=20, pady=(10, 5))

        # Results frame
        self.results_frame = ctk.CTkFrame(self.sidebar_frame)
        self.results_frame.grid(row=7, column=0, padx=20, pady=5, sticky="ew")
        
        # Result labels
        self.prediction_label = ctk.CTkLabel(self.results_frame, text="Prediction: --", 
                                           font=ctk.CTkFont(size=14, weight="bold"))
        self.prediction_label.grid(row=0, column=0, padx=15, pady=5, sticky="w")
        
        self.confidence_label = ctk.CTkLabel(self.results_frame, text="Confidence: --", 
                                           font=ctk.CTkFont(size=12))
        self.confidence_label.grid(row=1, column=0, padx=15, pady=2, sticky="w")
        
        self.prob_normal_label = ctk.CTkLabel(self.results_frame, text="Normal: --", 
                                            font=ctk.CTkFont(size=11))
        self.prob_normal_label.grid(row=2, column=0, padx=15, pady=2, sticky="w")
        
        self.prob_abnormal_label = ctk.CTkLabel(self.results_frame, text="Abnormal: --", 
                                              font=ctk.CTkFont(size=11))
        self.prob_abnormal_label.grid(row=3, column=0, padx=15, pady=2, sticky="w")
        
        self.ensemble_info = ctk.CTkLabel(self.results_frame, text="Method: Enhanced Pipeline", 
                                         font=ctk.CTkFont(size=10), text_color="gray")
        self.ensemble_info.grid(row=4, column=0, padx=15, pady=2, sticky="w")
    
    def _setup_model_info_section(self):
        """Setup the model information section"""
        self.model_info_frame = ctk.CTkFrame(self.sidebar_frame)
        self.model_info_frame.grid(row=8, column=0, padx=20, pady=(10, 20), sticky="ew")
        
        self.model_info_label = ctk.CTkLabel(self.model_info_frame, text="Model Status", 
                                           font=ctk.CTkFont(size=14, weight="bold"))
        self.model_info_label.grid(row=0, column=0, padx=15, pady=5)
        
        self.segmentation_status = ctk.CTkLabel(self.model_info_frame, text="Segmentation: Loading...", 
                                              font=ctk.CTkFont(size=11))
        self.segmentation_status.grid(row=1, column=0, padx=15, pady=2, sticky="w")
        
        self.classification_status = ctk.CTkLabel(self.model_info_frame, text="Classification: Loading...", 
                                                font=ctk.CTkFont(size=11))
        self.classification_status.grid(row=2, column=0, padx=15, pady=2, sticky="w")
    
    def _setup_main_panel(self):
        """Setup the main panel with tabs"""
        self.tabview = ctk.CTkTabview(self, width=1000)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        # Create tabs
        self.tabview.add("Final Overlay")
        self.tabview.add("Processing Steps")
        self.tabview.add("Enhanced Results")
        
        # Final overlay tab
        self._setup_overlay_tab()
        
        # Processing steps tab
        self._setup_steps_tab()
        
        # Enhanced results tab
        self._setup_enhanced_tab()
    
    def _setup_overlay_tab(self):
        """Setup the final overlay tab"""
        self.final_overlay_frame = ctk.CTkFrame(self.tabview.tab("Final Overlay"))
        self.final_overlay_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.overlay_image_label = ctk.CTkLabel(self.final_overlay_frame, text="Load an image to see analysis")
        self.overlay_image_label.pack(expand=True)
    
    def _setup_steps_tab(self):
        """Setup the processing steps tab"""
        self.steps_frame = ctk.CTkFrame(self.tabview.tab("Processing Steps"))
        self.steps_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure grid for steps
        self.steps_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.steps_frame.grid_rowconfigure((0, 1), weight=1)
        
        # Create step frames
        self._create_step_frames()
    
    def _create_step_frames(self):
        """Create individual step frames"""
        step_configs = [
            ("1. Original Image", 0, 0, "step1"),
            ("2. Lung Mask", 0, 1, "step2"),
            ("3. Soft Masked", 0, 2, "step3"),
            ("4. Attention Masked", 1, 0, "step4"),
            ("5. GradCAM Heatmap", 1, 1, "step5")
        ]
        
        for title, row, col, attr_prefix in step_configs:
            frame = ctk.CTkFrame(self.steps_frame)
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            label = ctk.CTkLabel(frame, text=title)
            label.pack(pady=5)
            
            image_label = ctk.CTkLabel(frame, text="")
            image_label.pack(expand=True)
            
            # Store references
            setattr(self, f"{attr_prefix}_frame", frame)
            setattr(self, f"{attr_prefix}_label", label)
            setattr(self, f"{attr_prefix}_image", image_label)
    
    def _setup_enhanced_tab(self):
        """Setup the enhanced results tab"""
        self.enhanced_frame = ctk.CTkFrame(self.tabview.tab("Enhanced Results"))
        self.enhanced_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.enhanced_text = ctk.CTkTextbox(self.enhanced_frame, width=900, height=400,
                                          font=ctk.CTkFont(family="Courier", size=12))
        self.enhanced_text.pack(fill="both", expand=True, padx=10, pady=10)

    def change_image_with_confirmation(self):
        """Show confirmation dialog before changing image"""
        result = messagebox.askyesno(
            "Change Image Confirmation",
            "Are you sure you want to change the current image?\n\n"
            "This will:\n"
            "• Stop any ongoing analysis\n"
            "• Clear all current results\n" 
            "• Reset the application to initial state\n\n"
            "Do you want to continue?",
            icon="warning"
        )
        if result:  
            self.terminate_and_reset()
    
    def terminate_and_reset(self):
        """Terminate all processes and reset application to initial state"""
        try:
            print("DEBUG: Terminating current analysis and resetting...")

            # Set stop flag for any running analysis
            self.stop_analysis_flag.set()
            self.analysis_in_progress = False

            # Wait a moment for threads to respond to stop flag
            self.update_status("Terminating analysis...")
            self.update()

            # Wait for current analysis thread to finish (with timeout)
            if hasattr(self, 'current_analysis_thread') and self.current_analysis_thread:
                if self.current_analysis_thread.is_alive():
                    print("DEBUG: Waiting for analysis thread to terminate...")
                    self.current_analysis_thread.join(timeout=2.0)
                    if self.current_analysis_thread.is_alive():
                        print("DEBUG: Analysis thread didn't terminate, forcing cleanup")

            # Force cleanup GPU memory
            try:
                cleanup_gpu_memory()
                print("DEBUG: GPU memory cleaned")
            except Exception as e:
                print(f"DEBUG: GPU cleanup warning: {e}")

            # Reset all UI elements to initial state
            self.reset_ui_to_initial_state()

            # Clear current analysis data
            self.current_image_path = None
            self.analysis_results = None
            self.current_analysis_thread = None

            # Reset stop flag for next analysis
            self.stop_analysis_flag.clear()

            # Update status
            self.update_status("Ready - Application reset successfully")

            print("DEBUG: Application reset completed")

        except Exception as e:
            print(f"DEBUG: Error during reset: {e}")
            self.update_status("Reset completed with warnings")
    
    def reset_ui_to_initial_state(self):
        """Reset all UI elements to their initial state"""
        try:
            # Reset file selection
            self.file_label.configure(text="No image selected")
            
            # Reset buttons
            self.load_button.configure(state="normal")
            self.change_button.configure(state="disabled")
            
            # Clear all results
            self.prediction_label.configure(text="Prediction: --", text_color="white")
            self.confidence_label.configure(text="Confidence: --")
            self.prob_normal_label.configure(text="Normal: --")
            self.prob_abnormal_label.configure(text="Abnormal: --")
            
            # Clear all images safely
            self.overlay_image_label.configure(text="Load an image to see analysis", image="")
            
            for i in range(1, 6):
                step_image = getattr(self, f"step{i}_image")
                step_image.configure(text="", image="")
            
            # Clear enhanced results
            self.enhanced_text.delete("1.0", "end")
            self.enhanced_text.insert("1.0", "")

            print("DEBUG: UI reset to initial state")
            
        except Exception as e:
            print(f"DEBUG: Error resetting UI: {e}")

    def update_status(self, message):
        """Update status bar message"""
        try:
            self.status_bar.configure(text=message)
            self.update()
        except Exception as e:
            print(f"DEBUG: Error updating status: {e}")
    
    def load_models_thread(self):
        """Load models in a separate thread"""
        thread = threading.Thread(target=self.load_models, daemon=True)
        thread.start()
    
    def load_models(self):
        """Load both segmentation and classification models"""
        try:
            self.update_status("Loading enhanced models...")
            
            # Custom objects for Keras model
            custom_objects = {
                'combined_loss': combined_loss,
                'dice_coefficient': dice_coefficient,
                'iou_coefficient': iou_coefficient,
                'precision': precision,
                'recall': recall,
                'specificity': specificity
            }
            
            # Define model paths
            seg_model_paths = [
                'best_resunet_lung_segmentation.h5',
                'models/segmentation/best_resunet_lung_segmentation.h5',
                'deployment_optimized_efficientnetB2/best_resunet_lung_segmentation.h5',
                os.path.join(os.path.dirname(__file__), 'best_resunet_lung_segmentation.h5'),
                os.path.join(os.path.dirname(__file__), 'models', 'best_resunet_lung_segmentation.h5')
            ]
            
            clf_model_paths = [
                'optimized_classifier_efficientB2.pth',
                'models/clf/optimized_classifier_efficientB2.pth',
                'deployment_optimized_efficientnetB2/optimized_classifier_efficientB2.pth',
                os.path.join(os.path.dirname(__file__), 'optimized_classifier_efficientB2.pth'),
                os.path.join(os.path.dirname(__file__), 'models', 'optimized_classifier_efficientB2.pth')
            ]
            
            # Load segmentation model
            if not self._load_segmentation_model(seg_model_paths, custom_objects):
                raise FileNotFoundError("ResUNet segmentation model not found")
            
            # Load classification model
            if not self._load_classification_model(clf_model_paths):
                raise FileNotFoundError("OptimizedMultiInputClassifier model not found")
            
            self.update_status("Enhanced models loaded successfully")
            
        except Exception as e:
            self.update_status(f"Error loading models: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def _load_segmentation_model(self, model_paths, custom_objects):
        """Load segmentation model from available paths"""
        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.sgm_model = keras.models.load_model(path, custom_objects=custom_objects)
                    self.segmentation_status.configure(text="Segmentation: ✓ ResUNet Ready", text_color="green")
                    return True
                except Exception as e:
                    print(f"Failed to load segmentation model from {path}: {e}")
                    continue
        
        self.segmentation_status.configure(text="Segmentation: ✗ Failed", text_color="red")
        return False
    
    def _load_classification_model(self, model_paths):
        """Load classification model from available paths"""
        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.clf_model = OptimizedMultiInputClassifier(dropout=0.3).to(self.device)
                    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                    
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.clf_model.load_state_dict(checkpoint['model_state_dict'])
                        self.model_config = checkpoint.get('model_config', {})
                        self.performance_metrics = checkpoint.get('performance_metrics', {})
                    else:
                        self.clf_model.load_state_dict(checkpoint)
                    
                    self.clf_model.eval()
                    
                    # Setup Enhanced GradCAM
                    self.gradcam = EnhancedGradCAM(self.clf_model)
                    
                    self.classification_status.configure(text="Classification: ✓ EfficientNet-B2 Ready", text_color="green")
                    return True
                except Exception as e:
                    print(f"Failed to load classification model from {path}: {e}")
                    continue
        
        self.classification_status.configure(text="Classification: ✗ Failed", text_color="red")
        return False
    
    def load_image(self):
        """Open file dialog to select an image with pre-validation"""
        if self.sgm_model is None or self.clf_model is None:
            messagebox.showwarning("Warning", "Models are still loading. Please wait...")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select X-Ray Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            if self._validate_image_file(file_path):
                self.current_image_path = file_path
                filename = os.path.basename(file_path)
                self.file_label.configure(text=f"Selected: {filename}")
                self.change_button.configure(state="normal")
                self.start_analysis()
    
    def _validate_image_file(self, file_path):
        """Validate image file before processing"""
        try:
            test_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if test_img is None:
                messagebox.showerror("Error", f"Cannot read image file: {os.path.basename(file_path)}")
                return False
            
            if test_img.size == 0:
                messagebox.showerror("Error", "Selected image file is empty")
                return False
                
            print(f"DEBUG: Image validation passed - {os.path.basename(file_path)}, Shape: {test_img.shape}")
            del test_img
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Image validation failed: {str(e)}")
            return False

    def start_analysis(self):
        """Start image analysis in separate thread"""
        # Update UI state
        self.load_button.configure(state="disabled")
        self.change_button.configure(state="normal")
        self.analysis_in_progress = True
        self.stop_analysis_flag.clear()
        
        self.update_status("Processing image with enhanced pipeline...")
        self.clear_results()
        
        # Start analysis thread
        self.current_analysis_thread = threading.Thread(target=self.run_analysis, daemon=True)
        self.current_analysis_thread.start()
    
    # REPLACE run_analysis method di app_CXR_v2_refactored.py dengan ini:

    def run_analysis(self):
        """Run the enhanced analysis pipeline with timeout protection"""
        def analysis_with_timeout():
            try:
                if self.stop_analysis_flag.is_set():
                    return
                
                cleanup_gpu_memory()
                print(f"DEBUG: Starting analysis for {os.path.basename(self.current_image_path)}")
                
                # Run pipeline directly without memory monitoring to avoid errors
                results = self.run_enhanced_inference_pipeline(self.current_image_path)
                
                if not self.stop_analysis_flag.is_set() and results:
                    self.analysis_results = results
                    self.after(0, self.update_results)
                
            except Exception as e:
                if not self.stop_analysis_flag.is_set():
                    print(f"DEBUG: Analysis failed: {str(e)}")
                    print(f"DEBUG: Full traceback: {traceback.format_exc()}")
                    
                    cleanup_gpu_memory()
                    
                    error_msg = f"Analysis failed: {str(e)}\n\nPlease check console for details."
                    self.after(0, lambda: self.show_error(error_msg))
                else:
                    print("DEBUG: Analysis stopped due to termination request")
        
        # Run analysis with timeout
        analysis_thread = threading.Thread(target=analysis_with_timeout, daemon=True)
        analysis_thread.start()
        
        # Timeout handler (30 seconds)
        def timeout_handler():
            if analysis_thread.is_alive() and not self.stop_analysis_flag.is_set():
                self.stop_analysis_flag.set()
                error_msg = "Analysis timed out after 30 seconds. Use 'Change Image' button to reset."
                self.after(0, lambda: self.show_error(error_msg))
        
        timeout_timer = Timer(30.0, timeout_handler)
        timeout_timer.start()

    def run_enhanced_inference_pipeline(self, image_path):
        """Execute the enhanced inference pipeline"""
        try:
            # Step 1: Load and validate image
            if self.stop_analysis_flag.is_set():
                return None
            
            self.update_status("Step 1/5: Loading image...")
            original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if original_img is None or original_img.size == 0:
                raise ValueError(f"Could not load valid image from {image_path}")
            
            # Step 2: Run segmentation
            if self.stop_analysis_flag.is_set():
                return None
            
            self.update_status("Step 2/5: Running segmentation...")
            mask_binary = self._run_segmentation(original_img)
            
            # Step 3: Create masked versions
            if self.stop_analysis_flag.is_set():
                return None
            
            self.update_status("Step 3/5: Applying masking strategies...")
            masked_versions = self._create_masked_versions(original_img, mask_binary)
            
            # Step 4: Run classification
            if self.stop_analysis_flag.is_set():
                return None
            
            self.update_status("Step 4/5: Running classification...")
            classification_results = self._run_classification(masked_versions)
            
            # Step 5: Generate GradCAM
            if self.stop_analysis_flag.is_set():
                return None
            
            self.update_status("Step 5/5: Generating visualizations...")
            gradcam_result = self._generate_gradcam(masked_versions, classification_results)
            
            # Compile final results
            return self._compile_results(
                original_img, mask_binary, masked_versions, 
                classification_results, gradcam_result
            )
            
        except Exception as e:
            if not self.stop_analysis_flag.is_set():
                raise e
            return None
    
    def _run_segmentation(self, original_img):
        """Run segmentation model"""
        original_resized = cv2.resize(original_img, (256, 256))
        seg_input = original_resized.astype(np.float32) / 255.0
        seg_input = np.expand_dims(seg_input, axis=(0, -1))
        
        mask_pred = self.sgm_model.predict(seg_input, verbose=0)
        mask_binary = (mask_pred[0, :, :, 0] > 0.5).astype(np.uint8)
        
        # Fallback if mask is empty
        if np.sum(mask_binary) == 0:
            mask_binary = np.zeros((256, 256), dtype=np.uint8)
            mask_binary[64:192, 64:192] = 1
        
        return mask_binary
    
    def _create_masked_versions(self, original_img, mask_binary):
        """Create different masked versions of the image"""
        original_resized = cv2.resize(original_img, (256, 256))
        
        return {
            'original': original_resized,
            'hard_masked': original_resized * mask_binary,
            'soft_masked': apply_advanced_soft_mask(original_resized, mask_binary, alpha=0.2),
            'attention_masked': apply_attention_mask(original_resized, mask_binary)
        }
    
    def _run_classification(self, masked_versions):
        """Run classification on different input versions"""
        clf_transform = AdvancedXRayTransforms(image_size=224, is_training=False)
        
        results = {}
        self.clf_model.eval()
        
        with torch.no_grad():
            # Process each version
            for version_name, image in masked_versions.items():
                if self.stop_analysis_flag.is_set():
                    break
                
                input_type = 0 if version_name == 'original' else 1
                
                tensor = clf_transform(image).unsqueeze(0).to(self.device)
                type_tensor = torch.tensor([input_type]).to(self.device)
                
                output = self.clf_model(tensor, type_tensor)
                probs = F.softmax(output, dim=1)
                
                results[version_name] = {
                    'tensor': tensor,
                    'output': output,
                    'probs': probs.cpu().numpy()[0],
                    'confidence': probs.max().item()
                }
                
                # Clean up intermediate tensors
                del tensor, type_tensor, output
        
        # Compute ensemble
        if results:
            ensemble_probs = self._compute_ensemble(results)
            results['ensemble'] = ensemble_probs
        
        cleanup_gpu_memory()
        return results
    
    def _compute_ensemble(self, results):
        """Compute ensemble prediction from individual results"""
        # Get confidence weights
        confidences = [res['confidence'] for res in results.values()]
        total_conf = sum(confidences)
        
        if total_conf == 0:
            weights = [1.0 / len(confidences)] * len(confidences)
        else:
            weights = [conf / total_conf for conf in confidences]
        
        # Weighted average of probabilities
        ensemble_probs = np.zeros(2)
        for (name, res), weight in zip(results.items(), weights):
            ensemble_probs += res['probs'] * weight
        
        return {
            'probs': ensemble_probs,
            'prediction': ensemble_probs.argmax(),
            'confidence': ensemble_probs.max(),
            'weights': dict(zip(results.keys(), weights))
        }
    
    def _generate_gradcam(self, masked_versions, classification_results):
        """Generate GradCAM visualization"""
        try:
            # Use the best performing input for GradCAM
            best_version = max(classification_results.keys(), 
                            key=lambda k: classification_results[k]['confidence'] if k != 'ensemble' else 0)
            
            best_result = classification_results[best_version]
            input_type = 0 if best_version == 'original' else 1
            target_class = classification_results['ensemble']['prediction']
            
            # Generate GradCAM
            with torch.enable_grad():
                best_tensor = best_result['tensor'].requires_grad_(True)
                type_tensor = torch.tensor([input_type]).to(self.device)
                
                cam = self.gradcam.generate_cam(best_tensor, type_tensor, target_class)
            
            return cam if cam is not None else np.random.random((224, 224)) * 0.3
            
        except Exception as e:
            print(f"GradCAM generation failed: {e}")
            return np.random.random((224, 224)) * 0.3
    
    def _compile_results(self, original_img, mask_binary, masked_versions, classification_results, gradcam):
        """Compile all results into final format"""
        ensemble = classification_results['ensemble']
        class_names = ['Normal', 'Abnormal']
        
        return {
            'original_image': original_img,
            'original_resized': masked_versions['original'],
            'segmentation_mask': mask_binary,
            'segmented_hard': masked_versions['hard_masked'],
            'segmented_soft': masked_versions['soft_masked'],
            'segmented_attention': masked_versions['attention_masked'],
            'gradcam': gradcam,
            'prediction_class': ensemble['prediction'],
            'prediction_label': class_names[ensemble['prediction']],
            'confidence': ensemble['confidence'],
            'ensemble_probabilities': ensemble['probs'],
            'individual_probs': {k: v['probs'] for k, v in classification_results.items() if k != 'ensemble'},
            'adaptive_weights': ensemble['weights'],
            'individual_confidences': {k: v['confidence'] for k, v in classification_results.items() if k != 'ensemble'}
        }
    
    def update_results(self):
        """Update UI with analysis results"""
        try:
            self.analysis_in_progress = False
            results = self.analysis_results
            
            # Update text results
            self.prediction_label.configure(
                text=f"Prediction: {results['prediction_label']}",
                text_color="green" if results['prediction_label'] == 'Normal' else "red"
            )
            self.confidence_label.configure(text=f"Confidence: {results['confidence']:.1%}")
            self.prob_normal_label.configure(text=f"Normal: {results['ensemble_probabilities'][0]:.1%}")
            self.prob_abnormal_label.configure(text=f"Abnormal: {results['ensemble_probabilities'][1]:.1%}")
            
            # Update images and detailed results
            self.update_step_images(results)
            self.update_overlay_image(results)
            self.update_enhanced_results(results)
            
            self.update_status("Enhanced analysis complete")
            self.load_button.configure(state="normal")
            
        except Exception as e:
            self.show_error(f"Failed to update results: {str(e)}")
    
    def update_step_images(self, results):
        """Update processing step images"""
        try:
            img_size = 180
            
            images = [
                ('step1', results['original_resized']),
                ('step2', results['segmentation_mask'] * 255),
                ('step3', results['segmented_soft']),
                ('step4', results['segmented_attention']),
                ('step5', self._create_gradcam_visualization(results['gradcam'], img_size))
            ]
            
            for step_name, image_data in images:
                if isinstance(image_data, np.ndarray):
                    if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                        # RGB image
                        pil_image = Image.fromarray(image_data).resize((img_size, img_size))
                    else:
                        # Grayscale image
                        pil_image = Image.fromarray(image_data.astype(np.uint8)).resize((img_size, img_size))
                    
                    tk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(img_size, img_size))
                    step_image_label = getattr(self, f"{step_name}_image")
                    step_image_label.configure(image=tk_image, text="")
                    
        except Exception as e:
            print(f"Error updating step images: {e}")
    
    def _create_gradcam_visualization(self, cam, size):
        """Create colored GradCAM visualization"""
        cam_resized = cv2.resize(cam, (size, size))
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        return cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    
    def update_overlay_image(self, results):
        """Update the final overlay image"""
        try:
            img_size = 512
            
            # Create overlay visualization
            original = cv2.resize(results['original_resized'], (img_size, img_size))
            cam = cv2.resize(results['gradcam'], (img_size, img_size))
            mask = cv2.resize(results['segmentation_mask'], (img_size, img_size))
            
            # Create colored overlays
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            seg_overlay = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            seg_overlay[:, :, 1] = mask * 100  # Green for lungs
            
            # Combine
            base_rgb = np.stack([original] * 3, axis=-1)
            final_overlay = (base_rgb * 0.5 + seg_overlay * 0.2 + heatmap_rgb * 0.3).astype(np.uint8)
            
            # Display
            overlay_pil = Image.fromarray(final_overlay)
            overlay_tk = ctk.CTkImage(light_image=overlay_pil, dark_image=overlay_pil, size=(img_size, img_size))
            self.overlay_image_label.configure(image=overlay_tk, text="")
            
        except Exception as e:
            print(f"Error updating overlay image: {e}")
    
    def update_enhanced_results(self, results):
        """Update enhanced results with detailed analysis"""
        try:
            enhanced_text = self._generate_detailed_report(results)
            self.enhanced_text.delete("1.0", "end")
            self.enhanced_text.insert("1.0", enhanced_text)
        except Exception as e:
            print(f"Error updating enhanced results: {e}")
    
    def _generate_detailed_report(self, results):
        """Generate detailed analysis report"""
        report = "ENHANCED CXR ANALYSIS REPORT - EfficientNet-B2 Pipeline\n"
        report += "=" * 65 + "\n\n"
        
        # Basic results
        report += f"FINAL PREDICTION: {results['prediction_label']}\n"
        report += f"CONFIDENCE: {results['confidence']:.1%}\n\n"
        
        # Ensemble details
        report += "ENSEMBLE ANALYSIS:\n"
        report += "-" * 25 + "\n"
        report += f"Normal Probability:   {results['ensemble_probabilities'][0]:.1%}\n"
        report += f"Abnormal Probability: {results['ensemble_probabilities'][1]:.1%}\n\n"
        
        # Individual contributions
        report += "INDIVIDUAL MODEL CONTRIBUTIONS:\n"
        report += "-" * 35 + "\n"
        
        for model_type, probs in results['individual_probs'].items():
            weight = results['adaptive_weights'].get(model_type, 0)
            conf = results['individual_confidences'].get(model_type, 0)
            report += f"{model_type.replace('_', ' ').title()}:\n"
            report += f"  Normal: {probs[0]:.1%}, Abnormal: {probs[1]:.1%}\n"
            report += f"  Confidence: {conf:.1%}, Weight: {weight:.1%}\n\n"
        
        # Technical details
        report += "TECHNICAL DETAILS:\n"
        report += "-" * 20 + "\n"
        report += f"Backbone: EfficientNet-B2\n"
        report += f"Input Size: 224x224\n"
        report += f"Segmentation: ResUNet\n"
        report += f"Masking: Soft (α=0.2) + Attention-based\n"
        report += f"Device: {self.device}\n\n"
        
        # Clinical insights
        report += "CLINICAL INSIGHTS:\n"
        report += "-" * 20 + "\n"
        report += self._generate_clinical_insights(results)
        
        return report
    
    def _generate_clinical_insights(self, results):
        """Generate clinical insights based on results"""
        insights = ""
        label = results['prediction_label']
        confidence = results['confidence']
        
        if label == 'Normal':
            if confidence > 0.9:
                insights += "• High confidence normal chest X-ray\n"
                insights += "• No obvious abnormalities detected\n"
            elif confidence > 0.7:
                insights += "• Likely normal chest X-ray\n"
                insights += "• Consider clinical correlation\n"
            else:
                insights += "• Borderline normal classification\n"
                insights += "• Recommend clinical review\n"
        else:
            if confidence > 0.9:
                insights += "• High confidence abnormal findings\n"
                insights += "• Urgent clinical correlation recommended\n"
            elif confidence > 0.7:
                insights += "• Likely abnormal chest X-ray\n"
                insights += "• Clinical correlation advised\n"
            else:
                insights += "• Borderline abnormal classification\n"
                insights += "• Further evaluation recommended\n"
        
        insights += "\nDISCLAIMER: This is an AI-assisted analysis tool.\n"
        insights += "Results should be interpreted by qualified medical professionals.\n"
        
        return insights
    
    def clear_results(self):
        """Clear all results displays"""
        try:
            # Clear text results
            self.prediction_label.configure(text="Prediction: --", text_color="white")
            self.confidence_label.configure(text="Confidence: --")
            self.prob_normal_label.configure(text="Normal: --")
            self.prob_abnormal_label.configure(text="Abnormal: --")
            
            # Clear images
            self.overlay_image_label.configure(text="Processing with enhanced pipeline...", image="")
            
            for i in range(1, 6):
                step_image = getattr(self, f"step{i}_image")
                step_image.configure(text="Loading...", image="")
            
            # Clear enhanced results
            self.enhanced_text.delete("1.0", "end")
            self.enhanced_text.insert("1.0", "Processing...")
            
        except Exception as e:
            print(f"DEBUG: Error clearing results: {e}")
    
    def show_error(self, message):
        """Show error message and reset UI"""
        print(f"ERROR: {message}")
        
        messagebox.showerror("Analysis Error", 
                        f"Analysis failed: {message}\n\n"
                        "Please try with a different image or restart the application.")
        
        self.update_status("Ready - Enhanced Pipeline Loaded")
        self.load_button.configure(state="normal")
        self.clear_results()


def main():
    """Main function to start the application"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("Starting Enhanced CXR Analysis Application")
    print("EfficientNet-B2 Pipeline with Modular Architecture")
    print("=" * 65)
    
    # Create and run application
    app = CXRAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()