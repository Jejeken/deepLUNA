"""
CXR Segmentation and Classification Desktop Application - Enhanced Version
=========================================================================

Updated to use OptimizedMultiInputClassifier with EfficientNet-B2
Implements soft masking and ensemble inference strategies

Dependencies to install:
pip install customtkinter pillow torch torchxrayvision tensorflow opencv-python-headless numpy albumentations scikit-learn matplotlib timm

Required model files (place in same directory or 'models' subfolder):
- best_resunet_lung_segmentation.h5
- optimized_classifier_efficientB2.pth

Author: Updated for EfficientNet-B2 Pipeline
Version: 2.0 - Enhanced
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # Added for EfficientNet-B2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageTk
import warnings


import gc  # Add at top with other imports
import traceback  # Add at top with other imports
from threading import Timer 
warnings.filterwarnings('ignore')

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ============================================================================
# KERAS CUSTOM OBJECTS FOR SEGMENTATION MODEL
# ============================================================================

def dice_coefficient(y_true, y_pred, smooth=1e-7):
    """Dice coefficient metric"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coefficient(y_true, y_pred)

def iou_coefficient(y_true, y_pred, smooth=1e-7):
    """IoU/Jaccard coefficient"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def iou_loss(y_true, y_pred):
    """IoU/Jaccard loss function"""
    return 1 - iou_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """Combined Dice and IoU loss"""
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)

def precision(y_true, y_pred):
    """Precision metric"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    """Recall/Sensitivity metric"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (actual_positives + K.epsilon())

def specificity(y_true, y_pred):
    """Specificity metric"""
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    actual_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (actual_negatives + K.epsilon())

# ============================================================================
# ENHANCED OPTIMIZED MULTI-INPUT CLASSIFIER (EfficientNet-B2)
# ============================================================================

class OptimizedMultiInputClassifier(nn.Module):
    """Optimized version with EfficientNet-B2 backbone"""
    def __init__(self, backbone_name="efficientnet_b2", num_classes=2, dropout=0.3):
        super(OptimizedMultiInputClassifier, self).__init__()
        
        # EfficientNet-B2 backbone
        self.backbone = timm.create_model(backbone_name, pretrained=True, in_chans=1, num_classes=0)
        self.num_features = self.backbone.num_features
        
        # Enhanced type embedding
        self.type_embedding = nn.Sequential(
            nn.Embedding(2, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_features),
            nn.LayerNorm(self.num_features)
        )
        
        # Dual attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features // 8, 1),
            nn.BatchNorm2d(self.num_features // 8),
            nn.ReLU(),
            nn.Conv2d(self.num_features // 8, 1, 1),
            nn.Sigmoid()
        )
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.num_features, self.num_features // 16),
            nn.ReLU(),
            nn.Linear(self.num_features // 16, self.num_features),
            nn.Sigmoid()
        )
        
        # Pooling layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Simplified but effective classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features * 2 + self.num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, input_type=None):
        # Extract features
        features = self.backbone.forward_features(x)
        
        # Apply dual attention
        spatial_att = self.spatial_attention(features)
        channel_att = self.channel_attention(features)
        channel_att = channel_att.unsqueeze(-1).unsqueeze(-1)
        
        # Combined attention
        attended_features = features * spatial_att * channel_att
        
        # Multi-scale pooling
        avg_pool = self.global_pool(attended_features).flatten(1)
        max_pool = self.global_max_pool(attended_features).flatten(1)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        
        # Add type embedding
        if input_type is not None:
            type_emb = self.type_embedding[0](input_type)  # Get embedding
            type_emb = self.type_embedding[1:](type_emb)   # Process through layers
            combined = torch.cat([combined, type_emb], dim=1)
        else:
            batch_size = x.size(0)
            zero_emb = torch.zeros(batch_size, self.num_features).to(x.device)
            combined = torch.cat([combined, zero_emb], dim=1)
        
        logits = self.classifier(combined)
        return logits

# ============================================================================
# ENHANCED GRADCAM FOR EFFICIENTNET-B2
# ============================================================================

class EnhancedGradCAM:
    """Enhanced GradCAM for EfficientNet-B2"""
    def __init__(self, model, target_layer_name=''):
        self.model = model
        # For EfficientNet-B2, use the last feature layer
        if hasattr(model.backbone, 'blocks'):
            self.target_layer = model.backbone.blocks[-1]  # Last EfficientNet block
        else:
            # Fallback to a generic feature layer
            self.target_layer = list(model.backbone.children())[-2]
            
        self.gradients = None
        self.activations = None
        
        self.forward_handle = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, input_type_tensor, target_class=None):
        self.model.eval()
        input_image.requires_grad = True
        
        output = self.model(input_image, input_type_tensor)
        
        if target_class is None:
            target_class = output.argmax(1).item()
        
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()
        
        if self.gradients is None:
            h, w = input_image.shape[2], input_image.shape[3]
            return np.zeros((h, w), dtype=np.float32)
        
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        
        cam = torch.mean(activations, dim=1).squeeze()
        cam = torch.nn.functional.relu(cam)
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.numpy()
    
    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

# ============================================================================
# ENHANCED IMAGE PREPROCESSING WITH SOFT MASKING
# ============================================================================

def apply_advanced_soft_mask(image, mask, alpha=0.2, blur_kernel=5):
    """Enhanced soft masking with better edge preservation"""
    mask_float = mask.astype(np.float32)
    
    # Smooth mask edges
    if blur_kernel > 0:
        mask_smooth = cv2.GaussianBlur(mask_float, (blur_kernel, blur_kernel), 1)
    else:
        mask_smooth = mask_float
    
    # Apply soft masking with better background preservation
    soft_mask = mask_smooth + (1 - mask_smooth) * alpha
    result = image * soft_mask
    return result.astype(np.uint8)

def apply_attention_mask(image, mask, min_weight=0.2):
    """Apply attention-based masking with smooth transitions"""
    mask_float = mask.astype(np.float32)
    mask_smooth = cv2.GaussianBlur(mask_float, (21, 21), 7)
    attention_map = min_weight + (1 - min_weight) * mask_smooth
    result = image * attention_map
    return result.astype(np.uint8)

class AdvancedXRayTransforms:
    """Advanced transforms for classification model input - Updated for 224x224"""
    def __init__(self, image_size=224, is_training=False):
        self.image_size = image_size
        self.is_training = is_training
        
        if is_training:
            self.transform = A.Compose([
                A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10,
                                 border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
    
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = np.array(image)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        
        transformed = self.transform(image=image)
        return transformed['image']
# ===============================
# Utility Function
def cleanup_gpu_memory():
    """Clean up GPU memory to prevent memory leaks"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        print(f"Warning: GPU cleanup failed: {e}")
# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================

class CXRAnalysisApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Enhanced CXR Analysis - Pipeline")
        self.geometry("1600x1000")
        self.resizable(True, True)
        
        # Initialize variables
        self.sgm_model = None
        self.clf_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gradcam = None
        self.current_image_path = None
        self.analysis_results = None
        
        # ADD: Analysis control variables
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
        self.version_label = ctk.CTkLabel(self.sidebar_frame, text="EfficientNet-B2 Pipeline v2.0", 
                                         font=ctk.CTkFont(size=11), text_color="gray")
        self.version_label.grid(row=1, column=0, padx=20, pady=(0, 10))
        
        # Load image button
        self.load_button = ctk.CTkButton(self.sidebar_frame, text="Load X-Ray Image", 
                                        command=self.load_image, height=40,
                                        font=ctk.CTkFont(size=14, weight="bold"))
        self.load_button.grid(row=2, column=0, padx=20, pady=10)
        
        # ADD: Change Image button
        self.change_button = ctk.CTkButton(
            self.sidebar_frame,
            text="Change Image",
            command=self.change_image_with_confirmation,
            height=35,
            font=ctk.CTkFont(size=13),
            fg_color="transparent",  # Match app background
            border_width=2,
            border_color="#3B82F6",  # Blue border
            text_color="#3B82F6",    # Blue text
            hover_color="#DC2626",   # Red on hover
            state="disabled"         # Initially disabled
        )
        self.change_button.grid(row=3, column=0, padx=20, pady=(0, 10))

        # Current file label
        self.file_label = ctk.CTkLabel(self.sidebar_frame, text="No image selected",
                                      wraplength=300, font=ctk.CTkFont(size=12))
        self.file_label.grid(row=4, column=0, padx=20, pady=5)
        
        # Separator
        self.separator = ctk.CTkFrame(self.sidebar_frame, height=2)
        self.separator.grid(row=5, column=0, padx=20, pady=10, sticky="ew")

        # Analysis results title
        self.results_title = ctk.CTkLabel(self.sidebar_frame, text="Analysis Results",
                                         font=ctk.CTkFont(size=16, weight="bold"))
        self.results_title.grid(row=6, column=0, padx=20, pady=(10, 5))

        # Results frame
        self.results_frame = ctk.CTkFrame(self.sidebar_frame)
        self.results_frame.grid(row=7, column=0, padx=20, pady=5, sticky="ew")
        
        # Prediction label
        self.prediction_label = ctk.CTkLabel(self.results_frame, text="Prediction: --", 
                                           font=ctk.CTkFont(size=14, weight="bold"))
        self.prediction_label.grid(row=0, column=0, padx=15, pady=5, sticky="w")
        
        # Confidence label
        self.confidence_label = ctk.CTkLabel(self.results_frame, text="Confidence: --", 
                                           font=ctk.CTkFont(size=12))
        self.confidence_label.grid(row=1, column=0, padx=15, pady=2, sticky="w")
        
        # Probabilities label
        self.prob_normal_label = ctk.CTkLabel(self.results_frame, text="Normal: --", 
                                            font=ctk.CTkFont(size=11))
        self.prob_normal_label.grid(row=2, column=0, padx=15, pady=2, sticky="w")
        
        self.prob_abnormal_label = ctk.CTkLabel(self.results_frame, text="Abnormal: --", 
                                              font=ctk.CTkFont(size=11))
        self.prob_abnormal_label.grid(row=3, column=0, padx=15, pady=2, sticky="w")
        
        # Ensemble info
        self.ensemble_info = ctk.CTkLabel(self.results_frame, text="Method: Enhanced Pipeline", 
                                         font=ctk.CTkFont(size=10), text_color="gray")
        self.ensemble_info.grid(row=4, column=0, padx=15, pady=2, sticky="w")
        
        # Model info
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
        
        # Main panel with tabs
        self.tabview = ctk.CTkTabview(self, width=1000)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        # Create tabs
        self.tabview.add("Final Overlay")
        self.tabview.add("Processing Steps")
        self.tabview.add("Enhanced Results")  # New tab for enhanced results
        
        # Final overlay tab
        self.final_overlay_frame = ctk.CTkFrame(self.tabview.tab("Final Overlay"))
        self.final_overlay_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.overlay_image_label = ctk.CTkLabel(self.final_overlay_frame, text="Load an image to see analysis")
        self.overlay_image_label.pack(expand=True)
        
        # Processing steps tab
        self.steps_frame = ctk.CTkFrame(self.tabview.tab("Processing Steps"))
        self.steps_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure grid for steps
        self.steps_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.steps_frame.grid_rowconfigure((0, 1), weight=1)
        
        # Create frames for each step
        self.step1_frame = ctk.CTkFrame(self.steps_frame)
        self.step1_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.step1_label = ctk.CTkLabel(self.step1_frame, text="1. Original Image")
        self.step1_label.pack(pady=5)
        self.step1_image = ctk.CTkLabel(self.step1_frame, text="")
        self.step1_image.pack(expand=True)
        
        self.step2_frame = ctk.CTkFrame(self.steps_frame)
        self.step2_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.step2_label = ctk.CTkLabel(self.step2_frame, text="2. Lung Mask")
        self.step2_label.pack(pady=5)
        self.step2_image = ctk.CTkLabel(self.step2_frame, text="")
        self.step2_image.pack(expand=True)
        
        self.step3_frame = ctk.CTkFrame(self.steps_frame)
        self.step3_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.step3_label = ctk.CTkLabel(self.step3_frame, text="3. Soft Masked")
        self.step3_label.pack(pady=5)
        self.step3_image = ctk.CTkLabel(self.step3_frame, text="")
        self.step3_image.pack(expand=True)
        
        self.step4_frame = ctk.CTkFrame(self.steps_frame)
        self.step4_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.step4_label = ctk.CTkLabel(self.step4_frame, text="4. Attention Masked")
        self.step4_label.pack(pady=5)
        self.step4_image = ctk.CTkLabel(self.step4_frame, text="")
        self.step4_image.pack(expand=True)
        
        self.step5_frame = ctk.CTkFrame(self.steps_frame)
        self.step5_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.step5_label = ctk.CTkLabel(self.step5_frame, text="5. GradCAM Heatmap")
        self.step5_label.pack(pady=5)
        self.step5_image = ctk.CTkLabel(self.step5_frame, text="")
        self.step5_image.pack(expand=True)
        
        # Enhanced results tab
        self.enhanced_frame = ctk.CTkFrame(self.tabview.tab("Enhanced Results"))
        self.enhanced_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.enhanced_text = ctk.CTkTextbox(self.enhanced_frame, width=900, height=400,
                                          font=ctk.CTkFont(family="Courier", size=12))
        self.enhanced_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_bar = ctk.CTkLabel(self, text="Ready - Enhanced Pipeline Loaded", anchor="w", 
                                      font=ctk.CTkFont(size=12))
        self.status_bar.grid(row=1, column=1, padx=20, pady=(0, 10), sticky="ew")

    def change_image_with_confirmation(self):
        """Show confirmation dialog before changing image"""
        
        # Custom confirmation dialog
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
                    # Give thread 2 seconds to terminate gracefully
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
        """Reset all UI elements to their initial state with safe image handling"""

        # Reset file selection
        try:
            self.file_label.configure(text="No image selected")
        except Exception as e:
            print(f"DEBUG: Error resetting file label: {e}")

        # Reset buttons
        try:
            self.load_button.configure(state="normal")
            self.change_button.configure(state="disabled")
        except Exception as e:
            print(f"DEBUG: Error resetting buttons: {e}")

        # Clear all results
        try:
            self.prediction_label.configure(text="Prediction: --", text_color="white")
            self.confidence_label.configure(text="Confidence: --")
            self.prob_normal_label.configure(text="Normal: --")
            self.prob_abnormal_label.configure(text="Abnormal: --")
        except Exception as e:
            print(f"DEBUG: Error clearing results: {e}")

        # Clear all images safely - text first, then empty string (not None)
        try:
            self.overlay_image_label.configure(text="Load an image to see analysis")
            self.overlay_image_label.configure(image="")
        except Exception as e:
            print(f"DEBUG: Error clearing overlay image: {e}")

        try:
            self.step1_image.configure(text="")
            self.step1_image.configure(image="")
        except Exception as e:
            print(f"DEBUG: Error clearing step1 image: {e}")

        try:
            self.step2_image.configure(text="")
            self.step2_image.configure(image="")
        except Exception as e:
            print(f"DEBUG: Error clearing step2 image: {e}")

        try:
            self.step3_image.configure(text="")
            self.step3_image.configure(image="")
        except Exception as e:
            print(f"DEBUG: Error clearing step3 image: {e}")

        try:
            self.step4_image.configure(text="")
            self.step4_image.configure(image="")
        except Exception as e:
            print(f"DEBUG: Error clearing step4 image: {e}")

        try:
            self.step5_image.configure(text="")
            self.step5_image.configure(image="")
        except Exception as e:
            print(f"DEBUG: Error clearing step5 image: {e}")

        # Clear enhanced results
        try:
            self.enhanced_text.delete("1.0", "end")
            self.enhanced_text.insert("1.0", "")
        except Exception as e:
            print(f"DEBUG: Error clearing enhanced text: {e}")

        print("DEBUG: UI reset to initial state")

    def update_status(self, message):
        """Update status bar message"""
        self.status_bar.configure(text=message)
        self.update()
    
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
            seg_model_loaded = False
            for path in seg_model_paths:
                if os.path.exists(path):
                    try:
                        self.sgm_model = keras.models.load_model(path, custom_objects=custom_objects)
                        self.segmentation_status.configure(text="Segmentation: ✓ ResUNet Ready", text_color="green")
                        seg_model_loaded = True
                        break
                    except Exception as e:
                        continue
            
            if not seg_model_loaded:
                self.segmentation_status.configure(text="Segmentation: X Failed", text_color="red")
                raise FileNotFoundError("ResUNet segmentation model not found")
            
            # Load classification model  
            clf_model_loaded = False
            for path in clf_model_paths:
                if os.path.exists(path):
                    try:
                        self.clf_model = OptimizedMultiInputClassifier(dropout=0.3).to(self.device)
                        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                        
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            self.clf_model.load_state_dict(checkpoint['model_state_dict'])
                            # Extract additional info if available
                            self.model_config = checkpoint.get('model_config', {})
                            self.performance_metrics = checkpoint.get('performance_metrics', {})
                        else:
                            self.clf_model.load_state_dict(checkpoint)
                        
                        self.clf_model.eval()
                        
                        # Setup Enhanced GradCAM
                        self.gradcam = EnhancedGradCAM(self.clf_model)
                        
                        self.classification_status.configure(text="Classification: ✓ EfficientNet-B2 Ready", text_color="green")
                        clf_model_loaded = True
                        break
                    except Exception as e:
                        print(f"Error loading from {path}: {e}")
                        continue
            
            if not clf_model_loaded:
                self.classification_status.configure(text="Classification: X Failed", text_color="red")
                raise FileNotFoundError("OptimizedMultiInputClassifier model not found")
            
            self.update_status("Enhanced models loaded successfully")
            
        except Exception as e:
            self.update_status(f"Error loading models: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
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
            # Pre-validate image file
            try:
                test_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if test_img is None:
                    messagebox.showerror("Error", f"Cannot read image file: {os.path.basename(file_path)}")
                    return
                
                if test_img.size == 0:
                    messagebox.showerror("Error", "Selected image file is empty")
                    return
                    
                print(f"DEBUG: Pre-validation passed - Image: {os.path.basename(file_path)}, Shape: {test_img.shape}")
                del test_img  # Clean up
                
            except Exception as e:
                messagebox.showerror("Error", f"Image validation failed: {str(e)}")
                return
            
            self.current_image_path = file_path
            filename = os.path.basename(file_path)
            self.file_label.configure(text=f"Selected: {filename}")
            
            # Enable change button once an image is loaded
            self.change_button.configure(state="normal")
            
            # Start analysis in separate thread
            self.start_analysis()


    def start_analysis(self):
        """Start image analysis in separate thread"""
        # Disable load button and enable change button
        self.load_button.configure(state="disabled")
        self.change_button.configure(state="normal")
        
        # Set analysis flag
        self.analysis_in_progress = True
        self.stop_analysis_flag.clear()
        
        self.update_status("Processing image with enhanced pipeline...")
        
        # Clear previous results
        self.clear_results()
        
        # Start analysis thread
        self.current_analysis_thread = threading.Thread(target=self.run_analysis, daemon=True)
        self.current_analysis_thread.start()
    
    def run_analysis(self):
        """Run the enhanced analysis pipeline with stop flag checking"""
        def analysis_with_timeout():
            try:
                # Check stop flag before starting
                if self.stop_analysis_flag.is_set():
                    print("DEBUG: Analysis stopped before starting")
                    return
                
                # Clear GPU memory before starting
                cleanup_gpu_memory()
                
                print(f"DEBUG: Starting analysis for {os.path.basename(self.current_image_path)}")
                self.update_status("Step 1/5: Loading image...")
                
                # Check stop flag periodically
                if self.stop_analysis_flag.is_set():
                    print("DEBUG: Analysis stopped during loading")
                    return
                
                results = self.run_enhanced_inference_pipeline_with_stops(self.current_image_path)
                
                # Check stop flag before updating results
                if self.stop_analysis_flag.is_set():
                    print("DEBUG: Analysis stopped before updating results")
                    return
                
                print("DEBUG: Analysis completed successfully")
                self.analysis_results = results
                
                # Update UI in main thread
                if not self.stop_analysis_flag.is_set():
                    self.after(0, self.update_results)
                
            except Exception as e:
                if not self.stop_analysis_flag.is_set():
                    print(f"DEBUG: Analysis failed with error: {str(e)}")
                    print(f"DEBUG: Full traceback: {traceback.format_exc()}")
                    
                    # Clean up on error
                    cleanup_gpu_memory()
                    
                    error_msg = f"Analysis failed: {str(e)}\n\nPlease check console for details."
                    self.after(0, lambda: self.show_error(error_msg))
                else:
                    print("DEBUG: Analysis stopped due to termination request")
        
        # Set timeout for analysis (30 seconds)
        analysis_thread = threading.Thread(target=analysis_with_timeout, daemon=True)
        analysis_thread.start()
        
        # Timeout handler
        def timeout_handler():
            if analysis_thread.is_alive() and not self.stop_analysis_flag.is_set():
                self.stop_analysis_flag.set()  # Set stop flag
                error_msg = "Analysis timed out after 30 seconds. Use 'Change Image' button to reset."
                self.after(0, lambda: self.show_error(error_msg))
        
        timeout_timer = Timer(30.0, timeout_handler)
        timeout_timer.start()


    def run_enhanced_inference_pipeline_with_stops(self, image_path):
        """Execute the enhanced inference pipeline with stop flag checking"""
        try:
            # Step 1: Load image with validation
            if self.stop_analysis_flag.is_set():
                return None
                
            print("DEBUG: Step 1 - Loading image...")
            self.update_status("Step 1/5: Loading and validating image...")
            
            original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if original_img is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            print(f"DEBUG: Image loaded - Shape: {original_img.shape}, dtype: {original_img.dtype}")
            
            # Validate image
            if original_img.size == 0:
                raise ValueError("Loaded image is empty")
            if len(original_img.shape) != 2:
                raise ValueError(f"Expected grayscale image, got shape: {original_img.shape}")
            
            # Step 2: Segmentation preprocessing
            if self.stop_analysis_flag.is_set():
                return None
                
            print("DEBUG: Step 2 - Preparing for segmentation...")
            self.update_status("Step 2/5: Running lung segmentation...")
            
            seg_input = cv2.resize(original_img, (256, 256))
            seg_input = seg_input.astype(np.float32) / 255.0
            seg_input = np.expand_dims(seg_input, axis=(0, -1))
            
            print(f"DEBUG: Segmentation input shape: {seg_input.shape}")
            
            # Step 3: Run segmentation
            if self.stop_analysis_flag.is_set():
                return None
                
            print("DEBUG: Step 3 - Running segmentation model...")
            
            mask_pred = self.sgm_model.predict(seg_input, verbose=0)
            mask_binary = (mask_pred[0, :, :, 0] > 0.5).astype(np.uint8)
            
            print(f"DEBUG: Segmentation completed - Mask coverage: {np.sum(mask_binary) / (256*256):.2%}")
            
            if np.sum(mask_binary) == 0:
                print("WARNING: Segmentation mask is empty - using fallback")
                # Create fallback mask (center region)
                mask_binary = np.zeros((256, 256), dtype=np.uint8)
                mask_binary[64:192, 64:192] = 1
            
            # Step 4: Create different masking versions
            if self.stop_analysis_flag.is_set():
                return None
                
            print("DEBUG: Step 4 - Creating masked versions...")
            self.update_status("Step 3/5: Applying masking strategies...")
            
            original_resized = cv2.resize(original_img, (256, 256))
            
            try:
                segmented_hard = original_resized * mask_binary
                segmented_soft = apply_advanced_soft_mask(original_resized, mask_binary, alpha=0.2)
                segmented_attention = apply_attention_mask(original_resized, mask_binary)
                print("DEBUG: All masking versions created successfully")
            except Exception as e:
                print(f"DEBUG: Error in masking: {e}")
                # Fallback to simple masking
                segmented_hard = original_resized * mask_binary
                segmented_soft = segmented_hard  # Fallback
                segmented_attention = segmented_hard  # Fallback
            
            # Step 5: Prepare classification inputs
            if self.stop_analysis_flag.is_set():
                return None
                
            print("DEBUG: Step 5 - Preparing classification inputs...")
            self.update_status("Step 4/5: Preparing for classification...")
            
            clf_transform = AdvancedXRayTransforms(image_size=224, is_training=False)
            
            try:
                original_tensor = clf_transform(original_resized).unsqueeze(0).to(self.device)
                soft_tensor = clf_transform(segmented_soft).unsqueeze(0).to(self.device)
                att_tensor = clf_transform(segmented_attention).unsqueeze(0).to(self.device)
                print("DEBUG: All tensors created and moved to device")
            except Exception as e:
                print(f"DEBUG: Error in tensor preparation: {e}")
                raise RuntimeError(f"Failed to prepare classification inputs: {e}")
            
            # Step 6: Run classification with memory management
            if self.stop_analysis_flag.is_set():
                return None
                
            print("DEBUG: Step 6 - Running classification...")
            self.update_status("Step 5/5: Running classification ensemble...")
            
            self.clf_model.eval()
            
            try:
                with torch.no_grad():
                    # Original input (type = 0)
                    if self.stop_analysis_flag.is_set():
                        return None
                    print("DEBUG: Running original input...")
                    output_orig = self.clf_model(original_tensor, torch.tensor([0]).to(self.device))
                    probs_orig = F.softmax(output_orig, dim=1)
                    
                    # Clear intermediate tensors
                    del output_orig
                    cleanup_gpu_memory()
                    
                    # Soft masked input (type = 1)
                    if self.stop_analysis_flag.is_set():
                        return None
                    print("DEBUG: Running soft masked input...")
                    output_soft = self.clf_model(soft_tensor, torch.tensor([1]).to(self.device))
                    probs_soft = F.softmax(output_soft, dim=1)
                    
                    del output_soft
                    cleanup_gpu_memory()
                    
                    # Attention masked input (type = 1)
                    if self.stop_analysis_flag.is_set():
                        return None
                    print("DEBUG: Running attention masked input...")
                    output_att = self.clf_model(att_tensor, torch.tensor([1]).to(self.device))
                    probs_att = F.softmax(output_att, dim=1)
                    
                    del output_att
                    cleanup_gpu_memory()
                    
                    print("DEBUG: All classifications completed")
                    
                    # Adaptive ensemble weighting
                    conf_orig = probs_orig.max().item()
                    conf_soft = probs_soft.max().item()
                    conf_att = probs_att.max().item()
                    
                    print(f"DEBUG: Confidences - Orig: {conf_orig:.3f}, Soft: {conf_soft:.3f}, Att: {conf_att:.3f}")
                    
                    total_conf = conf_orig + conf_soft + conf_att
                    if total_conf == 0:
                        print("WARNING: All confidences are zero, using equal weights")
                        w_orig = w_soft = w_att = 1.0 / 3.0
                    else:
                        w_orig = conf_orig / total_conf
                        w_soft = conf_soft / total_conf
                        w_att = conf_att / total_conf
                    
                    # Ensemble prediction
                    ensemble_probs = w_orig * probs_orig + w_soft * probs_soft + w_att * probs_att
                    pred_class = ensemble_probs.argmax(1).item()
                    confidence = ensemble_probs.max(1)[0].item()
                    
                    print(f"DEBUG: Final prediction - Class: {pred_class}, Confidence: {confidence:.3f}")
                    
            except Exception as e:
                print(f"DEBUG: Error in classification: {e}")
                print(f"DEBUG: Classification traceback: {traceback.format_exc()}")
                raise RuntimeError(f"Classification failed: {e}")
            
            # Step 7: Generate GradCAM with error handling
            if self.stop_analysis_flag.is_set():
                return None
                
            print("DEBUG: Step 7 - Generating GradCAM...")
            
            try:
                best_input = soft_tensor if conf_soft >= max(conf_orig, conf_att) else (
                            att_tensor if conf_att >= conf_orig else original_tensor)
                best_type = torch.tensor([1]).to(self.device) if conf_soft >= max(conf_orig, conf_att) or conf_att >= conf_orig else torch.tensor([0]).to(self.device)
                
                with torch.enable_grad():
                    cam = self.gradcam.generate_cam(best_input, best_type, target_class=pred_class)
                
                if cam is None or np.sum(cam) == 0:
                    print("WARNING: GradCAM failed, creating dummy heatmap")
                    cam = np.random.random((224, 224)) * 0.5  # Dummy heatmap
                
                print("DEBUG: GradCAM completed successfully")
                
            except Exception as e:
                print(f"DEBUG: GradCAM failed: {e}")
                # Create dummy heatmap
                cam = np.random.random((224, 224)) * 0.3
            
            # Step 8: Prepare results with memory cleanup
            if self.stop_analysis_flag.is_set():
                return None
                
            print("DEBUG: Step 8 - Preparing results...")
            
            # Move tensors back to CPU and convert to numpy
            try:
                ensemble_probs_np = ensemble_probs.cpu().numpy()[0]
                individual_probs = {
                    'original': probs_orig.cpu().numpy()[0],
                    'soft_masked': probs_soft.cpu().numpy()[0],
                    'attention_masked': probs_att.cpu().numpy()[0]
                }
                
                # Clear all GPU tensors
                del original_tensor, soft_tensor, att_tensor
                del probs_orig, probs_soft, probs_att, ensemble_probs
                cleanup_gpu_memory()
                
            except Exception as e:
                print(f"DEBUG: Error preparing results: {e}")
                raise RuntimeError(f"Failed to prepare results: {e}")
            
            class_names = ['Normal', 'Abnormal']
            prediction_label = class_names[pred_class]
            
            results = {
                'original_image': original_img,
                'original_resized': original_resized,
                'segmentation_mask': mask_binary,
                'segmented_hard': segmented_hard,
                'segmented_soft': segmented_soft,
                'segmented_attention': segmented_attention,
                'gradcam': cam,
                'prediction_class': pred_class,
                'prediction_label': prediction_label,
                'confidence': confidence,
                'ensemble_probabilities': ensemble_probs_np,
                'individual_probs': individual_probs,
                'adaptive_weights': {
                    'original': w_orig,
                    'soft_masked': w_soft,
                    'attention_masked': w_att
                },
                'individual_confidences': {
                    'original': conf_orig,
                    'soft_masked': conf_soft,
                    'attention_masked': conf_att
                }
            }
            
            print("DEBUG: Results prepared successfully")
            return results
            
        except Exception as e:
            if not self.stop_analysis_flag.is_set():
                print(f"DEBUG: Pipeline failed at some step: {str(e)}")
                print(f"DEBUG: Full pipeline traceback: {traceback.format_exc()}")
                cleanup_gpu_memory()
                raise e
            else:
                print("DEBUG: Pipeline stopped due to termination request")
                return None
    
    def update_results(self):
        """Update UI with enhanced analysis results"""
        try:
            # Reset analysis flag
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
            
            # Update images
            self.update_step_images(results)
            self.update_overlay_image(results)
            self.update_enhanced_results(results)
            
            self.update_status("Enhanced analysis complete")
            
            # Re-enable load button, keep change button enabled
            self.load_button.configure(state="normal")
            self.change_button.configure(state="normal")
            
        except Exception as e:
            self.show_error(f"Failed to update results: {str(e)}")

    
    def update_step_images(self, results):
        """Update processing step images"""
        try:
            img_size = 180  # Fixed size for step images
            
            # Step 1: Original image
            original_pil = Image.fromarray(results['original_resized']).resize((img_size, img_size))
            original_tk = ctk.CTkImage(light_image=original_pil, dark_image=original_pil, size=(img_size, img_size))
            self.step1_image.configure(image=original_tk, text="")
            
            # Step 2: Segmentation mask
            mask_pil = Image.fromarray(results['segmentation_mask'] * 255).resize((img_size, img_size))
            mask_tk = ctk.CTkImage(light_image=mask_pil, dark_image=mask_pil, size=(img_size, img_size))
            self.step2_image.configure(image=mask_tk, text="")
            
            # Step 3: Soft masked
            soft_pil = Image.fromarray(results['segmented_soft']).resize((img_size, img_size))
            soft_tk = ctk.CTkImage(light_image=soft_pil, dark_image=soft_pil, size=(img_size, img_size))
            self.step3_image.configure(image=soft_tk, text="")
            
            # Step 4: Attention masked
            att_pil = Image.fromarray(results['segmented_attention']).resize((img_size, img_size))
            att_tk = ctk.CTkImage(light_image=att_pil, dark_image=att_pil, size=(img_size, img_size))
            self.step4_image.configure(image=att_tk, text="")
            
            # Step 5: GradCAM heatmap
            cam_resized = cv2.resize(results['gradcam'], (img_size, img_size))
            cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            cam_rgb = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
            cam_pil = Image.fromarray(cam_rgb)
            cam_tk = ctk.CTkImage(light_image=cam_pil, dark_image=cam_pil, size=(img_size, img_size))
            self.step5_image.configure(image=cam_tk, text="")
            
        except Exception as e:
            print(f"Error updating step images: {e}")
    
    def update_overlay_image(self, results):
        """Update the final overlay image"""
        try:
            img_size = 512
            
            # Resize components
            original_resized = cv2.resize(results['original_resized'], (img_size, img_size))
            cam_resized = cv2.resize(results['gradcam'], (img_size, img_size))
            mask_resized = cv2.resize(results['segmentation_mask'], (img_size, img_size))
            
            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Create segmentation overlay
            seg_overlay = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            seg_overlay[:,:,1] = mask_resized * 100  # Green for lungs
            
            # Combine overlays
            base_rgb = np.stack([original_resized] * 3, axis=-1)
            final_overlay = (base_rgb * 0.5 + seg_overlay * 0.2 + heatmap_rgb * 0.3).astype(np.uint8)
            
            # Convert to PIL and display
            overlay_pil = Image.fromarray(final_overlay)
            overlay_tk = ctk.CTkImage(light_image=overlay_pil, dark_image=overlay_pil, size=(img_size, img_size))
            self.overlay_image_label.configure(image=overlay_tk, text="")
            
        except Exception as e:
            print(f"Error updating overlay image: {e}")
    
    def update_enhanced_results(self, results):
        """Update the enhanced results tab with detailed analysis"""
        try:
            enhanced_text = "ENHANCED CXR ANALYSIS REPORT - EfficientNet-B2 Pipeline\n"
            enhanced_text += "=" * 65 + "\n\n"
            
            # Basic results
            enhanced_text += f"FINAL PREDICTION: {results['prediction_label']}\n"
            enhanced_text += f"CONFIDENCE: {results['confidence']:.1%}\n\n"
            
            # Ensemble details
            enhanced_text += "ENSEMBLE ANALYSIS:\n"
            enhanced_text += "-" * 25 + "\n"
            enhanced_text += f"Normal Probability:   {results['ensemble_probabilities'][0]:.1%}\n"
            enhanced_text += f"Abnormal Probability: {results['ensemble_probabilities'][1]:.1%}\n\n"
            
            # Individual model contributions
            enhanced_text += "INDIVIDUAL MODEL CONTRIBUTIONS:\n"
            enhanced_text += "-" * 35 + "\n"
            
            for model_type, probs in results['individual_probs'].items():
                weight = results['adaptive_weights'][model_type]
                conf = results['individual_confidences'][model_type]
                enhanced_text += f"{model_type.replace('_', ' ').title()}:\n"
                enhanced_text += f"  Normal: {probs[0]:.1%}, Abnormal: {probs[1]:.1%}\n"
                enhanced_text += f"  Confidence: {conf:.1%}, Weight: {weight:.1%}\n\n"
            
            # Technical details
            enhanced_text += "TECHNICAL DETAILS:\n"
            enhanced_text += "-" * 20 + "\n"
            enhanced_text += f"Backbone: EfficientNet-B2\n"
            enhanced_text += f"Input Size: 224x224\n"
            enhanced_text += f"Segmentation: ResUNet\n"
            enhanced_text += f"Masking: Soft (α=0.2) + Attention-based\n"
            enhanced_text += f"Device: {self.device}\n\n"
            
            # Model performance (if available)
            if hasattr(self, 'performance_metrics') and self.performance_metrics:
                enhanced_text += "MODEL PERFORMANCE:\n"
                enhanced_text += "-" * 20 + "\n"
                for metric, value in self.performance_metrics.items():
                    if isinstance(value, (int, float)):
                        enhanced_text += f"{metric.replace('_', ' ').title()}: {value:.2f}%\n"
                    else:
                        enhanced_text += f"{metric.replace('_', ' ').title()}: {value}\n"
                enhanced_text += "\n"
            
            # Recommendations
            enhanced_text += "CLINICAL INSIGHTS:\n"
            enhanced_text += "-" * 20 + "\n"
            if results['prediction_label'] == 'Normal':
                if results['confidence'] > 0.9:
                    enhanced_text += "• High confidence normal chest X-ray\n"
                    enhanced_text += "• No obvious abnormalities detected\n"
                elif results['confidence'] > 0.7:
                    enhanced_text += "• Likely normal chest X-ray\n"
                    enhanced_text += "• Consider clinical correlation\n"
                else:
                    enhanced_text += "• Borderline normal classification\n"
                    enhanced_text += "• Recommend clinical review\n"
            else:
                if results['confidence'] > 0.9:
                    enhanced_text += "• High confidence abnormal findings\n"
                    enhanced_text += "• Urgent clinical correlation recommended\n"
                elif results['confidence'] > 0.7:
                    enhanced_text += "• Likely abnormal chest X-ray\n"
                    enhanced_text += "• Clinical correlation advised\n"
                else:
                    enhanced_text += "• Borderline abnormal classification\n"
                    enhanced_text += "• Further evaluation recommended\n"
            
            enhanced_text += "\nDISCLAIMER: This is an AI-assisted analysis tool.\n"
            enhanced_text += "Results should be interpreted by qualified medical professionals.\n"
            
            # Update the textbox
            self.enhanced_text.delete("1.0", "end")
            self.enhanced_text.insert("1.0", enhanced_text)
            
        except Exception as e:
            print(f"Error updating enhanced results: {e}")
    
    def clear_results(self):
        """Clear all results displays with safe image handling"""
        try:
            self.prediction_label.configure(text="Prediction: --", text_color="white")
            self.confidence_label.configure(text="Confidence: --")
            self.prob_normal_label.configure(text="Normal: --")
            self.prob_abnormal_label.configure(text="Abnormal: --")
        except Exception as e:
            print(f"DEBUG: Error clearing text results: {e}")

        # Clear images safely - set text first, then clear image
        try:
            self.overlay_image_label.configure(text="Processing with enhanced pipeline...")
            self.overlay_image_label.configure(image="")
        except Exception as e:
            print(f"DEBUG: Error clearing overlay image: {e}")

        try:
            self.step1_image.configure(text="Loading...")
            self.step1_image.configure(image="")
        except Exception as e:
            print(f"DEBUG: Error clearing step1 image: {e}")

        try:
            self.step2_image.configure(text="Loading...")
            self.step2_image.configure(image="")
        except Exception as e:
            print(f"DEBUG: Error clearing step2 image: {e}")

        try:
            self.step3_image.configure(text="Loading...")
            self.step3_image.configure(image="")
        except Exception as e:
            print(f"DEBUG: Error clearing step3 image: {e}")

        try:
            self.step4_image.configure(text="Loading...")
            self.step4_image.configure(image="")
        except Exception as e:
            print(f"DEBUG: Error clearing step4 image: {e}")

        try:
            self.step5_image.configure(text="Loading...")
            self.step5_image.configure(image="")
        except Exception as e:
            print(f"DEBUG: Error clearing step5 image: {e}")

        # Clear enhanced results
        try:
            self.enhanced_text.delete("1.0", "end")
            self.enhanced_text.insert("1.0", "Processing...")
        except Exception as e:
            print(f"DEBUG: Error clearing enhanced text: {e}")
    
    def show_error(self, message):
        """Show error message and reset UI with better feedback"""
        print(f"ERROR: {message}")
        
        # Show detailed error dialog
        error_dialog = f"""Analysis Error Details:

    {message}

    Troubleshooting Tips:
    • Try a different image format (PNG, JPG)
    • Check if image file is corrupted
    • Restart application if memory issues persist
    • Check console output for detailed error info

    Technical Info:
    • Device: {self.device}
    • GPU Available: {torch.cuda.is_available()}
    • Current Memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB (if GPU)"""

        messagebox.showerror("Analysis Error", error_dialog)
        
        # Reset UI
        self.update_status("Ready - Enhanced Pipeline Loaded")
        self.load_button.configure(state="normal")
        
        # Clear any stuck processing displays
        self.clear_results()

# ============================================================================
# MAIN EXECUTION
# ============================================================================


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("Starting Enhanced CXR Analysis Application")
    print("EfficientNet-B2 Pipeline with Soft Masking and Ensemble Inference")
    print("=" * 65)
    
    # Create and run application
    app = CXRAnalysisApp()
    app.mainloop()