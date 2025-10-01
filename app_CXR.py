"""
CXR Segmentation and Classification Desktop Application
======================================================

Dependencies to install:
pip install customtkinter pillow torch torchxrayvision tensorflow opencv-python-headless numpy albumentations scikit-learn matplotlib

Required model files (place in same directory or 'models' subfolder):
- best_resunet_lung_segmentation.h5
- enhanced_cxr_classifier_final.pth

Author: Jefta Tarigan
Version: 1.0
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
import torchxrayvision as xrv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageTk
import warnings
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
# PYTORCH CLASSIFICATION MODEL DEFINITION
# ============================================================================

class EnhancedBinaryClassifier(nn.Module):
    """Enhanced model with attention mechanism and better architecture"""
    def __init__(self, backbone_name="densenet121-res224-all", num_classes=2, dropout=0.5):
        super(EnhancedBinaryClassifier, self).__init__()
        
        self.backbone = xrv.models.DenseNet(weights=backbone_name)
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 224, 224)
            backbone_features = self.backbone.features(dummy_input)
            num_features = backbone_features.shape[1]
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_features // 16),
            nn.ReLU(),
            nn.Linear(num_features // 16, num_features),
            nn.Sigmoid()
        )
        
        # Enhanced classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.backbone.classifier = nn.Identity()
    
    def forward(self, x):
        features = self.backbone.features(x)
        
        attention_weights = self.attention(features)
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
        attended_features = features * attention_weights
        
        avg_pool = self.global_pool(attended_features)
        max_pool = self.global_max_pool(attended_features)
        
        combined = torch.cat([avg_pool, max_pool], dim=1)
        combined = torch.flatten(combined, 1)
        
        logits = self.classifier(combined)
        return logits

# ============================================================================
# GRADCAM IMPLEMENTATION
# ============================================================================

class GradCAM:
    """GradCAM implementation for visualizing model predictions"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.forward_handle = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        input_image.requires_grad = True
        
        output = self.model(input_image)
        
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
# IMAGE PREPROCESSING
# ============================================================================

class AdvancedXRayTransforms:
    """Advanced transforms for classification model input"""
    def __init__(self, image_size=224):
        self.image_size = image_size
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
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        
        transformed = self.transform(image=image)
        return transformed['image']

# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================

class CXRAnalysisApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("CXR Segmentation and Classification")
        self.geometry("1400x800")
        self.resizable(True, True)
        
        # Initialize variables
        self.sgm_model = None
        self.clf_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gradcam = None
        self.current_image_path = None
        self.analysis_results = None
        
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
        self.sidebar_frame.grid_rowconfigure(6, weight=1)  # Make results area expandable
        
        # Sidebar title
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="CXR Analysis Tool", 
                                      font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Load image button
        self.load_button = ctk.CTkButton(self.sidebar_frame, text="Load X-Ray Image", 
                                        command=self.load_image, height=40,
                                        font=ctk.CTkFont(size=14, weight="bold"))
        self.load_button.grid(row=1, column=0, padx=20, pady=10)
        
        # Current file label
        self.file_label = ctk.CTkLabel(self.sidebar_frame, text="No image selected", 
                                      wraplength=300, font=ctk.CTkFont(size=12))
        self.file_label.grid(row=2, column=0, padx=20, pady=5)
        
        # Separator
        self.separator = ctk.CTkFrame(self.sidebar_frame, height=2)
        self.separator.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        # Analysis results title
        self.results_title = ctk.CTkLabel(self.sidebar_frame, text="Analysis Results", 
                                         font=ctk.CTkFont(size=16, weight="bold"))
        self.results_title.grid(row=4, column=0, padx=20, pady=(10, 5))
        
        # Results frame
        self.results_frame = ctk.CTkFrame(self.sidebar_frame)
        self.results_frame.grid(row=5, column=0, padx=20, pady=5, sticky="ew")
        
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
        
        # Model info
        self.model_info_frame = ctk.CTkFrame(self.sidebar_frame)
        self.model_info_frame.grid(row=6, column=0, padx=20, pady=(10, 20), sticky="ew")
        
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
        
        # Final overlay tab
        self.final_overlay_frame = ctk.CTkFrame(self.tabview.tab("Final Overlay"))
        self.final_overlay_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.overlay_image_label = ctk.CTkLabel(self.final_overlay_frame, text="Load an image to see analysis")
        self.overlay_image_label.pack(expand=True)
        
        # Processing steps tab
        self.steps_frame = ctk.CTkFrame(self.tabview.tab("Processing Steps"))
        self.steps_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure grid for steps
        self.steps_frame.grid_columnconfigure((0, 1), weight=1)
        self.steps_frame.grid_rowconfigure((0, 1), weight=1)
        
        # Create frames for each step
        self.step1_frame = ctk.CTkFrame(self.steps_frame)
        self.step1_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.step1_label = ctk.CTkLabel(self.step1_frame, text="Original Image")
        self.step1_label.pack(pady=5)
        self.step1_image = ctk.CTkLabel(self.step1_frame, text="")
        self.step1_image.pack(expand=True)
        
        self.step2_frame = ctk.CTkFrame(self.steps_frame)
        self.step2_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.step2_label = ctk.CTkLabel(self.step2_frame, text="Lung Mask")
        self.step2_label.pack(pady=5)
        self.step2_image = ctk.CTkLabel(self.step2_frame, text="")
        self.step2_image.pack(expand=True)
        
        self.step3_frame = ctk.CTkFrame(self.steps_frame)
        self.step3_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.step3_label = ctk.CTkLabel(self.step3_frame, text="Segmented Lungs")
        self.step3_label.pack(pady=5)
        self.step3_image = ctk.CTkLabel(self.step3_frame, text="")
        self.step3_image.pack(expand=True)
        
        self.step4_frame = ctk.CTkFrame(self.steps_frame)
        self.step4_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.step4_label = ctk.CTkLabel(self.step4_frame, text="GradCAM Heatmap")
        self.step4_label.pack(pady=5)
        self.step4_image = ctk.CTkLabel(self.step4_frame, text="")
        self.step4_image.pack(expand=True)
        
        # Status bar
        self.status_bar = ctk.CTkLabel(self, text="Ready", anchor="w", 
                                      font=ctk.CTkFont(size=12))
        self.status_bar.grid(row=1, column=1, padx=20, pady=(0, 10), sticky="ew")
        
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
            self.update_status("Loading models...")
            
            # Custom objects for Keras model
            custom_objects = {
                'combined_loss': combined_loss,
                'dice_coefficient': dice_coefficient,
                'iou_coefficient': iou_coefficient,
                'precision': precision,
                'recall': recall,
                'specificity': specificity
            }
            
            # Try different paths for model files
            seg_model_paths = [
                'best_resunet_lung_segmentation.h5',
                'models/best_resunet_lung_segmentation.h5',
                os.path.join(os.path.dirname(__file__), 'best_resunet_lung_segmentation.h5'),
                os.path.join(os.path.dirname(__file__), 'models', 'best_resunet_lung_segmentation.h5')
            ]
            
            clf_model_paths = [
                'torchxray_densenet_attention.pth',
                'models/torchxray_densenet_attention.pth',
                os.path.join(os.path.dirname(__file__), 'torchxray_densenet_attention.pth'),
                os.path.join(os.path.dirname(__file__), 'models', 'torchxray_densenet_attention.pth')
            ]
            
            # Load segmentation model
            seg_model_loaded = False
            for path in seg_model_paths:
                if os.path.exists(path):
                    try:
                        self.sgm_model = keras.models.load_model(path, custom_objects=custom_objects)
                        self.segmentation_status.configure(text="Segmentation: ✓ Ready", text_color="green")
                        seg_model_loaded = True
                        break
                    except Exception as e:
                        continue
            
            if not seg_model_loaded:
                self.segmentation_status.configure(text="Segmentation: ✗ Failed", text_color="red")
                raise FileNotFoundError("Segmentation model not found")
            
            # Load classification model
            clf_model_loaded = False
            for path in clf_model_paths:
                if os.path.exists(path):
                    try:
                        self.clf_model = EnhancedBinaryClassifier(dropout=0.5).to(self.device)
                        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                        
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            self.clf_model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            self.clf_model.load_state_dict(checkpoint)
                        
                        self.clf_model.eval()
                        
                        # Setup GradCAM
                        target_layer = self.clf_model.backbone.features[-1]
                        self.gradcam = GradCAM(self.clf_model, target_layer)
                        
                        self.classification_status.configure(text="Classification: ✓ Ready", text_color="green")
                        clf_model_loaded = True
                        break
                    except Exception as e:
                        continue
            
            if not clf_model_loaded:
                self.classification_status.configure(text="Classification: ✗ Failed", text_color="red")
                raise FileNotFoundError("Classification model not found")
            
            self.update_status("Models loaded successfully")
            
        except Exception as e:
            self.update_status(f"Error loading models: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def load_image(self):
        """Open file dialog to select an image"""
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
            self.current_image_path = file_path
            filename = os.path.basename(file_path)
            self.file_label.configure(text=f"Selected: {filename}")
            
            # Start analysis in separate thread
            self.start_analysis()
    
    def start_analysis(self):
        """Start image analysis in separate thread"""
        self.load_button.configure(state="disabled")
        self.update_status("Processing image...")
        
        # Clear previous results
        self.clear_results()
        
        # Start analysis thread
        analysis_thread = threading.Thread(target=self.run_analysis, daemon=True)
        analysis_thread.start()
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            results = self.run_inference_pipeline(self.current_image_path)
            self.analysis_results = results
            
            # Update UI in main thread
            self.after(0, self.update_results)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.after(0, lambda: self.show_error(error_msg))
    
    def run_inference_pipeline(self, image_path):
        """Execute the inference pipeline"""
        # Load and preprocess image for segmentation
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Resize and normalize for segmentation model
        seg_input = cv2.resize(original_img, (256, 256))
        seg_input = seg_input.astype(np.float32) / 255.0
        seg_input = np.expand_dims(seg_input, axis=(0, -1))
        
        # Run segmentation inference
        mask_pred = self.sgm_model.predict(seg_input, verbose=0)
        mask_binary = (mask_pred[0, :, :, 0] > 0.5).astype(np.uint8)
        
        # Apply mask to original image
        original_resized = cv2.resize(original_img, (256, 256))
        segmented_img = original_resized * mask_binary
        
        # Preprocess segmented image for classification
        clf_transform = AdvancedXRayTransforms(image_size=224)
        clf_input = clf_transform(segmented_img)
        clf_input = clf_input.unsqueeze(0).to(self.device)
        
        # Run classification inference
        with torch.no_grad():
            clf_output = self.clf_model(clf_input)
            clf_probs = torch.nn.functional.softmax(clf_output, dim=1)
            pred_class = clf_output.argmax(1).item()
            confidence = clf_probs.max(1)[0].item()
        
        # Generate GradCAM
        with torch.enable_grad():
            cam = self.gradcam.generate_cam(clf_input, target_class=pred_class)
        
        # Prepare results
        class_names = ['Normal', 'Abnormal']
        prediction_label = class_names[pred_class]
        
        results = {
            'original_image': original_img,
            'original_resized': original_resized,
            'segmentation_mask': mask_binary,
            'segmented_image': segmented_img,
            'gradcam': cam,
            'prediction_class': pred_class,
            'prediction_label': prediction_label,
            'confidence': confidence,
            'probabilities': clf_probs.cpu().numpy()[0]
        }
        
        return results
    
    def update_results(self):
        """Update UI with analysis results"""
        try:
            results = self.analysis_results
            
            # Update text results
            self.prediction_label.configure(
                text=f"Prediction: {results['prediction_label']}",
                text_color="green" if results['prediction_label'] == 'Normal' else "red"
            )
            self.confidence_label.configure(text=f"Confidence: {results['confidence']:.1%}")
            self.prob_normal_label.configure(text=f"Normal: {results['probabilities'][0]:.1%}")
            self.prob_abnormal_label.configure(text=f"Abnormal: {results['probabilities'][1]:.1%}")
            
            # Update images
            self.update_step_images(results)
            self.update_overlay_image(results)
            
            self.update_status("Analysis complete")
            self.load_button.configure(state="normal")
            
        except Exception as e:
            self.show_error(f"Failed to update results: {str(e)}")
    
    def update_step_images(self, results):
        """Update processing step images"""
        try:
            # Convert images to display format
            img_size = 200  # Fixed size for step images
            
            # Original image
            original_pil = Image.fromarray(results['original_resized']).resize((img_size, img_size))
            original_tk = ctk.CTkImage(light_image=original_pil, dark_image=original_pil, size=(img_size, img_size))
            self.step1_image.configure(image=original_tk, text="")
            
            # Segmentation mask
            mask_pil = Image.fromarray(results['segmentation_mask'] * 255).resize((img_size, img_size))
            mask_tk = ctk.CTkImage(light_image=mask_pil, dark_image=mask_pil, size=(img_size, img_size))
            self.step2_image.configure(image=mask_tk, text="")
            
            # Segmented image
            segmented_pil = Image.fromarray(results['segmented_image']).resize((img_size, img_size))
            segmented_tk = ctk.CTkImage(light_image=segmented_pil, dark_image=segmented_pil, size=(img_size, img_size))
            self.step3_image.configure(image=segmented_pil, text="")
            
            # GradCAM heatmap
            cam_resized = cv2.resize(results['gradcam'], (img_size, img_size))
            cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            cam_rgb = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
            cam_pil = Image.fromarray(cam_rgb)
            cam_tk = ctk.CTkImage(light_image=cam_pil, dark_image=cam_pil, size=(img_size, img_size))
            self.step4_image.configure(image=cam_tk, text="")
            
        except Exception as e:
            print(f"Error updating step images: {e}")
    
    def update_overlay_image(self, results):
        """Update the final overlay image"""
        try:
            # Create composite overlay image
            img_size = 400
            
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
    
    def clear_results(self):
        """Clear all results displays"""
        self.prediction_label.configure(text="Prediction: --", text_color="white")
        self.confidence_label.configure(text="Confidence: --")
        self.prob_normal_label.configure(text="Normal: --")
        self.prob_abnormal_label.configure(text="Abnormal: --")
        
        # Clear images
        self.overlay_image_label.configure(image=None, text="Processing...")
        self.step1_image.configure(image=None, text="Loading...")
        self.step2_image.configure(image=None, text="Loading...")
        self.step3_image.configure(image=None, text="Loading...")
        self.step4_image.configure(image=None, text="Loading...")
    
    def show_error(self, message):
        """Show error message and reset UI"""
        messagebox.showerror("Error", message)
        self.update_status("Ready")
        self.load_button.configure(state="normal")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create and run application
    app = CXRAnalysisApp()
    app.mainloop()