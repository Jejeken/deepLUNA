"""
GradCAM Implementation for Model Interpretability
=================================================

Enhanced GradCAM specifically optimized for EfficientNet-B2 architecture.
"""

import torch
import torch.nn.functional as F
import numpy as np


class EnhancedGradCAM:
    """
    Enhanced GradCAM implementation optimized for EfficientNet-B2
    
    Provides model interpretability by visualizing important regions
    for classification decisions.
    """
    
    def __init__(self, model, target_layer_name=''):
        """
        Initialize GradCAM
        
        Args:
            model: The classification model
            target_layer_name: Name of target layer (auto-detected if empty)
        """
        self.model = model
        
        # Auto-detect target layer for EfficientNet-B2
        if hasattr(model.backbone, 'blocks'):
            self.target_layer = model.backbone.blocks[-1]  # Last EfficientNet block
        else:
            # Fallback to generic feature layer
            self.target_layer = list(model.backbone.children())[-2]
            
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward activations"""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients"""
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, input_type_tensor, target_class=None):
        """
        Generate Class Activation Map
        
        Args:
            input_image: Input image tensor with gradients enabled
            input_type_tensor: Input type tensor for the model
            target_class: Target class for visualization (auto-detect if None)
            
        Returns:
            CAM as numpy array
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Ensure input requires gradient
        input_image.requires_grad = True
        
        # Forward pass
        output = self.model(input_image, input_type_tensor)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax(1).item()
        
        # Clear previous gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        score = output[0, target_class]
        score.backward()
        
        # Handle case where gradients are not computed
        if self.gradients is None:
            h, w = input_image.shape[2], input_image.shape[3]
            return np.zeros((h, w), dtype=np.float32)
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # Global average pooling of gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight activations with pooled gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        
        # Average across channels to get final CAM
        cam = torch.mean(activations, dim=1).squeeze()
        
        # Apply ReLU to focus on positive contributions
        cam = torch.nn.functional.relu(cam)
        
        # Normalize CAM
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.numpy()
    
    def generate_guided_gradcam(self, input_image, input_type_tensor, target_class=None):
        """
        Generate Guided GradCAM for enhanced visualization
        
        Args:
            input_image: Input image tensor with gradients enabled
            input_type_tensor: Input type tensor for the model
            target_class: Target class for visualization
            
        Returns:
            Guided GradCAM as numpy array
        """
        # Generate regular GradCAM
        cam = self.generate_cam(input_image, input_type_tensor, target_class)
        
        # Get guided gradients (simplified version)
        input_image.grad = None
        self.model.zero_grad()
        
        output = self.model(input_image, input_type_tensor)
        
        if target_class is None:
            target_class = output.argmax(1).item()
        
        score = output[0, target_class]
        score.backward()
        
        guided_gradients = input_image.grad.detach().cpu().numpy()[0, 0]
        
        # Resize CAM to match input size
        h, w = guided_gradients.shape
        cam_resized = np.array(torch.nn.functional.interpolate(
            torch.from_numpy(cam).unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy())
        
        # Element-wise multiplication
        guided_gradcam = guided_gradients * cam_resized
        
        return guided_gradcam
    
    def remove_hooks(self):
        """Remove registered hooks to prevent memory leaks"""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()
    
    def __del__(self):
        """Destructor to ensure hooks are removed"""
        try:
            self.remove_hooks()
        except:
            pass