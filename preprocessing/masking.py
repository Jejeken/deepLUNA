"""
Masking Functions for Image Preprocessing
=========================================

Contains advanced masking techniques for preserving important image context
while focusing on regions of interest.
"""

import numpy as np
import cv2


def apply_advanced_soft_mask(image, mask, alpha=0.2, blur_kernel=5):
    """
    Apply advanced soft masking with better edge preservation
    
    This technique preserves background information by blending the original
    image with the masked region, preventing complete information loss.
    
    Args:
        image: Input grayscale image (numpy array)
        mask: Binary mask (numpy array, same size as image)
        alpha: Blending factor for background preservation (0.0-1.0)
               Lower values preserve more background context
        blur_kernel: Gaussian blur kernel size for edge smoothing
                    Set to 0 to disable blurring
    
    Returns:
        Soft-masked image as uint8 numpy array
    """
    # Convert mask to float for precise calculations
    mask_float = mask.astype(np.float32)
    
    # Apply Gaussian blur for smoother edges (if enabled)
    if blur_kernel > 0:
        # Ensure kernel size is odd
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        mask_smooth = cv2.GaussianBlur(mask_float, (blur_kernel, blur_kernel), 1)
    else:
        mask_smooth = mask_float
    
    # Create soft masking weights
    # Formula: foreground regions get full weight (1.0)
    #          background regions get reduced weight (alpha)
    soft_mask = mask_smooth + (1 - mask_smooth) * alpha
    
    # Apply soft masking
    result = image * soft_mask
    
    return result.astype(np.uint8)


def apply_attention_mask(image, mask, min_weight=0.2):
    """
    Apply attention-based masking with smooth gradient transitions
    
    Creates smooth attention weights that gradually decrease from
    foreground to background regions, mimicking visual attention mechanisms.
    
    Args:
        image: Input grayscale image (numpy array)
        mask: Binary mask (numpy array, same size as image)
        min_weight: Minimum attention weight for background regions (0.0-1.0)
                   Higher values preserve more background information
    
    Returns:
        Attention-masked image as uint8 numpy array
    """
    # Convert mask to float
    mask_float = mask.astype(np.float32)
    
    # Apply strong Gaussian blur to create smooth attention gradients
    mask_smooth = cv2.GaussianBlur(mask_float, (21, 21), 7)
    
    # Create attention map with smooth transitions
    # Formula: attention_weight = min_weight + (1 - min_weight) * smooth_mask
    attention_map = min_weight + (1 - min_weight) * mask_smooth
    
    # Apply attention-based weighting
    result = image * attention_map
    
    return result.astype(np.uint8)


def apply_hard_mask(image, mask):
    """
    Apply traditional hard masking (binary multiplication)
    
    Sets background pixels to zero, keeping only foreground regions.
    This is the traditional masking approach but may cause information loss.
    
    Args:
        image: Input grayscale image (numpy array)
        mask: Binary mask (numpy array, same size as image)
    
    Returns:
        Hard-masked image as uint8 numpy array
    """
    return (image * mask).astype(np.uint8)


def create_edge_preserving_mask(mask, edge_width=3):
    """
    Create edge-preserving mask that maintains boundary information
    
    Dilates the mask boundaries to preserve edge information that might
    be important for classification tasks.
    
    Args:
        mask: Binary mask (numpy array)
        edge_width: Width of edge preservation in pixels
    
    Returns:
        Edge-preserving mask as numpy array
    """
    # Create morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                      (edge_width*2+1, edge_width*2+1))
    
    # Dilate mask to include edge regions
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    
    return dilated_mask


def combine_masks(*masks, weights=None):
    """
    Combine multiple masks with optional weighting
    
    Args:
        masks: Variable number of mask arrays
        weights: Optional weights for each mask (must sum to 1.0)
                If None, equal weights are used
    
    Returns:
        Combined mask as numpy array
    """
    if not masks:
        raise ValueError("At least one mask must be provided")
    
    # Convert all masks to float
    masks_float = [mask.astype(np.float32) for mask in masks]
    
    # Set equal weights if not provided
    if weights is None:
        weights = [1.0 / len(masks)] * len(masks)
    
    # Validate weights
    if len(weights) != len(masks):
        raise ValueError("Number of weights must match number of masks")
    
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")
    
    # Combine masks
    combined = np.zeros_like(masks_float[0])
    for mask, weight in zip(masks_float, weights):
        combined += mask * weight
    
    return combined.astype(np.uint8)


def validate_mask(mask, image_shape):
    """
    Validate mask properties and compatibility with image
    
    Args:
        mask: Mask to validate
        image_shape: Shape of the target image
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if mask is None:
        return False, "Mask is None"
    
    if mask.shape != image_shape[:2]:  # Compare with H, W only
        return False, f"Mask shape {mask.shape} doesn't match image shape {image_shape[:2]}"
    
    if mask.dtype not in [np.uint8, np.float32, np.float64]:
        return False, f"Invalid mask dtype: {mask.dtype}"
    
    # Check if mask contains valid values
    unique_values = np.unique(mask)
    if mask.dtype == np.uint8:
        if not all(val in [0, 1] or (val >= 0 and val <= 255) for val in unique_values):
            return False, "Uint8 mask should contain values in range [0, 255] or binary [0, 1]"
    else:
        if not all(val >= 0.0 and val <= 1.0 for val in unique_values):
            return False, "Float mask should contain values in range [0.0, 1.0]"
    
    return True, "Valid mask"