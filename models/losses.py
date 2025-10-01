"""
Custom Loss Functions for Segmentation Model
============================================

Contains Keras custom loss functions and metrics for ResUNet segmentation model.
"""

from tensorflow.keras import backend as K


def dice_coefficient(y_true, y_pred, smooth=1e-7):
    """
    Dice coefficient metric for segmentation evaluation
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient score
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """
    Dice loss function (1 - dice_coefficient)
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        
    Returns:
        Dice loss
    """
    return 1 - dice_coefficient(y_true, y_pred)


def iou_coefficient(y_true, y_pred, smooth=1e-7):
    """
    IoU/Jaccard coefficient metric
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU coefficient score
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def iou_loss(y_true, y_pred):
    """
    IoU/Jaccard loss function (1 - iou_coefficient)
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        
    Returns:
        IoU loss
    """
    return 1 - iou_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """
    Combined Dice and IoU loss function
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        
    Returns:
        Combined loss (0.5 * dice_loss + 0.5 * iou_loss)
    """
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)


def precision(y_true, y_pred):
    """
    Precision metric for binary segmentation
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        
    Returns:
        Precision score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def recall(y_true, y_pred):
    """
    Recall/Sensitivity metric for binary segmentation
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        
    Returns:
        Recall score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (actual_positives + K.epsilon())


def specificity(y_true, y_pred):
    """
    Specificity metric for binary segmentation
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        
    Returns:
        Specificity score
    """
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    actual_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (actual_negatives + K.epsilon())