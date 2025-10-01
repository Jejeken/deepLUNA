"""
Classification Models for CXR Analysis
======================================

Contains the OptimizedMultiInputClassifier with EfficientNet-B2 backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class OptimizedMultiInputClassifier(nn.Module):
    """
    Optimized multi-input classifier with EfficientNet-B2 backbone
    
    Supports both original and segmented image inputs with type-aware processing.
    Features dual attention mechanism and ensemble-friendly architecture.
    """
    
    def __init__(self, backbone_name="efficientnet_b2", num_classes=2, dropout=0.3):
        """
        Initialize the classifier
        
        Args:
            backbone_name: Name of the EfficientNet backbone
            num_classes: Number of output classes (default: 2 for binary)
            dropout: Dropout rate for regularization
        """
        super(OptimizedMultiInputClassifier, self).__init__()
        
        # EfficientNet-B2 backbone
        self.backbone = timm.create_model(backbone_name, pretrained=True, in_chans=1, num_classes=0)
        self.num_features = self.backbone.num_features
        
        # Enhanced type embedding for input type awareness
        self.type_embedding = nn.Sequential(
            nn.Embedding(2, 64),  # 2 types: original (0), segmented (1)
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
        
        # Multi-scale global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features * 2 + self.num_features, 256),  # avg + max + type embedding
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, input_type=None):
        """
        Forward pass with optional input type information
        
        Args:
            x: Input tensor (B, C, H, W)
            input_type: Type of input (0 for original, 1 for segmented)
            
        Returns:
            Logits for classification
        """
        # Extract features using backbone
        features = self.backbone.forward_features(x)
        
        # Apply dual attention mechanism
        spatial_att = self.spatial_attention(features)
        channel_att = self.channel_attention(features)
        channel_att = channel_att.unsqueeze(-1).unsqueeze(-1)
        
        # Combined attention application
        attended_features = features * spatial_att * channel_att
        
        # Multi-scale pooling
        avg_pool = self.global_pool(attended_features).flatten(1)
        max_pool = self.global_max_pool(attended_features).flatten(1)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        
        # Add type embedding if provided
        if input_type is not None:
            type_emb = self.type_embedding[0](input_type)  # Get embedding
            type_emb = self.type_embedding[1:](type_emb)   # Process through layers
            combined = torch.cat([combined, type_emb], dim=1)
        else:
            # Add zero embedding if no type specified
            batch_size = x.size(0)
            zero_emb = torch.zeros(batch_size, self.num_features).to(x.device)
            combined = torch.cat([combined, zero_emb], dim=1)
        
        # Final classification
        logits = self.classifier(combined)
        return logits
    
    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization
        
        Args:
            x: Input tensor
            
        Returns:
            Feature maps from backbone
        """
        return self.backbone.forward_features(x)