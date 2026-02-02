"""
Model architectures for NLC (Noctilucent Cloud) classification.
Supports EfficientNet-B0 and Vision Transformer (ViT-Small).

Model Recommendation for ~600 images:
=======================================
EfficientNet-B0 is STRONGLY RECOMMENDED because:
1. More parameter-efficient (5.3M params vs 22M for ViT-Small)
2. Better inductive biases for image classification
3. Requires less data to achieve good generalization
4. Faster training and inference
5. Proven performance on small datasets

ViT-Small should only be used if:
- Dataset grows to 5000+ samples
- Strong pre-training on similar domain is available
- Computational resources are not a concern
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Dict, Tuple, Any

from .config import ModelConfig


class NLCClassifier(nn.Module):
    """
    Base NLC classifier with common interface.
    Wraps pretrained backbone with custom classification head.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.backbone = None
        self.classifier = None
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning logits and features.
        
        Returns:
            dict with:
                - logits: Classification logits [B, num_classes]
                - probabilities: Softmax probabilities [B, num_classes]
                - features: Penultimate layer features for analysis
        """
        raise NotImplementedError
    
    def get_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prediction and confidence score.
        
        Returns:
            predictions: Predicted class indices [B]
            confidence: Confidence scores [B]
        """
        output = self.forward(x)
        probs = output['probabilities']
        confidence, predictions = torch.max(probs, dim=1)
        return predictions, confidence
    
    def compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction entropy for uncertainty estimation.
        High entropy = high uncertainty.
        
        Returns:
            entropy: Entropy values [B]
        """
        output = self.forward(x)
        probs = output['probabilities']
        # Add small epsilon to avoid log(0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        # Normalize by max entropy (log(num_classes))
        max_entropy = torch.log(torch.tensor(self.config.num_classes, dtype=torch.float))
        return entropy / max_entropy


class EfficientNetNLC(NLCClassifier):
    """
    EfficientNet-B0 based NLC classifier.
    
    Recommended for this project due to:
    - Efficient architecture (5.3M parameters)
    - Strong ImageNet pretraining transfers well
    - Good performance on small datasets
    - Fast inference suitable for real-time classification
    
    Supports multi-task learning:
    - Binary classification: NLC present/absent
    - Multi-label classification: NLC types (Type 1-4)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Load pretrained EfficientNet-B0
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=config.pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Shared feature processing
        self.shared_fc = nn.Sequential(
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout_rate / 2),
        )
        
        # Binary classification head (NLC present/absent)
        self.binary_classifier = nn.Linear(256, config.num_classes)
        
        # Multi-label type classification head (Type 1-4)
        self.use_type_classification = getattr(config, 'use_type_classification', True)
        self.num_nlc_types = getattr(config, 'num_nlc_types', 4)
        if self.use_type_classification:
            self.type_classifier = nn.Linear(256, self.num_nlc_types)
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier weights with proper scaling."""
        for m in [self.shared_fc, self.binary_classifier]:
            if isinstance(m, nn.Sequential):
                for layer in m.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        if self.use_type_classification:
            nn.init.kaiming_normal_(self.type_classifier.weight, mode='fan_out', nonlinearity='relu')
            if self.type_classifier.bias is not None:
                nn.init.constant_(self.type_classifier.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features from backbone
        features = self.backbone(x)
        
        # Shared feature processing
        shared_features = self.shared_fc(features)
        
        # Binary classification
        logits = self.binary_classifier(shared_features)
        probabilities = F.softmax(logits, dim=1)
        
        output = {
            'logits': logits,
            'probabilities': probabilities,
            'features': features
        }
        
        # Multi-label type classification (if enabled)
        if self.use_type_classification:
            type_logits = self.type_classifier(shared_features)
            type_probabilities = torch.sigmoid(type_logits)  # Multi-label uses sigmoid
            output['type_logits'] = type_logits
            output['type_probabilities'] = type_probabilities
        
        return output
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze


class ViTNLC(NLCClassifier):
    """
    Vision Transformer (ViT-Small) based NLC classifier.
    
    Use this when:
    - Dataset has 5000+ samples
    - You need attention maps for interpretability
    - Computational resources are available
    
    NOT recommended for ~600 samples due to:
    - Requires more data to generalize
    - Higher computational cost
    - Prone to overfitting on small datasets
    
    Supports multi-task learning:
    - Binary classification: NLC present/absent
    - Multi-label classification: NLC types (Type 1-4)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Load pretrained ViT-Small
        self.backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=config.pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Shared feature processing
        self.shared_fc = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(p=config.dropout_rate / 2),
        )
        
        # Binary classification head
        self.binary_classifier = nn.Linear(256, config.num_classes)
        
        # Multi-label type classification head
        self.use_type_classification = getattr(config, 'use_type_classification', True)
        self.num_nlc_types = getattr(config, 'num_nlc_types', 4)
        if self.use_type_classification:
            self.type_classifier = nn.Linear(256, self.num_nlc_types)
        
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier weights."""
        for m in [self.shared_fc, self.binary_classifier]:
            if isinstance(m, nn.Sequential):
                for layer in m.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.trunc_normal_(layer.weight, std=0.02)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        if self.use_type_classification:
            nn.init.trunc_normal_(self.type_classifier.weight, std=0.02)
            if self.type_classifier.bias is not None:
                nn.init.constant_(self.type_classifier.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features from backbone
        features = self.backbone(x)
        
        # Shared feature processing
        shared_features = self.shared_fc(features)
        
        # Binary classification
        logits = self.binary_classifier(shared_features)
        probabilities = F.softmax(logits, dim=1)
        
        output = {
            'logits': logits,
            'probabilities': probabilities,
            'features': features
        }
        
        # Multi-label type classification
        if self.use_type_classification:
            type_logits = self.type_classifier(shared_features)
            type_probabilities = torch.sigmoid(type_logits)
            output['type_logits'] = type_logits
            output['type_probabilities'] = type_probabilities
        
        return output
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention maps for interpretability.
        Useful for understanding what regions the model focuses on.
        """
        # This requires accessing intermediate attention weights
        # Implementation depends on specific ViT variant
        pass
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze


def create_model(config: ModelConfig, device: str = 'cpu') -> NLCClassifier:
    """
    Factory function to create the appropriate model.
    
    Args:
        config: Model configuration
        device: Device to place model on
    
    Returns:
        Initialized model
    """
    if config.model_name == 'efficientnet_b0':
        model = EfficientNetNLC(config)
    elif config.model_name == 'vit_small':
        model = ViTNLC(config)
    else:
        raise ValueError(f"Unknown model: {config.model_name}")
    
    return model.to(device)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def model_summary(model: NLCClassifier) -> str:
    """Generate a summary of the model."""
    params = count_parameters(model)
    
    summary = f"""
Model Summary
=============
Architecture: {model.config.model_name}
Number of classes: {model.config.num_classes}
Pretrained: {model.config.pretrained}
Dropout rate: {model.config.dropout_rate}

Parameters:
  Total: {params['total']:,}
  Trainable: {params['trainable']:,}
  Frozen: {params['frozen']:,}

Input size: {model.config.image_size}x{model.config.image_size}
"""
    return summary


class ModelWithTemperature(nn.Module):
    """
    Wrapper that adds temperature scaling for calibrated confidence scores.
    Temperature scaling helps produce well-calibrated probabilities.
    """
    
    def __init__(self, model: NLCClassifier, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = self.model(x)
        # Ensure temperature is on the same device as logits
        temp = self.temperature.to(output['logits'].device)
        logits = output['logits'] / temp
        probabilities = F.softmax(logits, dim=1)
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
            'features': output['features'],
            'temperature': self.temperature.item()
        }
        
        # Pass through type classification outputs if present
        if 'type_logits' in output:
            result['type_logits'] = output['type_logits']
        if 'type_probabilities' in output:
            result['type_probabilities'] = output['type_probabilities']
            
        return result
    
    def calibrate(self, val_loader, device: str = 'cpu'):
        """
        Learn optimal temperature on validation set.
        Uses Expected Calibration Error (ECE) minimization.
        """
        self.model.eval()
        
        nll_criterion = nn.CrossEntropyLoss()
        
        # Collect all logits and labels
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                output = self.model(images)
                logits_list.append(output['logits'].cpu())
                labels_list.append(labels)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # Optimize temperature
        self.temperature.requires_grad = True
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        print(f"Optimal temperature: {self.temperature.item():.3f}")
        return self.temperature.item()


if __name__ == "__main__":
    # Quick test
    from .config import get_default_config
    
    config = get_default_config()
    
    print("Testing EfficientNet-B0:")
    model = create_model(config.model, device='cpu')
    print(model_summary(model))
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Output probabilities: {output['probabilities']}")
    
    # Test confidence
    preds, conf = model.get_confidence(x)
    print(f"Predictions: {preds}, Confidence: {conf}")
    
    # Test entropy
    entropy = model.compute_entropy(x)
    print(f"Entropy: {entropy}")
