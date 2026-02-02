"""
Configuration settings for NLC (Noctilucent Cloud) Image Classifier.
NASA Citizen Science Project - Space Cloud Watch
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration."""
    csv_path: str = ""  # Path to the Space Cloud Watch CSV
    image_cache_dir: str = "images_cache"  # Local cache for downloaded images
    train_split: float = 0.8  # 80% training, 20% validation
    random_seed: int = 42
    
    # Relevant CSV columns
    label_column: str = "did you see nlc?"  # Primary label: Yes/No
    nlc_types_column: str = "types of nlc"  # Multi-label: Type 1-4
    image_columns: List[str] = field(default_factory=lambda: [
        "please upload an image of nlc (if available, optional)",
        "please upload an image of nlc (if available, optional, please do not upload images when you do not see nlc.)",
        "upload an image of nlc (please do not upload images when you do not see nlc. non nlc images will be archived)"
    ])
    
    # Label mappings
    binary_labels: dict = field(default_factory=lambda: {
        "Yes": 1,
        "No": 0
    })
    
    nlc_type_labels: dict = field(default_factory=lambda: {
        "Type 1 (Veil)": 0,
        "Type 2 (Bands)": 1,
        "Type 3 (Waves or Billows)": 2,
        "Type 4 (Whirls)": 3
    })


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Model selection: "efficientnet_b0" or "vit_small"
    model_name: str = "efficientnet_b0"
    
    # For ~600 images, EfficientNet-B0 is recommended
    # ViT requires more data but can be used with strong augmentation
    pretrained: bool = True
    num_classes: int = 2  # Binary: NLC present/absent
    num_nlc_types: int = 4  # Multi-label: Type 1, 2, 3, 4
    dropout_rate: float = 0.3
    
    # Multi-task learning
    use_type_classification: bool = True  # Also predict NLC types
    type_loss_weight: float = 0.5  # Weight for type classification loss
    
    # Image preprocessing
    image_size: int = 224  # Standard for both models
    normalize_mean: tuple = (0.485, 0.456, 0.406)  # ImageNet stats
    normalize_std: tuple = (0.229, 0.224, 0.225)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 16  # Small batch for small dataset
    num_epochs: int = 30
    learning_rate: float = 1e-4  # Lower LR for fine-tuning
    weight_decay: float = 1e-4
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # "cosine" or "step"
    warmup_epochs: int = 3
    
    # Early stopping
    patience: int = 7
    min_delta: float = 0.001
    
    # Data augmentation strength
    augmentation_strength: str = "strong"  # "light", "medium", "strong"
    
    # Mixed precision training
    use_amp: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True


@dataclass
class InferenceConfig:
    """Inference and human-in-the-loop configuration."""
    # Confidence thresholds for human review routing
    high_confidence_threshold: float = 0.85  # Auto-accept predictions above this
    low_confidence_threshold: float = 0.65   # Flag for human review below this
    
    # Out-of-distribution detection
    ood_entropy_threshold: float = 0.9  # High entropy = uncertain
    
    # Human review flags
    flag_low_confidence: bool = True
    flag_high_entropy: bool = True
    flag_novel_patterns: bool = True
    
    # Output settings
    output_dir: str = "predictions"
    save_confidence_scores: bool = True
    generate_review_queue: bool = True


@dataclass
class NLCConfig:
    """Master configuration combining all settings."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Device settings
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    num_workers: int = 0  # Use 0 for macOS to avoid multiprocessing issues
    
    # Logging
    log_dir: str = "logs"
    verbose: bool = True
    
    def get_device(self) -> str:
        """Determine the best available device."""
        import torch
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def setup_directories(self):
        """Create necessary directories."""
        dirs = [
            self.data.image_cache_dir,
            self.training.checkpoint_dir,
            self.inference.output_dir,
            self.log_dir
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


def get_default_config() -> NLCConfig:
    """Get default configuration."""
    return NLCConfig()


def create_config_for_dataset_size(num_samples: int) -> NLCConfig:
    """
    Create optimized config based on dataset size.
    
    For ~600 images (typical citizen science dataset):
    - Use EfficientNet-B0 (better for small datasets)
    - Strong augmentation
    - Conservative training settings
    """
    config = NLCConfig()
    
    if num_samples < 500:
        # Very small dataset - maximum regularization
        config.model.model_name = "efficientnet_b0"
        config.model.dropout_rate = 0.5
        config.training.augmentation_strength = "strong"
        config.training.batch_size = 8
        config.training.num_epochs = 50
    elif num_samples < 1000:
        # Small dataset (typical for this project)
        config.model.model_name = "efficientnet_b0"
        config.model.dropout_rate = 0.3
        config.training.augmentation_strength = "strong"
        config.training.batch_size = 16
    elif num_samples < 5000:
        # Medium dataset - ViT becomes viable
        config.model.model_name = "efficientnet_b0"  # Still prefer EfficientNet
        config.model.dropout_rate = 0.2
        config.training.augmentation_strength = "medium"
        config.training.batch_size = 32
    else:
        # Large dataset - ViT preferred
        config.model.model_name = "vit_small"
        config.model.dropout_rate = 0.1
        config.training.augmentation_strength = "medium"
        config.training.batch_size = 64
    
    return config
