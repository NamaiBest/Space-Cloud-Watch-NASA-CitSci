"""
NLC Classifier Package
NASA Citizen Science - Space Cloud Watch
"""

from .config import NLCConfig, get_default_config, create_config_for_dataset_size
from .models import NLCClassifier, EfficientNetNLC, ViTNLC, create_model
from .data import create_dataloaders, analyze_dataset
from .train import train_nlc_classifier, Trainer
from .inference import NLCInferenceEngine, load_inference_engine, PredictionResult

__version__ = "1.0.0"
__all__ = [
    "NLCConfig",
    "get_default_config",
    "create_config_for_dataset_size",
    "NLCClassifier",
    "EfficientNetNLC", 
    "ViTNLC",
    "create_model",
    "create_dataloaders",
    "analyze_dataset",
    "train_nlc_classifier",
    "Trainer",
    "NLCInferenceEngine",
    "load_inference_engine",
    "PredictionResult"
]
