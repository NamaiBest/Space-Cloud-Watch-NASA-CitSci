"""
Inference module for NLC classification with human-in-the-loop support.
Provides confidence scores, uncertainty estimation, and flagging for review.

Key Features:
- Confidence-based prediction routing
- Entropy-based uncertainty detection
- Out-of-distribution (OOD) flagging
- Human review queue generation
- Batch and single-image inference
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm

from .config import NLCConfig, InferenceConfig
from .models import NLCClassifier, create_model, ModelWithTemperature
from .data import get_validation_transforms, ImageCache


class ReviewReason(Enum):
    """Reasons for flagging an image for human review."""
    LOW_CONFIDENCE = "low_confidence"
    HIGH_ENTROPY = "high_entropy"
    BORDERLINE_PREDICTION = "borderline_prediction"
    POTENTIAL_OOD = "potential_out_of_distribution"
    CONFLICTING_SIGNALS = "conflicting_signals"
    AUTOMATIC_ACCEPT = "automatic_accept"


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    image_id: str
    image_path: str
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    entropy: float
    needs_review: bool
    review_reason: str
    review_priority: int  # 1 = highest priority, 3 = lowest
    # NLC Type predictions (multi-label)
    nlc_types: List[str]  # List of detected NLC types
    nlc_type_probabilities: Dict[str, float]  # Per-type probabilities
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchPredictionResults:
    """Results from batch prediction."""
    predictions: List[PredictionResult]
    summary: Dict[str, Any]
    review_queue: List[PredictionResult]
    auto_accepted: List[PredictionResult]
    timestamp: str


class NLCInferenceEngine:
    """
    Inference engine with human-in-the-loop support.
    
    Workflow:
    1. Image input (URL, path, or tensor)
    2. Model prediction with confidence
    3. Uncertainty estimation (entropy, temperature-scaled)
    4. Decision routing:
       - High confidence → Auto-accept (with logging)
       - Low confidence → Flag for human review
       - Borderline → Add to review queue with priority
    5. Generate review queue for human annotators
    """
    
    def __init__(
        self,
        model: NLCClassifier,
        config: NLCConfig,
        device: str = 'cpu',
        use_temperature_scaling: bool = True
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.transform = get_validation_transforms(config)
        self.image_cache = ImageCache(config.data.image_cache_dir)
        
        # Class labels
        self.class_labels = {0: "No NLC", 1: "NLC Present"}
        
        # Temperature scaling for better calibration
        if use_temperature_scaling:
            self.model = ModelWithTemperature(model)
        
        self.model.eval()
    
    def load_image(self, source: Union[str, Path, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Load and preprocess image from various sources.
        
        Args:
            source: URL, file path, PIL Image, or tensor
        
        Returns:
            Preprocessed tensor ready for inference
        """
        if isinstance(source, torch.Tensor):
            return source.unsqueeze(0) if source.dim() == 3 else source
        
        if isinstance(source, Image.Image):
            image = source.convert('RGB')
        elif isinstance(source, (str, Path)):
            source = str(source)
            if source.startswith('http'):
                # Download from URL
                path = self.image_cache.download_and_cache(source)
                if path is None:
                    raise ValueError(f"Failed to download image from {source}")
                image = Image.open(path).convert('RGB')
            else:
                image = Image.open(source).convert('RGB')
        else:
            raise ValueError(f"Unsupported image source type: {type(source)}")
        
        return self.transform(image).unsqueeze(0)
    
    @torch.no_grad()
    def predict(
        self,
        image_source: Union[str, Path, Image.Image, torch.Tensor],
        image_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PredictionResult:
        """
        Make a single prediction with confidence and uncertainty estimation.
        
        Args:
            image_source: Image input (URL, path, PIL Image, or tensor)
            image_id: Unique identifier for the image
            metadata: Additional metadata to include in result
        
        Returns:
            PredictionResult with prediction, confidence, and review flag
        """
        # Load and preprocess image
        image_tensor = self.load_image(image_source).to(self.device)
        
        # Get model output
        output = self.model(image_tensor)
        probs = output['probabilities'][0].cpu().numpy()
        
        # Get prediction and confidence
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        
        # Calculate entropy for uncertainty
        entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy
        
        # Get NLC type predictions (if model supports it and NLC is detected)
        nlc_types = []
        nlc_type_probs = {}
        type_names = ['Type 1 (Veil)', 'Type 2 (Bands)', 'Type 3 (Waves)', 'Type 4 (Whirls)']
        
        if 'type_probabilities' in output and predicted_class == 1:
            type_probs = output['type_probabilities'][0].cpu().numpy()
            for i, name in enumerate(type_names):
                prob = float(type_probs[i])
                nlc_type_probs[name] = prob
                if prob > 0.5:  # Threshold for multi-label
                    nlc_types.append(name)
        else:
            # Model doesn't support types or no NLC detected
            for name in type_names:
                nlc_type_probs[name] = 0.0
        
        # Determine if review is needed
        needs_review, review_reason, priority = self._determine_review_status(
            confidence, normalized_entropy, probs
        )
        
        # Create probabilities dict
        prob_dict = {self.class_labels[i]: float(p) for i, p in enumerate(probs)}
        
        return PredictionResult(
            image_id=image_id or str(hash(str(image_source))),
            image_path=str(image_source) if isinstance(image_source, (str, Path)) else "tensor_input",
            predicted_class=predicted_class,
            predicted_label=self.class_labels[predicted_class],
            confidence=confidence,
            probabilities=prob_dict,
            entropy=normalized_entropy,
            needs_review=needs_review,
            review_reason=review_reason.value,
            review_priority=priority,
            nlc_types=nlc_types,
            nlc_type_probabilities=nlc_type_probs,
            metadata=metadata or {}
        )
    
    def _determine_review_status(
        self,
        confidence: float,
        entropy: float,
        probs: np.ndarray
    ) -> Tuple[bool, ReviewReason, int]:
        """
        Determine if prediction needs human review.
        
        Returns:
            (needs_review, reason, priority)
        """
        inf_config = self.config.inference
        
        # Check confidence thresholds
        if confidence >= inf_config.high_confidence_threshold:
            # High confidence - auto accept
            return False, ReviewReason.AUTOMATIC_ACCEPT, 0
        
        if confidence < inf_config.low_confidence_threshold:
            # Low confidence - definitely needs review
            return True, ReviewReason.LOW_CONFIDENCE, 1
        
        # Check entropy
        if entropy > inf_config.ood_entropy_threshold:
            # High entropy - might be OOD
            return True, ReviewReason.HIGH_ENTROPY, 2
        
        # Check for borderline cases
        prob_diff = abs(probs[0] - probs[1])
        if prob_diff < 0.2:
            return True, ReviewReason.BORDERLINE_PREDICTION, 2
        
        # Medium confidence - flag for review but lower priority
        return True, ReviewReason.CONFLICTING_SIGNALS, 3
    
    @torch.no_grad()
    def predict_batch(
        self,
        image_sources: List[Union[str, Path]],
        image_ids: Optional[List[str]] = None,
        batch_size: int = 16,
        show_progress: bool = True
    ) -> BatchPredictionResults:
        """
        Make predictions on a batch of images.
        
        Args:
            image_sources: List of image URLs or paths
            image_ids: Optional list of image identifiers
            batch_size: Batch size for inference
            show_progress: Whether to show progress bar
        
        Returns:
            BatchPredictionResults with all predictions and review queue
        """
        predictions = []
        
        if image_ids is None:
            image_ids = [str(i) for i in range(len(image_sources))]
        
        iterator = tqdm(
            zip(image_sources, image_ids),
            total=len(image_sources),
            desc="Processing images",
            disable=not show_progress
        )
        
        for source, img_id in iterator:
            try:
                result = self.predict(source, image_id=img_id)
                predictions.append(result)
            except Exception as e:
                print(f"Error processing {source}: {e}")
                # Create error result
                predictions.append(PredictionResult(
                    image_id=img_id,
                    image_path=str(source),
                    predicted_class=-1,
                    predicted_label="ERROR",
                    confidence=0.0,
                    probabilities={},
                    entropy=1.0,
                    needs_review=True,
                    review_reason="processing_error",
                    review_priority=1,
                    metadata={"error": str(e)}
                ))
        
        # Separate into review queue and auto-accepted
        review_queue = [p for p in predictions if p.needs_review]
        auto_accepted = [p for p in predictions if not p.needs_review]
        
        # Sort review queue by priority
        review_queue.sort(key=lambda x: (x.review_priority, -x.entropy))
        
        # Generate summary
        summary = self._generate_summary(predictions)
        
        return BatchPredictionResults(
            predictions=predictions,
            summary=summary,
            review_queue=review_queue,
            auto_accepted=auto_accepted,
            timestamp=datetime.now().isoformat()
        )
    
    def _generate_summary(self, predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Generate summary statistics from predictions."""
        total = len(predictions)
        if total == 0:
            return {}
        
        needs_review = sum(1 for p in predictions if p.needs_review)
        errors = sum(1 for p in predictions if p.predicted_class == -1)
        
        valid_preds = [p for p in predictions if p.predicted_class >= 0]
        
        if valid_preds:
            nlc_positive = sum(1 for p in valid_preds if p.predicted_class == 1)
            avg_confidence = np.mean([p.confidence for p in valid_preds])
            avg_entropy = np.mean([p.entropy for p in valid_preds])
        else:
            nlc_positive = 0
            avg_confidence = 0.0
            avg_entropy = 0.0
        
        review_reasons = {}
        for p in predictions:
            reason = p.review_reason
            review_reasons[reason] = review_reasons.get(reason, 0) + 1
        
        return {
            'total_images': total,
            'auto_accepted': total - needs_review,
            'needs_review': needs_review,
            'errors': errors,
            'nlc_positive_predictions': nlc_positive,
            'nlc_negative_predictions': len(valid_preds) - nlc_positive,
            'average_confidence': float(avg_confidence),
            'average_entropy': float(avg_entropy),
            'review_reasons_distribution': review_reasons,
            'human_review_percentage': (needs_review / total) * 100 if total > 0 else 0
        }
    
    def generate_review_report(
        self,
        results: BatchPredictionResults,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a human-readable review report.
        
        Args:
            results: Batch prediction results
            output_path: Optional path to save report
        
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 70)
        report.append("NLC CLASSIFICATION REVIEW REPORT")
        report.append(f"Generated: {results.timestamp}")
        report.append("=" * 70)
        report.append("")
        
        # Summary
        s = results.summary
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total images processed: {s['total_images']}")
        report.append(f"Auto-accepted (high confidence): {s['auto_accepted']}")
        report.append(f"Flagged for human review: {s['needs_review']} ({s['human_review_percentage']:.1f}%)")
        report.append(f"Processing errors: {s['errors']}")
        report.append("")
        report.append(f"NLC Positive predictions: {s['nlc_positive_predictions']}")
        report.append(f"NLC Negative predictions: {s['nlc_negative_predictions']}")
        report.append(f"Average confidence: {s['average_confidence']:.3f}")
        report.append(f"Average entropy: {s['average_entropy']:.3f}")
        report.append("")
        
        # Review queue
        report.append("ITEMS REQUIRING HUMAN REVIEW")
        report.append("-" * 40)
        
        priority_labels = {1: "HIGH", 2: "MEDIUM", 3: "LOW"}
        
        for priority in [1, 2, 3]:
            items = [r for r in results.review_queue if r.review_priority == priority]
            if items:
                report.append(f"\n[{priority_labels[priority]} PRIORITY] - {len(items)} items")
                for item in items[:10]:  # Limit display
                    report.append(f"  • {item.image_id}")
                    report.append(f"    Prediction: {item.predicted_label} ({item.confidence:.1%})")
                    report.append(f"    Reason: {item.review_reason}")
                    report.append(f"    Entropy: {item.entropy:.3f}")
                if len(items) > 10:
                    report.append(f"  ... and {len(items) - 10} more")
        
        report.append("")
        report.append("=" * 70)
        report.append("IMPORTANT: This model is an ASSISTIVE TOOL.")
        report.append("Human verification is required for all flagged items.")
        report.append("Do not rely solely on model predictions for scientific conclusions.")
        report.append("=" * 70)
        
        report_str = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_str)
            print(f"Report saved to {output_path}")
        
        return report_str
    
    def save_results(
        self,
        results: BatchPredictionResults,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Save prediction results to files.
        
        Args:
            results: Batch prediction results
            output_dir: Directory to save outputs
        
        Returns:
            Dict with paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all predictions as JSON
        all_preds_path = output_dir / f"predictions_{timestamp}.json"
        with open(all_preds_path, 'w') as f:
            json.dump({
                'predictions': [p.to_dict() for p in results.predictions],
                'summary': results.summary,
                'timestamp': results.timestamp
            }, f, indent=2)
        
        # Save review queue as separate file
        review_path = output_dir / f"review_queue_{timestamp}.json"
        with open(review_path, 'w') as f:
            json.dump({
                'review_queue': [p.to_dict() for p in results.review_queue],
                'count': len(results.review_queue)
            }, f, indent=2)
        
        # Save human-readable report
        report_path = output_dir / f"review_report_{timestamp}.txt"
        self.generate_review_report(results, str(report_path))
        
        # Save auto-accepted for logging/audit
        accepted_path = output_dir / f"auto_accepted_{timestamp}.json"
        with open(accepted_path, 'w') as f:
            json.dump({
                'auto_accepted': [p.to_dict() for p in results.auto_accepted],
                'count': len(results.auto_accepted)
            }, f, indent=2)
        
        return {
            'all_predictions': str(all_preds_path),
            'review_queue': str(review_path),
            'report': str(report_path),
            'auto_accepted': str(accepted_path)
        }


def load_inference_engine(
    checkpoint_path: str,
    config: Optional[NLCConfig] = None,
    device: Optional[str] = None
) -> NLCInferenceEngine:
    """
    Load a trained model for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration (loaded from checkpoint if None)
        device: Device to use (auto-detected if None)
    
    Returns:
        Initialized inference engine
    """
    from .config import get_default_config, ModelConfig
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load config from checkpoint or use provided
    if config is None:
        config = get_default_config()
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            if 'model' in saved_config:
                for k, v in saved_config['model'].items():
                    if hasattr(config.model, k):
                        setattr(config.model, k, v)
    
    # Determine device
    if device is None:
        device = config.get_device()
    
    # Create and load model
    model = create_model(config.model, 'cpu')  # Load on CPU first
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)  # Then move to target device
    
    return NLCInferenceEngine(model, config, device)


if __name__ == "__main__":
    # Example usage
    from .config import get_default_config
    from .models import create_model
    
    config = get_default_config()
    model = create_model(config.model, 'cpu')
    
    engine = NLCInferenceEngine(model, config, 'cpu')
    
    # Example single prediction
    # result = engine.predict("https://example.com/nlc_image.jpg")
    # print(result)
