# NLC (Noctilucent Cloud) Image Classification Pipeline

## NASA Citizen Science - Space Cloud Watch

A vision-only deep learning pipeline for classifying noctilucent clouds (NLCs) in citizen science sky imagery. Designed as an **assistive tool** with explicit human-in-the-loop support.

![NLC Example](https://upload.wikimedia.org/wikipedia/commons/d/d7/Noctilucent_clouds_over_Uppsala%2C_Sweden.jpg)

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Dataset Analysis](#dataset-analysis)
5. [Model Recommendation](#model-recommendation)
6. [Training Pipeline](#training-pipeline)
7. [Inference & Human Review](#inference--human-review)
8. [Confidence Thresholds](#confidence-thresholds)
9. [Failure Modes & Limitations](#failure-modes--limitations)
10. [API Reference](#api-reference)

---

## Overview

This pipeline provides:

- **Binary classification**: NLC Present vs. No NLC
- **Confidence scoring**: Probability estimates for each prediction
- **Human-in-the-loop**: Automatic flagging of uncertain predictions for human review
- **Out-of-distribution detection**: Entropy-based uncertainty estimation

### Key Design Principles

1. **Model assists, never replaces**: All predictions are assistive suggestions
2. **Uncertainty-aware**: Low-confidence predictions are flagged
3. **Transparent**: Full probability distributions and review reasoning provided
4. **Calibrated**: Temperature scaling for well-calibrated confidence scores

---

## Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision timm pandas pillow requests scikit-learn tqdm matplotlib numpy
```

---

## Quick Start

### 1. Analyze Your Dataset

```bash
python main.py analyze --csv space-cloud-watch_data.csv
```

### 2. Train the Model

```bash
python main.py train --csv space-cloud-watch_data.csv --epochs 30
```

### 3. Predict Single Image

```bash
python main.py predict --checkpoint checkpoints/best_model.pt --input image.jpg
```

### 4. Batch Predict with Review Queue

```bash
python main.py batch-predict --checkpoint checkpoints/best_model.pt --csv data.csv --output-dir predictions/
```

---

## Dataset Analysis

Based on the Space Cloud Watch CSV analysis:

| Metric | Value |
|--------|-------|
| Total Observations | ~450 |
| Observations with Images | ~100-150 |
| NLC Positive | ~30-40% |
| NLC Negative | ~60-70% |
| Geographic Range | Global (40°N - 65°N primarily) |

### Relevant CSV Fields

| Field | Purpose |
|-------|---------|
| `did you see nlc?` | Primary label (Yes/No) |
| `types of nlc` | Multi-label (Type 1-4) |
| `upload an image...` | Image URL columns |
| `latitude`, `longitude` | Location for analysis |
| `observedDate` | Temporal distribution |

### Data Preprocessing Strategy

1. **Label Extraction**: Parse "did you see nlc?" → Binary (0/1)
2. **Image Filtering**: Keep only samples with valid image URLs
3. **Stratified Split**: 80% train / 20% validation maintaining class ratio
4. **Class Balancing**: Weighted sampling for imbalanced classes
5. **Augmentation**: Strong augmentation for small datasets

---

## Model Recommendation

### For ~600 Samples: **EfficientNet-B0** ✅

| Criteria | EfficientNet-B0 | ViT-Small |
|----------|-----------------|-----------|
| Parameters | 5.3M | 22M |
| Data Efficiency | ★★★★★ | ★★☆☆☆ |
| Training Speed | ★★★★★ | ★★★☆☆ |
| Inference Speed | ★★★★★ | ★★★☆☆ |
| Small Dataset Performance | ★★★★★ | ★★☆☆☆ |
| Attention Maps | ❌ | ✅ |

### Reasoning

1. **Inductive Bias**: EfficientNet's convolutional architecture has locality bias that helps with limited data
2. **Parameter Efficiency**: 4x fewer parameters = less overfitting risk
3. **Proven Transfer**: ImageNet pretraining transfers well to natural images
4. **Practical**: Faster training and inference for iterative development

### When to Consider ViT

- Dataset grows to 5000+ samples
- Need attention visualization for interpretability
- Have access to domain-specific pretraining

---

## Training Pipeline

### Data Augmentation

Strong augmentation is applied for small datasets:

```python
transforms.Compose([
    transforms.Resize((272, 272)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Training Features

- **Differential Learning Rates**: Lower LR for backbone, higher for classifier
- **Cosine Annealing**: With warmup epochs
- **Early Stopping**: Patience-based on validation F1
- **Mixed Precision**: AMP for faster training (CUDA)
- **Checkpointing**: Save best model by F1 score

### Hyperparameters (Default)

```python
batch_size = 16
learning_rate = 1e-4
weight_decay = 1e-4
epochs = 30
warmup_epochs = 3
dropout = 0.3
```

---

## Inference & Human Review

### Prediction Workflow

```
Image Input → Model → Confidence Score → Decision Router
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
              High Confidence           Medium Confidence          Low Confidence
              (≥85%)                    (65-85%)                   (<65%)
                    │                         │                         │
              Auto-Accept              Flag for Review           Priority Review
              (with logging)           (Priority 3)              (Priority 1)
```

### Review Reasons

| Reason | Description | Priority |
|--------|-------------|----------|
| `low_confidence` | Confidence < 65% | 1 (High) |
| `high_entropy` | Prediction entropy > 0.9 | 2 (Medium) |
| `borderline_prediction` | Class probabilities within 20% | 2 (Medium) |
| `conflicting_signals` | Other uncertainty indicators | 3 (Low) |
| `automatic_accept` | High confidence, no flags | - |

---

## Confidence Thresholds

### Recommended Settings

| Threshold | Value | Meaning |
|-----------|-------|---------|
| High Confidence | ≥ 0.85 | Auto-accept (still logged) |
| Low Confidence | < 0.65 | Mandatory human review |
| Entropy | > 0.9 | Potential OOD detection |

### Adjusting Thresholds

```bash
# More conservative (more human review)
python main.py batch-predict --high-threshold 0.90 --low-threshold 0.70 ...

# More permissive (less human review)
python main.py batch-predict --high-threshold 0.80 --low-threshold 0.60 ...
```

### Expected Review Distribution

For a well-calibrated model on this dataset:
- ~40-50% auto-accepted
- ~30-40% medium-priority review
- ~10-20% high-priority review

---

## Failure Modes & Limitations

### ⚠️ Known Failure Modes

#### 1. **Confusion with Similar Cloud Types**

| Confusing Feature | Why It Fails | Mitigation |
|-------------------|--------------|------------|
| Cirrus clouds | Similar appearance, lower altitude | Model sees similar patterns |
| Aurora borealis | Can co-occur with NLCs | Check for color differences |
| Light pollution | Similar blue/white coloring | Geographic context helps |
| Sunlight on regular clouds | Golden hour confusion | Time-of-day metadata |

#### 2. **Image Quality Issues**

- **Overexposed images**: Washes out NLC detail
- **Underexposed images**: NLCs may be invisible
- **Motion blur**: Common in night photography
- **Compression artifacts**: JPEG artifacts mimic wave patterns

#### 3. **Geographic/Temporal Bias**

- Training data primarily from Northern latitudes (45°N - 65°N)
- Seasonal bias (NLCs mainly visible May-August in Northern Hemisphere)
- May underperform on Southern Hemisphere data

#### 4. **Camera Variation**

- Different white balance settings affect NLC color
- Smartphone vs DSLR quality differences
- RAW vs processed image differences

### Detection Strategies

```python
# High entropy indicates model uncertainty
if prediction.entropy > 0.9:
    print("⚠️ Model is uncertain - likely OOD or ambiguous")

# Low confidence indicates poor fit
if prediction.confidence < 0.65:
    print("⚠️ Low confidence - human review required")

# Borderline predictions
if abs(probs[0] - probs[1]) < 0.2:
    print("⚠️ Borderline case - could be either class")
```

### Recommendations for Deployment

1. **Never auto-publish**: All predictions should be logged for audit
2. **Expert spot-checks**: Randomly sample auto-accepted predictions
3. **Feedback loop**: Use corrections to improve model
4. **Geographic awareness**: Consider location when interpreting
5. **Temporal context**: NLCs are seasonal - unexpected detections warrant scrutiny

---

## API Reference

### Python API

```python
from nlc_classifier import (
    get_default_config,
    create_model,
    train_nlc_classifier,
    load_inference_engine
)

# Configuration
config = get_default_config()
config.model.model_name = "efficientnet_b0"

# Training
model, results = train_nlc_classifier("data.csv", config)

# Inference
engine = load_inference_engine("checkpoints/best_model.pt")
result = engine.predict("image.jpg")

print(f"Prediction: {result.predicted_label}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Needs Review: {result.needs_review}")
```

### CLI Commands

```bash
# Dataset analysis
python main.py analyze --csv data.csv --output analysis.json

# Training
python main.py train --csv data.csv --model efficientnet_b0 --epochs 30

# Single prediction
python main.py predict --checkpoint model.pt --input image.jpg

# Batch prediction
python main.py batch-predict --checkpoint model.pt --csv data.csv --output-dir results/
```

---

## File Structure

```
CitSci/
├── main.py                    # CLI entry point
├── nlc_classifier/
│   ├── __init__.py
│   ├── config.py              # Configuration dataclasses
│   ├── data.py                # Data loading & augmentation
│   ├── models.py              # Model architectures
│   ├── train.py               # Training pipeline
│   └── inference.py           # Inference with human-in-loop
├── checkpoints/               # Saved models
├── images_cache/              # Downloaded images
├── logs/                      # Training logs
├── predictions/               # Inference outputs
└── README.md
```

---

## Citation

If you use this pipeline for research, please cite:

```bibtex
@software{nlc_classifier_2026,
  title = {NLC Image Classification Pipeline for Citizen Science},
  author = {Space Cloud Watch Team},
  year = {2026},
  url = {https://citsci.org/projects/space-cloud-watch/}
}
```

---

## Acknowledgments

- NASA Citizen Science Program
- CitSci.org Platform
- Space Cloud Watch Contributors

---

## ⚠️ Disclaimer

**This model is an ASSISTIVE TOOL and must NOT replace human classification for scientific purposes.**

All auto-accepted predictions are logged and subject to audit. For publications or critical analyses, expert review of all classifications is strongly recommended.

---

## License

CC-BY 3.0 - See [CitSci Terms](https://citsci.org/terms)
