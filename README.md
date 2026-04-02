# NLC (Noctilucent Cloud) Image Classification Pipeline

## NASA Citizen Science — Space Cloud Watch

A deep learning pipeline for classifying noctilucent clouds (NLCs) in citizen science sky imagery, with built-in human-in-the-loop support.

---

## Key Features

- **Binary Classification** — Detects NLC presence vs. absence with calibrated confidence scores
- **Multi-Label Type Classification** — Identifies NLC types (Veil, Bands, Waves, Whirls) when present
- **Human-in-the-Loop** — Automatic flagging of uncertain predictions for expert review
- **Multi-Source Training** — Combines three independent datasets for robust generalisation
- **Web Portal** — Local dashboard for training, prediction, and dataset exploration

---

## Model Performance

### Current Model (Multi-Source, v2)

Trained on **890 images** from three sources with non-NLC cloud types as strong negatives.

| Metric | Value |
|---|---|
| Validation Accuracy | **91.0%** |
| F1 Score | **0.907** |
| Precision | 0.940 |
| Recall | 0.876 |
| Best Epoch | 28 / 30 |
| Training Time | 17 min (Apple Silicon MPS) |

### Previous Model (Single-Source, v1)

Trained only on Space Cloud Watch CSV (~219 images), no hard negatives.

| Metric | Value |
|---|---|
| Validation Accuracy | 100%* |
| F1 Score | 0.0 |
| Precision | 0.0 |
| Recall | 0.0 |

_*100% accuracy with 0.0 F1 indicates degenerate predictions (model predicted a single class for all samples due to severe class imbalance in the small dataset)._

### Why Multi-Source Training Matters

The single-source model was unusable: it overfit on a tiny, imbalanced dataset and failed to distinguish NLCs from other clouds. The multi-source model fixes this by:

1. **Balanced classes** — 445 NLC-positive vs 445 non-NLC samples
2. **Hard negatives** — Cirrus, nacreous, contrails, and other visually similar clouds force the model to learn NLC-specific features
3. **Diverse positives** — NLC images from both professional galleries and citizen science observations

---

## Training Data Sources

| Source | Samples | Type | Description |
|---|---|---|---|
| Cloud Appreciation Society | 445 | 7 NLC + 438 non-NLC | 49 cloud types; 51 are "hard negatives" (visually similar to NLCs) |
| Spaceweather Gallery | 225 | NLC positive | Curated NLC images with local files |
| Space Cloud Watch CSV | 220 | Mixed | Citizen science observations with images |
| **Total** | **890** | **445 pos / 445 neg** | Perfectly balanced dataset |

### Hard-Negative Cloud Types

These cloud types look most similar to NLCs and are weighted as strong negatives during training:

`cirrus` · `cirrostratus` · `cirrocumulus` · `nacreous` · `undulatus` · `fibratus` · `contrail` · `crepuscular-rays` · `sun-pillar`

---

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
python -m venv .venv
source .venv/bin/activate

pip install torch torchvision timm pandas pillow requests scikit-learn tqdm matplotlib numpy flask
```

---

## Usage

### Train (Multi-Source)

```bash
python main.py train \
  --csv space-cloud-watch_*.csv \
  --gallery-csv spaceweather_gallery_data.csv \
  --cas-dir "Cloud Appreciation Society data" \
  --epochs 30 --batch-size 16
```

### Train (Single CSV, Legacy)

```bash
python main.py train --csv space-cloud-watch_*.csv --epochs 30
```

### Analyze Dataset

```bash
python main.py analyze --csv space-cloud-watch_*.csv
```

### Predict Single Image

```bash
python main.py predict --checkpoint checkpoints/best_model.pt --input image.jpg
```

### Batch Predict with Review Queue

```bash
python main.py batch-predict \
  --checkpoint checkpoints/best_model.pt \
  --csv data.csv \
  --output-dir predictions/
```

### Interactive Image Navigator

```bash
python test_image_navigator.py
```

### Web Portal

```bash
python main.py portal --port 5001
# Open http://localhost:5001
```

The portal provides:
- Dataset overview with source counts and class balance
- Training controls with live progress monitoring
- Image prediction with drag-and-drop support

---

## Technical Architecture

| Component | Detail |
|---|---|
| Base Model | EfficientNet-B0 (ImageNet pretrained, 5.3M params) |
| Binary Head | NLC presence detection (CrossEntropyLoss) |
| Type Head | Multi-label NLC type classification (BCEWithLogitsLoss) |
| Optimizer | AdamW with differential learning rates |
| Scheduler | Cosine annealing with 3-epoch warmup |
| Augmentation | Strong (random crop, flip, color jitter, perspective, erasing) |
| Input Size | 224×224 RGB with ImageNet normalization |
| Hardware | MPS (Apple Silicon), CUDA, or CPU |

### Prediction Workflow

```
Image → EfficientNet-B0 → Confidence Score → Decision Router
                                                  │
                    ┌─────────────────────────────┼─────────────────────────┐
                    │                             │                         │
              High Confidence              Medium Confidence          Low Confidence
              (≥ 85%)                      (65–85%)                   (< 65%)
                    │                             │                         │
              Auto-Accept                  Flag for Review           Priority Review
```

---

## Confidence Thresholds

| Threshold | Value | Meaning |
|---|---|---|
| High Confidence | ≥ 0.85 | Auto-accept (still logged) |
| Low Confidence | < 0.65 | Mandatory human review |
| Entropy | > 0.9 | Potential out-of-distribution input |

```bash
# More conservative (more human review)
python main.py batch-predict --high-threshold 0.90 --low-threshold 0.70 ...

# More permissive (less human review)
python main.py batch-predict --high-threshold 0.80 --low-threshold 0.60 ...
```

---

## Failure Modes and Limitations

### Known Weaknesses

| Issue | Why | Mitigation |
|---|---|---|
| Cirrus confusion | Similar wispy appearance | Hard-negative training helps, but edge cases remain |
| Aurora co-occurrence | Can appear alongside NLCs | Check for colour differences |
| Light pollution | Similar blue/white tones | Geographic context helps |
| Overexposed images | Washes out NLC detail | Flag for review via entropy |

### Geographic and Temporal Bias

- Training data skews toward Northern latitudes (45°N–65°N)
- NLCs are seasonal (mainly May–August in Northern Hemisphere)
- Southern Hemisphere data is underrepresented

### Deployment Recommendations

1. All predictions should be logged for audit
2. Randomly spot-check auto-accepted predictions
3. Feed corrections back into training data
4. Consider location and season when interpreting results

---

## Project Structure

```
CitSci/
├── main.py                    # CLI entry point (train, predict, portal)
├── nlc_classifier/
│   ├── config.py              # Configuration dataclasses
│   ├── data.py                # Dataset, augmentation, dataloaders
│   ├── data_sources.py        # Multi-source data loading (CAS, Gallery, SCW)
│   ├── models.py              # EfficientNet-B0 / ViT model architectures
│   ├── train.py               # Training loop, metrics, checkpointing
│   └── inference.py           # Prediction engine with review queue
├── portal/
│   ├── app.py                 # Flask web portal
│   ├── templates/index.html   # Dashboard UI
│   └── static/style.css       # Portal styling
├── test_image_navigator.py    # Interactive image browser + predictor
├── test_predictions.py        # Batch prediction test script
├── Cloud Appreciation Society data/   # CAS cloud images (49 types)
├── spaceweather_gallery_images/       # Gallery NLC images
├── spaceweather_gallery_data.csv      # Gallery metadata
└── space-cloud-watch_*.csv            # Space Cloud Watch observations
```
