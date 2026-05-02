---
title: NLC Detector — NASA Space Cloud Watch
emoji: 🌌
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Upload a sky photo to detect noctilucent clouds (NLC)
---

# NLC Detector — NASA Space Cloud Watch

Upload a sky photograph and find out whether **noctilucent clouds (NLC)** are present, along with per-type classification (Veil, Bands, Waves, Whirls).

## Model

- **Architecture:** EfficientNet-B0 (pretrained on ImageNet, fine-tuned)
- **Task 1:** Binary classification — NLC present vs. absent
- **Task 2:** Multi-label — NLC type (Type 1–4)
- **Validation accuracy:** 91.0% | F1: 0.907
- **Training data:** 890 images from 3 sources (Space Cloud Watch, Spaceweather Gallery, Cloud Appreciation Society)

## Confidence Thresholds

| Level | Threshold | Action |
|---|---|---|
| High | ≥ 85% | Auto-accepted |
| Medium | 65–85% | Review recommended |
| Low | < 65% | Priority review |

> This is an assistive AI tool. Predictions should be verified by a trained observer for scientific use.
