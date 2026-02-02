#!/usr/bin/env python3
"""
Interactive NLC Image Navigator & Predictor

This script allows you to:
1. Browse images in the "Test Image" folder
2. Select an image using simple number input
3. Run the trained NLC model on the selected image
4. See detailed prediction results

What the model predicts:
========================
- Binary Classification: NLC Present (Yes) or No NLC (No)
- Confidence Score: How confident the model is (0-100%)
- Entropy: Uncertainty measure (lower = more certain)
- Review Flag: Whether human review is recommended
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from nlc_classifier.inference import load_inference_engine


def get_image_files(folder_path: str) -> list:
    """Get all image files from the folder."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    images = []
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return []
    
    for file in sorted(folder.iterdir()):
        if file.is_file() and file.suffix.lower() in valid_extensions:
            images.append(file)
    
    return images


def display_menu(images: list) -> None:
    """Display the image selection menu."""
    print("\n" + "=" * 60)
    print("🌌 NLC (Noctilucent Cloud) Image Navigator")
    print("=" * 60)
    print("\nAvailable images in 'Test Image' folder:\n")
    
    for idx, img in enumerate(images, 1):
        size_kb = img.stat().st_size / 1024
        print(f"  [{idx}] {img.name} ({size_kb:.1f} KB)")
    
    print(f"\n  [0] Exit")
    print("-" * 60)


def display_prediction_result(result, image_path: str) -> None:
    """Display the prediction result in a nice format."""
    print("\n" + "=" * 60)
    print("🔬 PREDICTION RESULTS")
    print("=" * 60)
    
    print(f"\n📁 Image: {Path(image_path).name}")
    print("-" * 40)
    
    # Main prediction
    nlc_status = "🌌 NLC PRESENT" if result.predicted_class == 1 else "❌ NO NLC DETECTED"
    print(f"\n  🎯 Prediction: {nlc_status}")
    print(f"  📊 Confidence: {result.confidence:.1%}")
    
    # Probability breakdown
    print(f"\n  📈 Class Probabilities:")
    for label, prob in result.probabilities.items():
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        print(f"      {label}: {bar} {prob:.1%}")
    
    # NLC Types (if NLC detected)
    if result.predicted_class == 1 and hasattr(result, 'nlc_types'):
        print(f"\n  🏷️  NLC Types Detected:")
        if result.nlc_types:
            for t in result.nlc_types:
                print(f"      ✓ {t}")
        else:
            print(f"      (No specific type detected)")
        
        print(f"\n  📊 Type Probabilities:")
        for type_name, prob in result.nlc_type_probabilities.items():
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            marker = "✓" if prob > 0.5 else " "
            print(f"    {marker} {type_name}: {bar} {prob:.1%}")
    
    # Uncertainty metrics
    print(f"\n  🔍 Uncertainty Metrics:")
    print(f"      Entropy: {result.entropy:.4f} (lower = more certain)")
    
    # Review recommendation
    print(f"\n  👤 Human Review:")
    if result.needs_review:
        print(f"      ⚠️  REVIEW RECOMMENDED")
        print(f"      Reason: {result.review_reason}")
        print(f"      Priority: {'🔴 HIGH' if result.review_priority == 1 else '🟡 MEDIUM' if result.review_priority == 2 else '🟢 LOW'}")
    else:
        print(f"      ✅ Auto-accepted (high confidence)")
    
    print("\n" + "=" * 60)


def explain_model():
    """Explain what the model predicts."""
    print("\n" + "=" * 60)
    print("📚 WHAT DOES THIS MODEL PREDICT?")
    print("=" * 60)
    print("""
The NLC (Noctilucent Cloud) Classifier is a vision-only multi-task model that:

┌─────────────────────────────────────────────────────────────┐
│  TASK 1: Binary Classification (NLC Detection)             │
├─────────────────────────────────────────────────────────────┤
│  • Class 0: NO NLC      - No noctilucent clouds visible     │
│  • Class 1: NLC PRESENT - Noctilucent clouds detected       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  TASK 2: Multi-Label Classification (NLC Types)            │
├─────────────────────────────────────────────────────────────┤
│  If NLC is detected, the model also predicts which types:   │
│                                                             │
│  • Type 1 (Veil)   - Tenuous, featureless layer             │
│  • Type 2 (Bands)  - Long parallel streaks                  │
│  • Type 3 (Waves)  - Short, closely spaced undulations      │
│  • Type 4 (Whirls) - Partial or complete rings              │
│                                                             │
│  NOTE: An image can have MULTIPLE types simultaneously!     │
└─────────────────────────────────────────────────────────────┘

Additional outputs for each prediction:

  📊 CONFIDENCE SCORE (0-100%)
     How certain the model is about NLC presence/absence.
     > 85% = High confidence (auto-accept)
     65-85% = Medium confidence (may need review)
     < 65% = Low confidence (needs human review)

  🏷️  NLC TYPE PROBABILITIES
     Per-type probability (threshold: 50% to include)
     Images can have multiple types detected

  📈 ENTROPY
     Uncertainty measure based on probability distribution
     Lower entropy = model is more certain

  👤 REVIEW FLAG
     Whether the prediction should be reviewed by a human

┌─────────────────────────────────────────────────────────────┐
│  MODEL DETAILS                                               │
├─────────────────────────────────────────────────────────────┤
│  Architecture: EfficientNet-B0 (pretrained on ImageNet)     │
│  Training: Multi-task learning (detection + type)           │
│  Training Data: ~219 citizen science observations           │
│  Train/Test Split: 80/20 stratified                         │
│  Input Size: 224x224 pixels                                 │
└─────────────────────────────────────────────────────────────┘
""")
    input("Press Enter to continue...")


def main():
    test_folder = Path(__file__).parent / "Test Image"
    
    # Load the model once
    print("\n🔄 Loading NLC classification model...")
    try:
        engine = load_inference_engine('checkpoints/best_model.pt', device='cpu')
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   Make sure 'checkpoints/best_model.pt' exists.")
        return
    
    while True:
        # Get available images
        images = get_image_files(test_folder)
        
        if not images:
            print(f"\n⚠️  No images found in '{test_folder}'")
            print("   Please add some images (.jpg, .png, etc.) to the folder.")
            break
        
        # Display menu
        display_menu(images)
        
        print("\nOptions:")
        print("  Enter a number (1-{}) to analyze an image".format(len(images)))
        print("  Enter 'i' for model info")
        print("  Enter 'a' to analyze ALL images")
        print("  Enter '0' or 'q' to quit")
        
        choice = input("\n👉 Your choice: ").strip().lower()
        
        if choice in ['0', 'q', 'quit', 'exit']:
            print("\n👋 Goodbye!")
            break
        
        if choice == 'i':
            explain_model()
            continue
        
        if choice == 'a':
            # Analyze all images
            print("\n" + "=" * 60)
            print("📊 BATCH ANALYSIS - All Images")
            print("=" * 60)
            
            for idx, img in enumerate(images, 1):
                print(f"\n[{idx}/{len(images)}] Analyzing: {img.name}")
                try:
                    result = engine.predict(str(img))
                    status = "🌌 NLC" if result.predicted_class == 1 else "❌ No NLC"
                    review = "⚠️ Review" if result.needs_review else "✅ Auto"
                    print(f"    {status} | Confidence: {result.confidence:.1%} | {review}")
                except Exception as e:
                    print(f"    ❌ Error: {str(e)[:50]}")
            
            print("\n" + "=" * 60)
            input("\nPress Enter to continue...")
            continue
        
        # Try to parse as number
        try:
            idx = int(choice)
            if idx < 1 or idx > len(images):
                print(f"\n⚠️  Please enter a number between 1 and {len(images)}")
                continue
            
            selected_image = images[idx - 1]
            print(f"\n🔍 Analyzing: {selected_image.name}...")
            
            try:
                result = engine.predict(str(selected_image))
                display_prediction_result(result, str(selected_image))
            except Exception as e:
                print(f"\n❌ Error analyzing image: {e}")
            
            input("\nPress Enter to continue...")
            
        except ValueError:
            print(f"\n⚠️  Invalid input. Please enter a number or 'q' to quit.")


if __name__ == "__main__":
    main()
