#!/usr/bin/env python
"""
Main entry point for NLC (Noctilucent Cloud) Classification Pipeline.
NASA Citizen Science - Space Cloud Watch

Usage:
    # Analyze dataset
    python main.py analyze --csv path/to/data.csv
    
    # Train model
    python main.py train --csv path/to/data.csv [--config config.json]
    
    # Run inference
    python main.py predict --checkpoint best_model.pt --input image.jpg
    
    # Batch inference with review queue
    python main.py batch-predict --checkpoint best_model.pt --csv path/to/data.csv
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description="NLC Image Classification Pipeline for NASA Space Cloud Watch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
  # Analyze the dataset
  python main.py analyze --csv data.csv

  # Train with default settings (recommended)
  python main.py train --csv data.csv

  # Train with custom config
  python main.py train --csv data.csv --epochs 50 --batch-size 8

  # Predict single image
  python main.py predict --checkpoint checkpoints/best_model.pt --input image.jpg

  # Batch predict with human review queue
  python main.py batch-predict --checkpoint checkpoints/best_model.pt --csv data.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset')
    analyze_parser.add_argument('--csv', required=True, help='Path to CSV file')
    analyze_parser.add_argument('--output', help='Output path for analysis report')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--csv', required=True, help='Path to CSV file')
    train_parser.add_argument('--model', choices=['efficientnet_b0', 'vit_small'],
                             default='efficientnet_b0', help='Model architecture')
    train_parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    train_parser.add_argument('--resume', help='Resume from checkpoint')
    train_parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], 
                             default='auto', help='Device to use')
    
    # Predict command (single image)
    predict_parser = subparsers.add_parser('predict', help='Predict single image')
    predict_parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    predict_parser.add_argument('--input', required=True, help='Image path or URL')
    predict_parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'],
                               default='auto', help='Device to use')
    
    # Batch predict command
    batch_parser = subparsers.add_parser('batch-predict', help='Batch prediction with review queue')
    batch_parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    batch_parser.add_argument('--csv', required=True, help='CSV with image URLs')
    batch_parser.add_argument('--output-dir', default='predictions', help='Output directory')
    batch_parser.add_argument('--high-threshold', type=float, default=0.85,
                             help='Confidence threshold for auto-accept')
    batch_parser.add_argument('--low-threshold', type=float, default=0.65,
                             help='Confidence threshold for required review')
    batch_parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'],
                             default='auto', help='Device to use')
    
    return parser


def cmd_analyze(args):
    """Analyze the dataset."""
    from nlc_classifier.config import get_default_config
    from nlc_classifier.data import parse_nlc_csv, analyze_dataset
    
    print("=" * 60)
    print("NLC Dataset Analysis")
    print("=" * 60)
    
    config = get_default_config()
    df = parse_nlc_csv(args.csv, config.data)
    stats = analyze_dataset(df)
    
    print(f"\nDataset: {args.csv}")
    print("-" * 40)
    print(f"Total observations: {stats['total_observations']}")
    print(f"Observations with images: {stats['observations_with_images']}")
    print(f"\nLabel Distribution:")
    for label, count in stats['label_distribution'].items():
        label_name = "NLC Present" if label == 1 else "No NLC"
        print(f"  {label_name}: {count} ({count/stats['total_observations']*100:.1f}%)")
    
    print(f"\nClass Imbalance Ratio: {stats['class_imbalance_ratio']:.2f}")
    
    if stats['nlc_type_distribution']:
        print(f"\nNLC Type Distribution (for positive samples):")
        type_names = {0: "Type 1 (Veil)", 1: "Type 2 (Bands)", 
                     2: "Type 3 (Waves)", 3: "Type 4 (Whirls)"}
        for type_id, count in stats['nlc_type_distribution'].items():
            print(f"  {type_names.get(type_id, f'Type {type_id}')}: {count}")
    
    print(f"\nGeographic Range:")
    print(f"  Latitude: {stats['latitude_range'][0]:.2f} to {stats['latitude_range'][1]:.2f}")
    print(f"  Longitude: {stats['longitude_range'][0]:.2f} to {stats['longitude_range'][1]:.2f}")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATION:")
    print("-" * 40)
    
    usable = stats['observations_with_images']
    if usable < 100:
        print("⚠️  Very small dataset. Consider collecting more data.")
        print("   Model will have limited generalization ability.")
    elif usable < 500:
        print("📊 Small dataset. Using EfficientNet-B0 with strong augmentation.")
        print("   Consider using cross-validation for robust evaluation.")
    elif usable < 1000:
        print("✓ Reasonable dataset size for fine-tuning.")
        print("   EfficientNet-B0 recommended with medium augmentation.")
    else:
        print("✓ Good dataset size. Both EfficientNet-B0 and ViT are viable.")
    
    if stats['class_imbalance_ratio'] > 3:
        print(f"\n⚠️  Significant class imbalance detected (ratio: {stats['class_imbalance_ratio']:.1f})")
        print("   Weighted sampling will be used during training.")
    
    print("=" * 60)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"\nAnalysis saved to {args.output}")


def cmd_train(args):
    """Train the model."""
    from nlc_classifier.config import get_default_config
    from nlc_classifier.train import train_nlc_classifier
    
    print("=" * 60)
    print("NLC Classifier Training")
    print("=" * 60)
    
    # Setup config
    config = get_default_config()
    config.model.model_name = args.model
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.checkpoint_dir = args.checkpoint_dir
    config.device = args.device
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model.model_name}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Device: {config.get_device()}")
    print("-" * 40)
    
    # Train
    model, results = train_nlc_classifier(
        args.csv,
        config,
        resume_from=args.resume
    )
    
    print("\nTraining complete!")
    print(f"Best model saved to: {args.checkpoint_dir}/best_model.pt")
    
    return model, results


def cmd_predict(args):
    """Predict single image."""
    from nlc_classifier.config import get_default_config
    from nlc_classifier.inference import load_inference_engine
    
    print("=" * 60)
    print("NLC Single Image Prediction")
    print("=" * 60)
    
    config = get_default_config()
    config.device = args.device
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    engine = load_inference_engine(args.checkpoint, config, config.get_device())
    
    # Predict
    print(f"Processing: {args.input}")
    result = engine.predict(args.input)
    
    print("\n" + "-" * 40)
    print("PREDICTION RESULT")
    print("-" * 40)
    print(f"Image: {result.image_path}")
    print(f"Predicted Class: {result.predicted_label}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Entropy: {result.entropy:.3f}")
    print(f"\nProbabilities:")
    for label, prob in result.probabilities.items():
        print(f"  {label}: {prob:.1%}")
    
    print(f"\nNeeds Human Review: {'YES' if result.needs_review else 'NO'}")
    if result.needs_review:
        print(f"Review Reason: {result.review_reason}")
        print(f"Priority: {result.review_priority}")
    
    print("-" * 40)
    print("⚠️  Note: This is an assistive prediction.")
    print("   Human verification is recommended for scientific use.")
    print("=" * 60)
    
    return result


def cmd_batch_predict(args):
    """Batch prediction with review queue."""
    from nlc_classifier.config import get_default_config
    from nlc_classifier.data import parse_nlc_csv
    from nlc_classifier.inference import load_inference_engine
    
    print("=" * 60)
    print("NLC Batch Prediction with Human Review Queue")
    print("=" * 60)
    
    config = get_default_config()
    config.device = args.device
    config.inference.high_confidence_threshold = args.high_threshold
    config.inference.low_confidence_threshold = args.low_threshold
    
    # Load data
    print(f"\nLoading data from {args.csv}...")
    df = parse_nlc_csv(args.csv, config.data)
    df_with_images = df[df['image_url'].notna()]
    
    print(f"Found {len(df_with_images)} images to process")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    engine = load_inference_engine(args.checkpoint, config, config.get_device())
    
    # Run batch prediction
    print("\nRunning predictions...")
    image_urls = df_with_images['image_url'].tolist()
    image_ids = df_with_images['observation_id'].tolist() if 'observation_id' in df_with_images.columns else None
    
    results = engine.predict_batch(image_urls, image_ids)
    
    # Save results
    print(f"\nSaving results to {args.output_dir}...")
    saved_files = engine.save_results(results, args.output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BATCH PREDICTION SUMMARY")
    print("=" * 60)
    s = results.summary
    print(f"Total images processed: {s['total_images']}")
    print(f"Auto-accepted (high confidence): {s['auto_accepted']}")
    print(f"Flagged for human review: {s['needs_review']} ({s['human_review_percentage']:.1f}%)")
    print(f"Processing errors: {s['errors']}")
    print(f"\nPredicted NLC positive: {s['nlc_positive_predictions']}")
    print(f"Predicted NLC negative: {s['nlc_negative_predictions']}")
    print(f"\nAverage confidence: {s['average_confidence']:.3f}")
    print(f"Average entropy: {s['average_entropy']:.3f}")
    
    print(f"\nReview reasons distribution:")
    for reason, count in s['review_reasons_distribution'].items():
        print(f"  {reason}: {count}")
    
    print(f"\nOutput files:")
    for file_type, path in saved_files.items():
        print(f"  {file_type}: {path}")
    
    print("\n" + "=" * 60)
    print("⚠️  IMPORTANT:")
    print("   Review queue contains images requiring human verification.")
    print("   Do not use auto-accepted predictions without expert review")
    print("   for scientific publications or critical decisions.")
    print("=" * 60)
    
    return results


def main():
    """Main entry point."""
    parser = setup_args()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'batch-predict':
        cmd_batch_predict(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
