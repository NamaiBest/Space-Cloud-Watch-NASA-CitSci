#!/usr/bin/env python3
"""Test the trained NLC classifier on sample images."""

from nlc_classifier.inference import load_inference_engine
import pandas as pd

def main():
    # Load model (use CPU for inference compatibility)
    print('Loading model...')
    engine = load_inference_engine('checkpoints/best_model.pt', device='cpu')
    print('✅ Model loaded!')

    # Get some test images with their ground truth labels
    df = pd.read_csv('space-cloud-watch_2026-Jan-16_163101_019bc7a4-c9b5-7851-b2e4-0d2c1a430048.csv')
    label_col = 'did you see nlc?'

    # Get image URL columns
    img_cols = [c for c in df.columns if 'image' in c.lower() or 'upload' in c.lower()]

    # Test on 20 images
    print('\n' + '='*60)
    print('Running predictions on sample images...')
    print('='*60)

    tested = 0
    correct = 0
    
    for idx, row in df.iterrows():
        if tested >= 20:
            break
        
        # Find first valid image URL in this row
        url = None
        for col in img_cols:
            if pd.notna(row.get(col)) and str(row.get(col)).startswith('http'):
                url = str(row[col])
                break
        
        if url is None:
            continue
        
        ground_truth = row[label_col] if pd.notna(row.get(label_col)) else 'Unknown'
        
        try:
            result = engine.predict(url)
            
            pred_label = "NLC Present" if result.predicted_class == 1 else "No NLC"
            is_correct = (result.predicted_class == 1 and ground_truth == 'Yes') or \
                         (result.predicted_class == 0 and ground_truth == 'No')
            
            if is_correct:
                correct += 1
                match_icon = '✓'
            else:
                match_icon = '✗'
            
            review_status = '⚠️ REVIEW' if result.needs_review else '✅ AUTO'
            
            print(f'\nImage {tested + 1}:')
            print(f'  Ground Truth: {ground_truth}')
            print(f'  Prediction:   {pred_label}')
            print(f'  Confidence:   {result.confidence:.1%}')
            print(f'  Entropy:      {result.entropy:.3f}')
            print(f'  Result: {match_icon} | Status: {review_status}')
            
            tested += 1
        except Exception as e:
            print(f'  ⚠️ Error loading image: {str(e)[:50]}...')

    print('\n' + '='*60)
    print(f'SUMMARY: {correct}/{tested} correct ({100*correct/tested:.1f}% accuracy)')
    print('='*60)


if __name__ == '__main__':
    main()
