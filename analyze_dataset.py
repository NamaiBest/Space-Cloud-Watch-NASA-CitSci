#!/usr/bin/env python
"""
Quick dataset analysis script for Space Cloud Watch NLC data.
Run this to understand the dataset before training.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_nlc_dataset(csv_path: str):
    """Analyze the NLC dataset and print comprehensive statistics."""
    
    print("=" * 70)
    print("SPACE CLOUD WATCH - NLC DATASET ANALYSIS")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(csv_path)
    
    print(f"\nDataset: {csv_path}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # Identify key columns
    print("\n" + "-" * 50)
    print("COLUMN ANALYSIS")
    print("-" * 50)
    
    for col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  {col[:50]:50} | {non_null:4} non-null")
    
    # Label distribution
    print("\n" + "-" * 50)
    print("LABEL DISTRIBUTION")
    print("-" * 50)
    
    label_col = 'did you see nlc?'
    if label_col in df.columns:
        label_counts = df[label_col].value_counts()
        total = len(df)
        print(f"\nPrimary Label: '{label_col}'")
        for label, count in label_counts.items():
            pct = count / total * 100
            print(f"  {label:20} | {count:4} ({pct:5.1f}%)")
        
        # Class imbalance
        if len(label_counts) >= 2:
            majority = label_counts.iloc[0]
            minority = label_counts.iloc[-1]
            ratio = majority / minority
            print(f"\nClass Imbalance Ratio: {ratio:.2f}:1")
            if ratio > 3:
                print("  ⚠️  Significant imbalance - weighted sampling recommended")
    
    # Image availability
    print("\n" + "-" * 50)
    print("IMAGE AVAILABILITY")
    print("-" * 50)
    
    image_cols = [col for col in df.columns if 'upload' in col.lower() or 'image' in col.lower()]
    
    total_with_images = 0
    for col in image_cols:
        urls = df[col].dropna()
        http_urls = urls[urls.astype(str).str.startswith('http')]
        if len(http_urls) > 0:
            print(f"\n  Column: {col[:60]}")
            print(f"    Valid image URLs: {len(http_urls)}")
            total_with_images += len(http_urls)
    
    # Deduplicate by checking unique URLs
    all_urls = set()
    for col in image_cols:
        urls = df[col].dropna()
        http_urls = urls[urls.astype(str).str.startswith('http')]
        all_urls.update(http_urls.tolist())
    
    print(f"\n  Total unique image URLs: {len(all_urls)}")
    
    # By label with images
    print("\n  Images by Label:")
    for col in image_cols:
        if df[col].notna().any():
            has_image = df[col].notna() & df[col].astype(str).str.startswith('http')
            label_with_images = df[has_image][label_col].value_counts()
            for label, count in label_with_images.items():
                print(f"    {label}: {count}")
            break
    
    # NLC Types distribution
    print("\n" + "-" * 50)
    print("NLC TYPE DISTRIBUTION")
    print("-" * 50)
    
    types_col = 'types of nlc'
    if types_col in df.columns:
        types_data = df[types_col].dropna()
        type_counts = {}
        for types_str in types_data:
            for t in str(types_str).split(','):
                t = t.strip()
                if t:
                    type_counts[t] = type_counts.get(t, 0) + 1
        
        print(f"\n  Total samples with type info: {len(types_data)}")
        for type_name, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {type_name:40} | {count:3}")
    
    # Geographic distribution
    print("\n" + "-" * 50)
    print("GEOGRAPHIC DISTRIBUTION")
    print("-" * 50)
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat = df['latitude'].dropna()
        lon = df['longitude'].dropna()
        
        print(f"\n  Latitude range:  {lat.min():.2f}° to {lat.max():.2f}°")
        print(f"  Longitude range: {lon.min():.2f}° to {lon.max():.2f}°")
        
        # Latitude bands
        print("\n  Latitude bands (for NLC visibility):")
        lat_bands = [
            (45, 55, "Mid-latitude (rare NLCs)"),
            (55, 65, "High-latitude (common NLCs)"),
            (65, 90, "Polar region (frequent NLCs)")
        ]
        for low, high, name in lat_bands:
            count = ((lat >= low) & (lat < high)).sum()
            print(f"    {low}°-{high}°N ({name}): {count}")
    
    # Temporal distribution
    print("\n" + "-" * 50)
    print("TEMPORAL DISTRIBUTION")
    print("-" * 50)
    
    date_col = 'observedDate'
    if date_col in df.columns:
        df['parsed_date'] = pd.to_datetime(df[date_col], format='%d-%b-%Y %H:%M:%S', errors='coerce')
        valid_dates = df['parsed_date'].dropna()
        
        print(f"\n  Date range: {valid_dates.min()} to {valid_dates.max()}")
        
        # By month
        print("\n  Observations by month:")
        month_counts = valid_dates.dt.month.value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month, count in month_counts.items():
            print(f"    {month_names[month-1]}: {count}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR TRAINING")
    print("=" * 70)
    
    usable_samples = len(all_urls)
    print(f"\n  Usable samples (with images): {usable_samples}")
    
    if usable_samples < 100:
        print("\n  ⚠️  VERY SMALL DATASET")
        print("      - Consider data augmentation strategies")
        print("      - Use cross-validation instead of single split")
        print("      - Expect limited generalization")
    elif usable_samples < 500:
        print("\n  📊 SMALL DATASET")
        print("      - Use EfficientNet-B0 (NOT ViT)")
        print("      - Apply strong data augmentation")
        print("      - Use dropout (0.3-0.5)")
        print("      - Consider early stopping")
    else:
        print("\n  ✅ REASONABLE DATASET SIZE")
        print("      - EfficientNet-B0 recommended")
        print("      - Standard augmentation should suffice")
    
    # Check imbalance
    if label_col in df.columns:
        label_counts = df[label_col].value_counts()
        if len(label_counts) >= 2:
            ratio = label_counts.iloc[0] / label_counts.iloc[-1]
            if ratio > 2:
                print(f"\n  ⚠️  CLASS IMBALANCE (ratio: {ratio:.1f}:1)")
                print("      - Weighted sampling enabled automatically")
                print("      - Monitor per-class metrics during training")
    
    print("\n" + "=" * 70)
    print("Ready to train! Run: python main.py train --csv <path>")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default to the CSV in the current directory
        csv_files = list(Path(".").glob("*.csv"))
        if csv_files:
            csv_path = str(csv_files[0])
            print(f"Using: {csv_path}")
        else:
            print("Usage: python analyze_dataset.py <path_to_csv>")
            sys.exit(1)
    
    analyze_nlc_dataset(csv_path)
