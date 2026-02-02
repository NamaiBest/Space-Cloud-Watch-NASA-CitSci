"""
Data loading, preprocessing, and augmentation for NLC classification.
Handles CSV parsing, image downloading/caching, and train/val splits.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config import NLCConfig, DataConfig


class ImageCache:
    """Manages local caching of downloaded images."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _url_to_filename(self, url: str) -> str:
        """Generate a unique filename from URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        ext = Path(urlparse(url).path).suffix or '.jpg'
        return f"{url_hash}{ext}"
    
    def get_cached_path(self, url: str) -> Optional[Path]:
        """Get path to cached image if it exists."""
        filename = self._url_to_filename(url)
        path = self.cache_dir / filename
        return path if path.exists() else None
    
    def download_and_cache(self, url: str, timeout: int = 30) -> Optional[Path]:
        """Download image and cache locally."""
        cached = self.get_cached_path(url)
        if cached:
            return cached
        
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            # Verify it's a valid image
            img = Image.open(BytesIO(response.content))
            img.verify()
            
            # Save to cache
            filename = self._url_to_filename(url)
            path = self.cache_dir / filename
            with open(path, 'wb') as f:
                f.write(response.content)
            
            return path
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None


def parse_nlc_csv(csv_path: str, config: DataConfig) -> pd.DataFrame:
    """
    Parse the Space Cloud Watch CSV and extract relevant fields.
    
    Returns DataFrame with columns:
    - observation_id: Unique identifier
    - image_url: URL to the image (if available)
    - label: Binary label (0=No NLC, 1=Yes NLC)
    - nlc_types: List of NLC types if Yes
    - latitude, longitude: Location data
    - observed_date: When observation was made
    """
    df = pd.read_csv(csv_path)
    
    # Rename columns for easier access
    column_mapping = {
        'observationid': 'observation_id',
        'did you see nlc?': 'nlc_seen',
        'types of nlc': 'nlc_types',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'observedDate': 'observed_date',
        'locationName': 'location_name',
        'comments_observation': 'comments'
    }
    
    # Apply mapping for columns that exist
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Extract image URLs from multiple possible columns
    def get_image_url(row):
        for col in config.image_columns:
            if col in row.index and pd.notna(row[col]) and str(row[col]).startswith('http'):
                return str(row[col])
        return None
    
    df['image_url'] = df.apply(get_image_url, axis=1)
    
    # Create binary label
    df['label'] = df['nlc_seen'].map({'Yes': 1, 'No': 0})
    
    # Parse NLC types into list
    def parse_types(type_str):
        if pd.isna(type_str) or not type_str:
            return []
        types = []
        for t in str(type_str).split(','):
            t = t.strip()
            if t in config.nlc_type_labels:
                types.append(config.nlc_type_labels[t])
        return types
    
    df['nlc_type_indices'] = df['nlc_types'].apply(parse_types)
    
    return df


def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the dataset and return statistics.
    Useful for understanding class imbalance and data quality.
    """
    total = len(df)
    with_images = df['image_url'].notna().sum()
    
    # Label distribution
    label_counts = df['label'].value_counts().to_dict()
    
    # NLC type distribution (for positive samples)
    positive_df = df[df['label'] == 1]
    type_counts = {}
    for types in positive_df['nlc_type_indices']:
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
    
    # Geographic distribution
    lat_range = (df['latitude'].min(), df['latitude'].max())
    lon_range = (df['longitude'].min(), df['longitude'].max())
    
    stats = {
        'total_observations': total,
        'observations_with_images': with_images,
        'label_distribution': label_counts,
        'nlc_type_distribution': type_counts,
        'latitude_range': lat_range,
        'longitude_range': lon_range,
        'class_imbalance_ratio': label_counts.get(0, 0) / max(label_counts.get(1, 1), 1),
        'usable_samples': with_images  # Only samples with images can be used
    }
    
    return stats


def get_augmentation_transforms(config: NLCConfig) -> transforms.Compose:
    """
    Get data augmentation transforms based on configuration.
    
    NLC-specific considerations:
    - Preserve color information (important for identifying NLC's bluish tint)
    - Avoid extreme rotations (horizon orientation matters)
    - Use horizontal flips (sky is symmetric)
    - Apply color jitter carefully (NLCs have distinctive colors)
    """
    strength = config.training.augmentation_strength
    size = config.model.image_size
    mean = config.model.normalize_mean
    std = config.model.normalize_std
    
    if strength == "light":
        return transforms.Compose([
            transforms.Resize((size + 32, size + 32)),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    elif strength == "medium":
        return transforms.Compose([
            transforms.Resize((size + 32, size + 32)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    else:  # strong - recommended for small datasets
        return transforms.Compose([
            transforms.Resize((size + 48, size + 48)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            # Random erasing helps with robustness
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            transforms.Normalize(mean=mean, std=std)
        ])


def get_validation_transforms(config: NLCConfig) -> transforms.Compose:
    """Get transforms for validation/inference (no augmentation)."""
    size = config.model.image_size
    mean = config.model.normalize_mean
    std = config.model.normalize_std
    
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


class NLCDataset(Dataset):
    """
    PyTorch Dataset for NLC image classification.
    Handles image loading from URLs or local cache.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: NLCConfig,
        transform: Optional[transforms.Compose] = None,
        is_training: bool = True,
        cache: Optional[ImageCache] = None
    ):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.transform = transform
        self.is_training = is_training
        self.cache = cache or ImageCache(config.data.image_cache_dir)
        
        # Filter to only samples with images
        self.df = self.df[self.df['image_url'].notna()].reset_index(drop=True)
        
        # Pre-download all images during initialization
        self._preload_images()
    
    def _preload_images(self):
        """Download and cache all images upfront."""
        valid_indices = []
        print(f"Preloading {len(self.df)} images...")
        
        for idx in tqdm(range(len(self.df)), desc="Caching images"):
            url = self.df.loc[idx, 'image_url']
            path = self.cache.download_and_cache(url)
            if path is not None:
                valid_indices.append(idx)
        
        # Keep only samples with successfully downloaded images
        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        print(f"Successfully cached {len(self.df)} images")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        
        # Load image from cache
        image_path = self.cache.get_cached_path(row['image_url'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Return a placeholder if image loading fails
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (self.config.model.image_size, self.config.model.image_size))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = int(row['label'])
        
        # Get NLC type labels (multi-label binary vector)
        nlc_type_vector = torch.zeros(4, dtype=torch.float32)  # 4 types
        for type_idx in row.get('nlc_type_indices', []):
            if 0 <= type_idx < 4:
                nlc_type_vector[type_idx] = 1.0
        
        # Additional metadata for analysis
        metadata = {
            'observation_id': row.get('observation_id', ''),
            'image_url': row['image_url'],
            'nlc_types': row.get('nlc_type_indices', []),
            'nlc_type_vector': nlc_type_vector,
            'latitude': row.get('latitude', 0),
            'longitude': row.get('longitude', 0)
        }
        
        return image, label, nlc_type_vector, metadata


def create_dataloaders(
    csv_path: str,
    config: NLCConfig
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create training and validation dataloaders from CSV.
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        stats: Dataset statistics
    """
    # Parse CSV
    df = parse_nlc_csv(csv_path, config.data)
    
    # Analyze dataset
    stats = analyze_dataset(df)
    print(f"\nDataset Statistics:")
    print(f"  Total observations: {stats['total_observations']}")
    print(f"  With images: {stats['observations_with_images']}")
    print(f"  Class distribution: {stats['label_distribution']}")
    print(f"  Class imbalance ratio: {stats['class_imbalance_ratio']:.2f}")
    
    # Filter to samples with images
    df_with_images = df[df['image_url'].notna()].copy()
    
    if len(df_with_images) == 0:
        raise ValueError("No samples with images found in the dataset!")
    
    # Create stratification key that considers both NLC presence and types
    # This ensures balanced split for both binary label and type distribution
    def create_stratify_key(row):
        if row['label'] == 0:
            return "no_nlc"
        else:
            # Create a key based on which types are present
            types = sorted(row.get('nlc_type_indices', []))
            if not types:
                return "nlc_no_type"
            return f"nlc_types_{'_'.join(map(str, types))}"
    
    df_with_images['stratify_key'] = df_with_images.apply(create_stratify_key, axis=1)
    
    # For rare combinations, group them to ensure stratification works
    key_counts = df_with_images['stratify_key'].value_counts()
    rare_keys = key_counts[key_counts < 2].index.tolist()
    df_with_images.loc[df_with_images['stratify_key'].isin(rare_keys), 'stratify_key'] = \
        df_with_images.loc[df_with_images['stratify_key'].isin(rare_keys), 'label'].apply(
            lambda x: 'nlc_other' if x == 1 else 'no_nlc'
        )
    
    # Stratified train/val split (80/20)
    train_df, val_df = train_test_split(
        df_with_images,
        test_size=1 - config.data.train_split,
        stratify=df_with_images['stratify_key'],
        random_state=config.data.random_seed
    )
    
    # Print split statistics
    print(f"\nSplit sizes:")
    print(f"  Training: {len(train_df)}")
    print(f"  Validation: {len(val_df)}")
    
    # Show type distribution in train/val
    train_nlc = train_df[train_df['label'] == 1]
    val_nlc = val_df[val_df['label'] == 1]
    print(f"\nNLC class distribution:")
    print(f"  Training: {(train_df['label'] == 1).sum()} NLC, {(train_df['label'] == 0).sum()} No NLC")
    print(f"  Validation: {(val_df['label'] == 1).sum()} NLC, {(val_df['label'] == 0).sum()} No NLC")
    
    # Count type distribution
    type_names = ['Type 1 (Veil)', 'Type 2 (Bands)', 'Type 3 (Waves)', 'Type 4 (Whirls)']
    print(f"\nNLC Type distribution (multi-label):")
    for i, name in enumerate(type_names):
        train_count = sum(1 for types in train_nlc['nlc_type_indices'] if i in types)
        val_count = sum(1 for types in val_nlc['nlc_type_indices'] if i in types)
        print(f"  {name}: Train={train_count}, Val={val_count}")
    
    # Create transforms
    train_transform = get_augmentation_transforms(config)
    val_transform = get_validation_transforms(config)
    
    # Create datasets
    cache = ImageCache(config.data.image_cache_dir)
    train_dataset = NLCDataset(train_df, config, train_transform, is_training=True, cache=cache)
    val_dataset = NLCDataset(val_df, config, val_transform, is_training=False, cache=cache)
    
    # Handle class imbalance with weighted sampling
    # Disable pin_memory for MPS (not supported)
    use_pin_memory = torch.cuda.is_available()
    
    if stats['class_imbalance_ratio'] > 2:
        print("\nUsing weighted sampling to handle class imbalance...")
        labels = train_dataset.df['label'].values
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=use_pin_memory,
            collate_fn=collate_fn
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=use_pin_memory,
            collate_fn=collate_fn
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, stats


def collate_fn(batch):
    """Custom collate function to handle metadata and type labels."""
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    type_vectors = torch.stack([item[2] for item in batch])
    metadata = [item[3] for item in batch]
    return images, labels, type_vectors, metadata


if __name__ == "__main__":
    # Quick test
    from .config import get_default_config
    config = get_default_config()
    config.data.csv_path = "../space-cloud-watch_2026-Jan-16_163101_019bc7a4-c9b5-7851-b2e4-0d2c1a430048.csv"
    
    train_loader, val_loader, stats = create_dataloaders(config.data.csv_path, config)
    print(f"\nDataLoader test:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
