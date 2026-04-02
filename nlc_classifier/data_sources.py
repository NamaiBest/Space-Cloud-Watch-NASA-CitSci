"""
Unified data loading for multi-source NLC classification training.

Loads and merges data from three sources:
1. Cloud Appreciation Society (CAS) — local folder with cloud-type subfolders
2. Spaceweather Gallery — CSV + local image files (all NLC positive)
3. Space Cloud Watch (SCW) — citizen science CSV with mixed labels

Each source is normalised into a common DataFrame schema so the training
pipeline can consume a single unified dataset.
"""

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Cloud types from CAS that are visually similar to NLCs (high-altitude,
# wispy, twilight-lit) — used as "hard negatives" during training.
HARD_NEGATIVE_TYPES = frozenset([
    "cirrus",
    "cirrostratus",
    "cirrocumulus",
    "nacreous",
    "undulatus",
    "fibratus",
    "contrail",
    "crepuscular-rays",
    "sun-pillar",
])

# Image extensions we look for inside CAS folders.
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


# ---------------------------------------------------------------------------
# Individual source loaders
# ---------------------------------------------------------------------------

def load_cas_data(cas_dir: str) -> pd.DataFrame:
    """Load Cloud Appreciation Society folder images.

    Each subfolder is a cloud type.  ``noctilucent/`` → label=1,
    all others → label=0.  Hard-negative cloud types are flagged.

    Returns a DataFrame with the unified column schema.
    """
    cas_path = Path(cas_dir)
    if not cas_path.is_dir():
        raise FileNotFoundError(f"CAS directory not found: {cas_dir}")

    rows: list[dict] = []
    for cloud_dir in sorted(cas_path.iterdir()):
        if not cloud_dir.is_dir():
            continue

        cloud_type = cloud_dir.name.lower()
        is_nlc = cloud_type == "noctilucent"
        label = 1 if is_nlc else 0
        is_hard_neg = cloud_type in HARD_NEGATIVE_TYPES

        # Walk recursively — images live in gallery_images/ & reference_images/
        for img_file in sorted(cloud_dir.rglob("*")):
            if not img_file.is_file():
                continue
            if img_file.suffix.lower() not in _IMG_EXTS:
                continue
            rows.append({
                "image_url": str(img_file.resolve()),
                "label": label,
                "source": "cas",
                "cloud_type": cloud_type,
                "is_hard_negative": is_hard_neg,
                "nlc_types": "",
                "nlc_type_indices": [],
                "observation_id": f"cas_{cloud_type}_{img_file.stem}",
                "latitude": None,
                "longitude": None,
            })

    if not rows:
        print(f"[CAS] No images found in {cas_path}")
        return pd.DataFrame(columns=[
            "image_url", "label", "source", "cloud_type", "is_hard_negative",
            "nlc_types", "nlc_type_indices", "observation_id", "latitude", "longitude",
        ])

    df = pd.DataFrame(rows)
    n_nlc = int((df["label"] == 1).sum())
    n_hard = int(df["is_hard_negative"].sum())
    n_easy = int(((df["label"] == 0) & ~df["is_hard_negative"]).sum())
    print(f"[CAS] Loaded {len(df)} images from {cas_path.name}/ "
          f"({n_nlc} NLC, {n_hard} hard-neg, {n_easy} easy-neg)")
    return df


def load_gallery_data(gallery_csv: str) -> pd.DataFrame:
    """Load Spaceweather Gallery CSV.

    All rows are NLC-positive.  The CSV already contains local image paths
    in the ``please upload an image …`` column.
    """
    df = pd.read_csv(gallery_csv)

    # Identify the local image path column (the one with absolute paths)
    img_col = None
    for col in df.columns:
        if "upload" in col.lower() or "image" in col.lower():
            # Pick the column whose first non-null value is a local path
            sample = df[col].dropna().iloc[0] if df[col].notna().any() else ""
            if str(sample).startswith("/"):
                img_col = col
                break
    if img_col is None:
        # Fallback: use the first column that looks like a path
        for col in df.columns:
            sample = df[col].dropna().iloc[0] if df[col].notna().any() else ""
            if str(sample).startswith("/"):
                img_col = col
                break

    if img_col is None:
        raise ValueError("Could not find a local image path column in gallery CSV")

    out = pd.DataFrame({
        "image_url": df[img_col].astype(str),
        "label": 1,
        "source": "gallery",
        "cloud_type": "noctilucent",
        "is_hard_negative": False,
        "nlc_types": df.get("types of nlc", ""),
        "nlc_type_indices": [[] for _ in range(len(df))],
        "observation_id": df.get("observationid", pd.Series(range(len(df)))).astype(str),
        "latitude": df.get("latitude", None),
        "longitude": df.get("longitude", None),
    })

    # Drop rows without a valid local image
    out = out[out["image_url"].apply(lambda p: os.path.isfile(p))].reset_index(drop=True)

    print(f"[Gallery] Loaded {len(out)} NLC-positive images from gallery CSV")
    return out


def load_scw_data(scw_csv: str, image_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load Space Cloud Watch CSV.

    Reuses the column conventions from the existing pipeline.
    """
    df = pd.read_csv(scw_csv)

    # Default image columns from the existing config
    if image_columns is None:
        image_columns = [
            "please upload an image of nlc (if available, optional)",
            "please upload an image of nlc (if available, optional, "
            "please do not upload images when you do not upload images "
            "when you do not see nlc.)",
            "upload an image of nlc (please do not upload images when "
            "you do not see nlc. non nlc images will be archived)",
        ]

    # Resolve image URL / path
    def _get_image(row):
        for col in image_columns:
            if col in row.index and pd.notna(row[col]):
                val = str(row[col]).strip()
                if val.startswith("http") or os.path.isfile(val):
                    return val
        return None

    label_col = "did you see nlc?"

    # Parse NLC type indices
    from .config import DataConfig
    type_labels = DataConfig().nlc_type_labels

    def _parse_types(type_str):
        if pd.isna(type_str) or not type_str:
            return []
        types = []
        for t in str(type_str).split(","):
            t = t.strip()
            if t in type_labels:
                types.append(type_labels[t])
        return types

    out = pd.DataFrame({
        "image_url": df.apply(_get_image, axis=1),
        "label": df[label_col].map({"Yes": 1, "No": 0}) if label_col in df.columns else 0,
        "source": "scw",
        "cloud_type": df.get("types of nlc", "").apply(
            lambda x: "noctilucent" if pd.notna(x) and str(x).strip() else ""
        ) if "types of nlc" in df.columns else "",
        "is_hard_negative": False,
        "nlc_types": df.get("types of nlc", ""),
        "nlc_type_indices": df.get("types of nlc", pd.Series(dtype=str)).apply(_parse_types),
        "observation_id": df.get("observationid", pd.Series(range(len(df)))).astype(str),
        "latitude": df.get("latitude", None),
        "longitude": df.get("longitude", None),
    })

    print(f"[SCW] Loaded {len(out)} rows ({(out['label'] == 1).sum()} NLC, "
          f"{(out['label'] == 0).sum()} No-NLC, "
          f"{out['image_url'].notna().sum()} with images)")
    return out


# ---------------------------------------------------------------------------
# Unified builder
# ---------------------------------------------------------------------------

def build_unified_dataset(
    cas_dir: Optional[str] = None,
    gallery_csv: Optional[str] = None,
    scw_csv: Optional[str] = None,
    image_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Merge all available data sources into a single DataFrame.

    Only sources whose path is provided (non-None, non-empty) will be loaded.
    Deduplicates by resolved image path.

    Returns
    -------
    pd.DataFrame
        Columns: image_url, label, source, cloud_type, is_hard_negative,
        nlc_types, nlc_type_indices, observation_id, latitude, longitude
    """
    frames: list[pd.DataFrame] = []

    if cas_dir:
        frames.append(load_cas_data(cas_dir))
    if gallery_csv:
        frames.append(load_gallery_data(gallery_csv))
    if scw_csv:
        frames.append(load_scw_data(scw_csv, image_columns))

    if not frames:
        raise ValueError("At least one data source must be provided")

    df = pd.concat(frames, ignore_index=True)

    # Deduplicate by resolved image path (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset=["image_url"], keep="first").reset_index(drop=True)
    if len(df) < before:
        print(f"[Unified] Removed {before - len(df)} duplicate image paths")

    # Ensure is_hard_negative defaults to False
    df["is_hard_negative"] = df["is_hard_negative"].fillna(False).astype(bool)

    # Summary
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    n_hard = df["is_hard_negative"].sum()
    n_img = df["image_url"].notna().sum()
    print(f"\n{'='*50}")
    print(f"Unified Dataset: {len(df)} total rows")
    print(f"  NLC positive:  {n_pos}")
    print(f"  Non-NLC:       {n_neg} ({n_hard} hard negatives)")
    print(f"  With images:   {n_img}")
    print(f"  Sources:       {df['source'].value_counts().to_dict()}")
    print(f"{'='*50}\n")

    return df
