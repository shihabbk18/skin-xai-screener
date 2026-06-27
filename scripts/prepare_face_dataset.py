"""
Prepare a face-only dermatology dataset from real metadata.

This script does not download or fabricate data. Put real dataset metadata and images
under data/raw/, then run this script to create train/val/test CSV splits.

Expected flexible columns:
- image path/id: image_path, filepath, path, file, image, image_id
- label: label, diagnosis, condition, dermatology_gradable_for_fitzpatrick_skin_type
- body/anatomy: body_site, anatomy, body_part, location
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


FACE_TERMS = {
    "face",
    "head",
    "forehead",
    "cheek",
    "nose",
    "chin",
    "perioral",
    "periocular",
    "eyelid",
    "lip",
}


def pick_column(columns: list[str], candidates: list[str]) -> str | None:
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--min-class-count", default=25, type=int)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    image_col = pick_column(list(df.columns), ["image_path", "filepath", "path", "file", "image", "image_id"])
    label_col = pick_column(list(df.columns), ["label", "diagnosis", "condition", "dermatology_gradable_for_fitzpatrick_skin_type"])
    site_col = pick_column(list(df.columns), ["body_site", "anatomy", "body_part", "location"])

    if not image_col or not label_col:
        raise SystemExit("Could not identify image and label columns. Rename columns or update candidates.")

    if site_col:
        mask = df[site_col].fillna("").astype(str).str.lower().apply(lambda value: any(term in value for term in FACE_TERMS))
        df = df[mask].copy()
    else:
        raise SystemExit("No anatomy/body site column found. Face-only filtering must be explicit, not guessed.")

    df = df[[image_col, label_col]].rename(columns={image_col: "image", label_col: "label"}).dropna()
    counts = df["label"].value_counts()
    keep = counts[counts >= args.min_class_count].index
    df = df[df["label"].isin(keep)].copy()

    train, temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.out_dir / "train.csv", index=False)
    val.to_csv(args.out_dir / "val.csv", index=False)
    test.to_csv(args.out_dir / "test.csv", index=False)
    print(f"Wrote {len(train)} train, {len(val)} val, {len(test)} test rows to {args.out_dir}")


if __name__ == "__main__":
    main()
