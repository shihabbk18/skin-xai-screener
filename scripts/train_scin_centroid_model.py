"""
Train a lightweight face-only skin screener from real SCIN data.

This script downloads public SCIN metadata/images from Google Cloud Storage,
filters to body_parts_head_or_neck == YES, keeps labels with enough examples,
extracts deterministic image features, and exports a JSON centroid model for
the browser app. It does not create or use fake data.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import random
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


BASE = "https://storage.googleapis.com/dx-scin-public-data"
CASES_URL = f"{BASE}/dataset/scin_cases.csv"
LABELS_URL = f"{BASE}/dataset/scin_labels.csv"


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read()


def top_label(value: str) -> str | None:
    if not isinstance(value, str) or value.strip() in {"", "{}"}:
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, dict) or not parsed:
        return None
    label, score = max(parsed.items(), key=lambda item: float(item[1]))
    if float(score) < 0.5:
        return None
    return str(label)


def feature_vector(image: Image.Image) -> list[float]:
    image = image.convert("RGB").resize((160, 160))
    arr = np.asarray(image).astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = arr.mean(axis=2)
    hsv_like_redness = np.clip(r - (g + b) / 2, 0, 1)
    brightness = gray
    texture = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES)).astype(np.float32) / 255.0

    feats: list[float] = []
    for channel in [r, g, b, gray, hsv_like_redness, texture]:
        feats.extend([float(channel.mean()), float(channel.std()), float(np.percentile(channel, 25)), float(np.percentile(channel, 75))])

    # Coarse spatial grid for face-region color/texture distribution.
    for channel in [hsv_like_redness, texture, brightness]:
        for y in range(4):
            for x in range(4):
                patch = channel[y * 40 : (y + 1) * 40, x * 40 : (x + 1) * 40]
                feats.append(float(patch.mean()))
    return feats


def main() -> None:
    random.seed(42)
    out_dir = Path("model")
    data_dir = Path("data/scin_face")
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases_path = data_dir / "scin_cases.csv"
    labels_path = data_dir / "scin_labels.csv"
    if not cases_path.exists():
        cases_path.write_bytes(download(CASES_URL))
    if not labels_path.exists():
        labels_path.write_bytes(download(LABELS_URL))

    cases = pd.read_csv(cases_path, dtype={"case_id": str})
    labels = pd.read_csv(labels_path, dtype={"case_id": str})
    df = cases.merge(labels, on="case_id")
    df = df[df["body_parts_head_or_neck"] == "YES"].copy()
    df["top_label"] = df["weighted_skin_condition_label"].apply(top_label)
    df = df.dropna(subset=["top_label"])

    counts = df["top_label"].value_counts()
    keep_labels = counts[counts >= 4].head(3).index.tolist()
    df = df[df["top_label"].isin(keep_labels)].copy()

    rows = []
    for _, row in df.iterrows():
        paths = [row.get("image_1_path"), row.get("image_2_path"), row.get("image_3_path")]
        for path in paths:
            if isinstance(path, str) and path:
                rows.append({"case_id": row["case_id"], "image_path": path, "label": row["top_label"]})

    random.shuffle(rows)
    max_per_label = 5
    used = {}
    selected = []
    for row in rows:
        used.setdefault(row["label"], 0)
        if used[row["label"]] >= max_per_label:
            continue
        used[row["label"]] += 1
        selected.append(row)

    features_by_label: dict[str, list[list[float]]] = {label: [] for label in keep_labels}
    manifest_rows = []
    print(f"Downloading/training on up to {max_per_label} real images for each of {len(keep_labels)} labels...", flush=True)
    for index, row in enumerate(selected, start=1):
        try:
            raw = download(f"{BASE}/{row['image_path']}")
            image = Image.open(BytesIO(raw))
            features = feature_vector(image)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {row['image_path']}: {exc}")
            continue
        features_by_label[row["label"]].append(features)
        manifest_rows.append(row)
        if index % 10 == 0:
            print(f"processed {index}/{len(selected)} images", flush=True)

    labels_final = [label for label, vectors in features_by_label.items() if len(vectors) >= 3]
    all_vectors = np.array([vector for label in labels_final for vector in features_by_label[label]], dtype=np.float32)
    mean = all_vectors.mean(axis=0)
    std = all_vectors.std(axis=0)
    std[std < 1e-6] = 1.0

    centroids = {}
    class_counts = {}
    for label in labels_final:
        vectors = np.array(features_by_label[label], dtype=np.float32)
        normalized = (vectors - mean) / std
        centroids[label] = normalized.mean(axis=0).round(6).tolist()
        class_counts[label] = int(len(vectors))

    model = {
        "modelType": "scin-face-centroid-v1",
        "source": "SCIN public dataset filtered to body_parts_head_or_neck == YES",
        "warning": "Research prototype. Not a medical device or diagnosis model.",
        "featureMean": mean.round(6).tolist(),
        "featureStd": std.round(6).tolist(),
        "labels": labels_final,
        "classCounts": class_counts,
        "centroids": centroids,
    }

    (out_dir / "face_model.json").write_text(json.dumps(model), encoding="utf-8")
    (out_dir / "labels.json").write_text(json.dumps(labels_final, indent=2), encoding="utf-8")
    with (data_dir / "face_training_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "image_path", "label"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("Trained labels:")
    for label in labels_final:
        print(f"- {label}: {class_counts[label]} images")
    print(f"Wrote {out_dir / 'face_model.json'}")


if __name__ == "__main__":
    main()
