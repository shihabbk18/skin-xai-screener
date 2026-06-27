"""
Train a real facial skin model from processed CSV splits.

This is a starter training script. It requires real image files referenced by
data/processed/train.csv, val.csv, and test.csv. It does not create fake samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf


IMG_SIZE = (224, 224)


def make_dataset(csv_path: Path, image_root: Path, labels: list[str], batch_size: int, shuffle: bool) -> tf.data.Dataset:
    df = pd.read_csv(csv_path)
    label_to_id = {label: index for index, label in enumerate(labels)}
    paths = [str(image_root / path) for path in df["image"].astype(str)]
    y = [label_to_id[label] for label in df["label"]]

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=42)

    def load(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image, tf.one_hot(label, len(labels))

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--image-root", default=Path("data/raw/images"), type=Path)
    parser.add_argument("--out-dir", default=Path("model_export"), type=Path)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    args = parser.parse_args()

    train_df = pd.read_csv(args.data_dir / "train.csv")
    labels = sorted(train_df["label"].unique())
    if len(labels) < 2:
        raise SystemExit("Need at least two real classes to train.")

    train_ds = make_dataset(args.data_dir / "train.csv", args.image_root, labels, args.batch_size, True)
    val_ds = make_dataset(args.data_dir / "val.csv", args.image_root, labels, args.batch_size, False)

    base = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(labels), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.out_dir / "keras_model")
    (args.out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"Saved model and labels to {args.out_dir}")
    print("Export to TensorFlow.js with:")
    print(f"tensorflowjs_converter --input_format=tf_saved_model {args.out_dir / 'keras_model'} model")


if __name__ == "__main__":
    main()
