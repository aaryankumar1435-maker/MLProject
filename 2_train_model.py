"""Stage 2 - LSTM Training.

Loads the (SEQ_LEN, 258) landmark sequences produced by
1_extract_landmarks.py, trains a 2-layer LSTM classifier, and saves the
best model plus the inverse label map used by 3_realtime_recognition.py.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models

from utils.mp_utils import FEATURE_DIM, SEQ_LEN


def load_dataset(processed_dir: Path):
    manifest_path = processed_dir / "manifest.csv"
    labels_path = processed_dir / "labels.json"
    if not manifest_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Expected manifest.csv and labels.json in {processed_dir}. Run 1_extract_landmarks.py first."
        )

    with open(labels_path, "r", encoding="utf-8") as f:
        label_to_index = json.load(f)

    sequences, targets = [], []
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequences.append(np.load(processed_dir / row["path"]))
            targets.append(label_to_index[row["label"]])

    X = np.stack(sequences).astype(np.float32)
    y = np.array(targets, dtype=np.int64)
    return X, y, label_to_index


def build_model(seq_len, feature_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(seq_len, feature_dim)),
        layers.Masking(mask_value=0.0),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.30),
        layers.LSTM(64),
        layers.Dropout(0.30),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the LSTM sign classifier.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y, label_to_index = load_dataset(processed_dir)
    index_to_label = {v: k for k, v in label_to_index.items()}
    num_classes = len(label_to_index)

    print(f"Loaded {len(X)} sequences, shape {X.shape}, {num_classes} classes")
    class_counts = Counter(y)
    print("Class counts:", {index_to_label[i]: c for i, c in sorted(class_counts.items())})

    y_onehot = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_onehot, test_size=args.val_split, random_state=args.seed, stratify=y,
    )
    print(f"Train: {len(X_train)}  Validation: {len(X_val)}")

    model = build_model(SEQ_LEN, FEATURE_DIM, num_classes)
    model.summary()

    checkpoint_path = models_dir / "sign_model.keras"
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_categorical_accuracy", patience=args.patience,
            restore_best_weights=True, mode="max",
        ),
        callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_categorical_accuracy",
            save_best_only=True, mode="max",
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cbs,
    )

    model.save(checkpoint_path)

    with open(models_dir / "sign_model.labels.json", "w", encoding="utf-8") as f:
        json.dump(index_to_label, f, indent=2, ensure_ascii=False)

    val_pred = np.argmax(model.predict(X_val), axis=1)
    val_true = np.argmax(y_val, axis=1)
    target_names = [index_to_label[i] for i in range(num_classes)]

    print("\nValidation classification report:")
    print(classification_report(val_true, val_pred, target_names=target_names, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(val_true, val_pred))

    print(f"\nSaved model to {checkpoint_path}")
    print(f"Saved label map to {models_dir / 'sign_model.labels.json'}")


if __name__ == "__main__":
    main()
