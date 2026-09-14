"""Stage 4 (letters) - Confusion matrix and per-letter accuracy for the
canonical letter model, on the held-out Testing split.

Reuses the same load_split logic as 2_train_letter_model.py (zero-vector
filtering, per-hand landmark normalization) so the evaluated accuracy
matches what training reported, then renders a confusion-matrix heatmap.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

import utils.custom_layers  # noqa: F401 - registers RandomLandmarkAugment/AddPositionEmbedding for model loading
from utils.hand_utils import normalize_landmarks


def load_split(processed_dir: Path, manifest_rows, label_to_index, split_name):
    sequences, targets = [], []
    skipped = 0
    for row in manifest_rows:
        if row["split"] != split_name:
            continue
        raw = np.load(processed_dir / row["path"])
        if not np.any(raw):
            skipped += 1
            continue
        sequences.append(normalize_landmarks(raw))
        targets.append(label_to_index[row["label"]])

    if skipped:
        print(f"  [{split_name}] skipped {skipped} samples with no hand detected")

    X = np.stack(sequences).astype(np.float32)
    y = np.array(targets, dtype=np.int64)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Evaluate the letter model and plot a confusion matrix.")
    parser.add_argument("--model", default="models/letter_model.keras")
    parser.add_argument("--labels", default="models/letter_model.labels.json")
    parser.add_argument("--processed-dir", default="data/letters_processed")
    parser.add_argument("--split", default="Testing", choices=["Training", "Validation", "Testing"])
    parser.add_argument("--out", default="models/letter_model_confusion_matrix.png")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    with open(args.labels, "r", encoding="utf-8") as f:
        index_to_label = {int(k): v for k, v in json.load(f).items()}
    label_to_index = {v: k for k, v in index_to_label.items()}
    num_classes = len(index_to_label)
    target_names = [index_to_label[i] for i in range(num_classes)]

    with open(processed_dir / "manifest.csv", "r", newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    X, y_true = load_split(processed_dir, manifest_rows, label_to_index, args.split)
    print(f"Evaluating on {len(X)} {args.split} samples across {num_classes} letters")

    model = tf.keras.models.load_model(args.model)
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)

    accuracy = float(np.mean(y_pred == y_true))
    print(f"\nOverall accuracy: {accuracy:.4f}")
    print("\nPer-letter precision/recall/F1:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(12, 11))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=target_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f", xticks_rotation="vertical")
    ax.set_title(f"ISL letter model - {args.split} set confusion matrix (row-normalized)\nOverall accuracy: {accuracy:.2%}")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved confusion matrix heatmap to {out_path}")

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + target_names)
        for i, row in enumerate(cm):
            writer.writerow([target_names[i]] + list(row))
    print(f"Saved raw confusion matrix counts to {csv_path}")


if __name__ == "__main__":
    main()
