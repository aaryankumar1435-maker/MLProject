"""Stage 2 (letters) - Static hand-shape classifier.

Loads the 126-d hand-landmark vectors produced by
1_extract_letter_landmarks.py and trains a feed-forward classifier using
the dataset's own Training/Validation/Testing split.

This is deliberately not an LSTM: ISL letters are held hand shapes, not
movements, so there's no temporal pattern to model - unlike the word
pipeline (2_train_model.py), which stays LSTM-based for when word
recognition is revisited. See letter_recognition_architecture.docx for the
full rationale.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras import callbacks, layers, models

from utils.custom_layers import AddPositionEmbedding, RandomLandmarkAugment
from utils.hand_utils import FEATURE_DIM, normalize_landmarks


def load_split(processed_dir: Path, manifest_rows, label_to_index, split_name):
    """Loads a split, dropping samples where MediaPipe detected no hand at
    all (an all-zero raw vector). Those carry no real signal and otherwise
    collide on a single degenerate input, which during training biases the
    model toward whichever letter had the most detection failures (see
    letter_recognition_architecture.docx for the diagnosis: this collapsed
    onto the letter 'O' in the RealSign dataset, giving it near-100% recall
    but ~28% precision).
    """
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

    if not sequences:
        return np.empty((0, FEATURE_DIM), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X = np.stack(sequences).astype(np.float32)
    y = np.array(targets, dtype=np.int64)
    return X, y


def build_mlp(feature_dim, num_classes):
    """Baseline: flat Dense classifier over the 126-d vector, no notion
    that it's landmark points at all."""
    model = models.Sequential([
        layers.Input(shape=(feature_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def build_wide_mlp(feature_dim, num_classes):
    """Wider/deeper variant with batch normalization, to check whether the
    baseline MLP is capacity-limited rather than data-limited."""
    model = models.Sequential([
        layers.Input(shape=(feature_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.30),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.20),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def build_transformer(feature_dim, num_classes):
    """Self-attention classifier over the 42 hand landmarks.

    Where pointnet extracts per-landmark features independently and only
    combines them via a final max-pool, this lets every landmark attend to
    every other landmark directly (e.g. relating fingertip position to
    palm/wrist position), which is closer to how a hand shape is actually
    defined by the joint configuration of all landmarks together. Two
    pre-norm transformer-encoder blocks (self-attention + FFN, each with a
    residual connection) over 64-d per-landmark embeddings, then mean-pool
    to a single vector for classification.
    """
    num_points = feature_dim // 3
    inputs = layers.Input(shape=(feature_dim,))
    x = layers.Reshape((num_points, 3))(inputs)
    x = layers.Dense(64)(x)
    x = AddPositionEmbedding()(x)

    for _ in range(2):
        attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x = layers.LayerNormalization()(x + attn_out)
        ffn = layers.Dense(128, activation="relu")(x)
        ffn = layers.Dense(64)(ffn)
        x = layers.LayerNormalization()(x + ffn)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


def build_pointnet(feature_dim, num_classes):
    """Lightweight PointNet-style classifier: treats the 42 landmarks (21
    per hand) as a point cloud rather than an arbitrary flat vector.
    A per-point shared MLP (Conv1D with kernel_size=1 is a per-point Dense
    layer) extracts per-landmark features, then global max-pooling collapses
    them into one permutation-invariant descriptor before classification.

    This follows the point-cloud treatment of hand landmarks in Thomas et
    al., "An Open-Source American Sign Language Fingerspell Recognition and
    Semantic Pose Retrieval Interface" (arXiv:2408.09311) - see
    letter_recognition_architecture.docx Section 6.
    """
    num_points = feature_dim // 3  # 42 landmarks (21 left + 21 right), 3 coords each
    inputs = layers.Input(shape=(feature_dim,))
    x = layers.Reshape((num_points, 3))(inputs)
    x = layers.Conv1D(64, kernel_size=1, activation="relu")(x)
    x = layers.Conv1D(128, kernel_size=1, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


ARCHS = {
    "mlp": build_mlp,
    "wide_mlp": build_wide_mlp,
    "pointnet": build_pointnet,
    "transformer": build_transformer,
}


def build_model(arch, feature_dim, num_classes, augment=False):
    if augment:
        inputs = layers.Input(shape=(feature_dim,))
        x = RandomLandmarkAugment()(inputs)
        outputs = ARCHS[arch](feature_dim, num_classes)(x)
        model = models.Model(inputs, outputs)
    else:
        model = ARCHS[arch](feature_dim, num_classes)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the static-frame ISL letter classifier.")
    parser.add_argument("--processed-dir", default="data/letters_processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--arch", choices=list(ARCHS), default="mlp", help="Model architecture to train")
    parser.add_argument("--augment", action="store_true",
                         help="Apply random rotation/scale/translation jitter to training data only (see RandomLandmarkAugment)")
    parser.add_argument("--tag", default=None, help="Output filename suffix, e.g. letter_model_<tag>.keras (default: --arch[_aug])")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for weight init/shuffling/augmentation, for reproducible comparisons across archs")
    args = parser.parse_args()
    tag = args.tag or (args.arch + ("_aug" if args.augment else ""))
    tf.keras.utils.set_random_seed(args.seed)

    processed_dir = Path(args.processed_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = processed_dir / "manifest.csv"
    labels_path = processed_dir / "labels.json"
    if not manifest_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Expected manifest.csv and labels.json in {processed_dir}. Run 1_extract_letter_landmarks.py first."
        )

    with open(labels_path, "r", encoding="utf-8") as f:
        label_to_index = json.load(f)
    index_to_label = {v: k for k, v in label_to_index.items()}
    num_classes = len(label_to_index)

    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    X_train, y_train = load_split(processed_dir, manifest_rows, label_to_index, "Training")
    X_val, y_val = load_split(processed_dir, manifest_rows, label_to_index, "Validation")
    X_test, y_test = load_split(processed_dir, manifest_rows, label_to_index, "Testing")

    print(f"Train: {len(X_train)}  Validation: {len(X_val)}  Test: {len(X_test)}  Classes: {num_classes}")
    print("Train class counts:", {index_to_label[i]: c for i, c in sorted(Counter(y_train).items())})

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
    y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    model = build_model(args.arch, FEATURE_DIM, num_classes, augment=args.augment)
    model.summary()

    checkpoint_path = models_dir / f"letter_model_{tag}.keras"
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
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cbs,
    )

    model.save(checkpoint_path)
    labels_path_out = models_dir / f"letter_model_{tag}.labels.json"
    with open(labels_path_out, "w", encoding="utf-8") as f:
        json.dump(index_to_label, f, indent=2, ensure_ascii=False)

    test_pred = np.argmax(model.predict(X_test), axis=1)
    test_true = np.argmax(y_test_oh, axis=1)
    target_names = [index_to_label[i] for i in range(num_classes)]

    test_accuracy = float(np.mean(test_pred == test_true))
    macro_f1 = float(f1_score(test_true, test_pred, average="macro"))

    print("\nHeld-out test set classification report:")
    print(classification_report(test_true, test_pred, target_names=target_names, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(test_true, test_pred))

    print(f"\nSaved model to {checkpoint_path}")
    print(f"Saved label map to {labels_path_out}")
    print(f"\nRESULT arch={args.arch} tag={tag} augment={args.augment} test_accuracy={test_accuracy:.4f} macro_f1={macro_f1:.4f}")


if __name__ == "__main__":
    main()
