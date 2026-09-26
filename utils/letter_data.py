"""Loading the extracted letter landmark vectors for training/evaluation.

Shared by 2_train_letter_model.py and 4_evaluate_letter_model.py so both
apply exactly the same filtering and normalization - evaluation numbers
only mean something if they match what training saw.
"""
from pathlib import Path

import numpy as np

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
        raw = np.load(Path(processed_dir) / row["path"])
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
