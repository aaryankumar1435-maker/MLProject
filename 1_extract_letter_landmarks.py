"""Stage 1 (letters) - Landmark Extraction for static ISL alphabet images.

Converts the ISL fingerspelling image dataset into fixed-length hand
landmark vectors consumed by 2_train_letter_model.py.

Expected dataset layout (as downloaded from the RealSign ISL alphabet
dataset, CC0-1.0: https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset):
    data/letters_raw/Training/<letter>/*.jpg
    data/letters_raw/Validation/<letter>/*.jpg
    data/letters_raw/Testing/<letter>/*.jpg

The dataset's own Training/Validation/Testing split is used as-is (recorded
in manifest.csv's `split` column) instead of re-splitting randomly, so
Testing is a genuine held-out test set in 2_train_letter_model.py.

Incremental: if an image's .npy output already exists, it's reused instead
of re-running MediaPipe on it. This lets more raw images be dropped into
data/letters_raw/<split>/<letter>/ later (e.g. to grow the training set)
and re-run without redoing already-processed images.

Output (under --out-dir, default data/letters_processed):
    <split>/<letter>/<image_stem>.npy   float32 array, shape (126,)
    labels.json                          {letter: class_index}
    manifest.csv                         path,label,split
    diagnostics.csv                      per-image hand-detection count
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from utils.hand_utils import create_hands, detect, extract_hand_landmarks, hands_present

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("Training", "Validation", "Testing")


def find_images(raw_dir: Path):
    """Yield (image_path, letter_label, split) for images under
    raw_dir/<split>/<letter>/<image>.<ext>.
    """
    for split in SPLITS:
        split_dir = raw_dir / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path, path.parent.name, split


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand-landmark vectors from ISL letter images.")
    parser.add_argument("--raw-dir", default="data/letters_raw")
    parser.add_argument("--out-dir", default="data/letters_processed")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"Raw dataset directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    items = list(find_images(raw_dir))
    if not items:
        print(f"No images found under {raw_dir} (expected Training/Validation/Testing/<letter>/*.jpg)", file=sys.stderr)
        sys.exit(1)

    labels = sorted({letter for _, letter, _ in items})
    label_to_index = {letter: i for i, letter in enumerate(labels)}
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_to_index, f, indent=2, ensure_ascii=False)

    manifest_rows = []
    diagnostics_rows = []
    no_hand_count = 0
    cached = 0

    hands = create_hands(static_image_mode=True, min_detection_confidence=args.min_detection_confidence)

    print(f"Found {len(items)} images across {len(labels)} letters. Extracting...")
    try:
        for i, (image_path, letter, split) in enumerate(items, 1):
            class_dir = out_dir / split / letter
            npy_path = class_dir / f"{image_path.stem}.npy"

            if npy_path.exists():
                # Already extracted in a previous run (e.g. dataset grew) -
                # reuse it instead of re-running MediaPipe on unchanged
                # images. Diagnostics are recomputed cheaply from the saved
                # vector rather than skipped, so counts stay accurate.
                vector = np.load(npy_path)
                n_hands = int(np.any(vector[:len(vector) // 2])) + int(np.any(vector[len(vector) // 2:]))
                cached += 1
            else:
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"  [skip] {image_path} - could not read image")
                    continue

                results = detect(image, hands)
                vector = extract_hand_landmarks(results)
                n_hands = hands_present(results)

                class_dir.mkdir(parents=True, exist_ok=True)
                np.save(npy_path, vector)

            if n_hands == 0:
                no_hand_count += 1

            rel_path = npy_path.relative_to(out_dir).as_posix()
            manifest_rows.append([rel_path, letter, split])
            diagnostics_rows.append([rel_path, letter, split, n_hands])

            if i % 2000 == 0 or i == len(items):
                print(f"  processed {i}/{len(items)} ({cached} reused from a previous run)")
    finally:
        hands.close()

    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "split"])
        writer.writerows(manifest_rows)

    with open(out_dir / "diagnostics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "split", "hands_detected"])
        writer.writerows(diagnostics_rows)

    print(f"\nDone. Wrote {len(manifest_rows)} vectors to {out_dir}")
    if no_hand_count:
        print(f"[warn] {no_hand_count}/{len(manifest_rows)} images had no hand detected (see diagnostics.csv)")


if __name__ == "__main__":
    main()
