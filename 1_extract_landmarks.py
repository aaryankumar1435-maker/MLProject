"""Stage 1 - Landmark Extraction.

Converts INCLUDE-dataset videos into fixed-length MediaPipe landmark
sequences consumed by 2_train_model.py.

Expected dataset layout:
    data/raw/<category>/<word>/<video>.mp4

Output (under --out-dir, default data/processed):
    <word>/<video_stem>.npy   float32 array, shape (SEQ_LEN, 258)
    labels.json               {word: class_index}
    manifest.csv              path,label  (path relative to --out-dir)
    diagnostics.csv           per-video landmark detection rates
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from utils.mp_utils import (
    SEQ_LEN,
    create_holistic,
    detect,
    extract_landmarks,
    landmarks_present,
    sample_uniform_indices,
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def find_videos(raw_dir: Path):
    """Yield (video_path, word_label) for every video under
    raw_dir/<category>/<word>/<video>.<ext>. The word label is the
    immediate parent folder name of the video file.
    """
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path, path.parent.name


def extract_video_landmarks(video_path: Path, holistic, seq_len: int):
    cap = cv2.VideoCapture(str(video_path))
    frame_features = []
    frame_presence = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = detect(frame, holistic)
        frame_features.append(extract_landmarks(results))
        frame_presence.append(landmarks_present(results))
    cap.release()

    if not frame_features:
        return None, None

    frame_features = np.stack(frame_features)
    indices = sample_uniform_indices(len(frame_features), seq_len)
    sequence = frame_features[indices].astype(np.float32)

    presence = np.array(frame_presence)
    detection_rates = presence.mean(axis=0)  # [pose, left_hand, right_hand]
    return sequence, detection_rates


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmark sequences from INCLUDE videos.")
    parser.add_argument("--raw-dir", default="data/raw", help="Root folder of INCLUDE videos (category/word/video)")
    parser.add_argument("--out-dir", default="data/processed", help="Where to write .npy sequences, labels.json, manifest.csv")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Frames per sequence (must match training/inference)")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"Raw dataset directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    videos = list(find_videos(raw_dir))
    if not videos:
        print(f"No videos found under {raw_dir} (expected category/word/video.mp4)", file=sys.stderr)
        sys.exit(1)

    labels = sorted({word for _, word in videos})
    label_to_index = {word: i for i, word in enumerate(labels)}
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_to_index, f, indent=2, ensure_ascii=False)

    manifest_rows = []
    diagnostics_rows = []

    holistic = create_holistic(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    print(f"Found {len(videos)} videos across {len(labels)} classes. Extracting...")
    try:
        for i, (video_path, word) in enumerate(videos, 1):
            sequence, detection_rates = extract_video_landmarks(video_path, holistic, args.seq_len)
            if sequence is None:
                print(f"  [skip] {video_path} - no frames could be read")
                continue

            class_dir = out_dir / word
            class_dir.mkdir(parents=True, exist_ok=True)
            npy_path = class_dir / f"{video_path.stem}.npy"
            np.save(npy_path, sequence)

            rel_path = npy_path.relative_to(out_dir).as_posix()
            manifest_rows.append([rel_path, word])
            diagnostics_rows.append([
                rel_path, word,
                f"{detection_rates[0]:.3f}", f"{detection_rates[1]:.3f}", f"{detection_rates[2]:.3f}",
            ])

            if detection_rates[0] < 0.5 or (detection_rates[1] < 0.1 and detection_rates[2] < 0.1):
                print(
                    f"  [warn] {video_path.name}: low landmark detection "
                    f"(pose={detection_rates[0]:.2f}, left_hand={detection_rates[1]:.2f}, "
                    f"right_hand={detection_rates[2]:.2f})"
                )

            if i % 25 == 0 or i == len(videos):
                print(f"  processed {i}/{len(videos)}")
    finally:
        holistic.close()

    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(manifest_rows)

    with open(out_dir / "diagnostics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "pose_detection_rate", "left_hand_detection_rate", "right_hand_detection_rate"])
        writer.writerows(diagnostics_rows)

    print(f"\nDone. Wrote {len(manifest_rows)} sequences to {out_dir}")
    print("labels.json, manifest.csv and diagnostics.csv written alongside the sequences.")


if __name__ == "__main__":
    main()
