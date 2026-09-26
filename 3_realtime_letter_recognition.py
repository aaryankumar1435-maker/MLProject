"""Stage 3 (letters) - Real-Time Letter Recognition.

Runs the trained static hand-shape classifier on a live webcam feed. Unlike
the word pipeline there's no 40-frame buffer: each frame is classified
independently, since ISL letters are held poses rather than movements. A
short majority-vote window smooths out single-frame flicker before a
prediction is accepted/spoken, and speech runs on a background thread so
it never blocks the camera loop.
"""
import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from utils.custom_layers import AddPositionEmbedding, RandomLandmarkAugment  # noqa: F401 - registers
# custom layers (used by some trained architectures, e.g. mlp+augment) so
# tf.keras.models.load_model can deserialize them below.
from utils.hand_utils import create_hands, detect, draw_hand_landmarks, extract_hand_landmarks, normalize_landmarks
from utils.tts_utils import SpeechWorker


def main():
    parser = argparse.ArgumentParser(description="Real-time ISL letter recognition from webcam.")
    parser.add_argument("--model", default="models/letter_model.keras")
    parser.add_argument("--labels", default="models/letter_model.labels.json")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--smoothing-window", type=int, default=8, help="Frames of majority vote before accepting a prediction")
    parser.add_argument("--cooldown", type=float, default=1.5, help="Seconds between repeated spoken predictions")
    parser.add_argument("--no-speech", action="store_true", help="Disable text-to-speech output")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    args = parser.parse_args()

    model_path = Path(args.model)
    labels_path = Path(args.labels)
    if not model_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Model/labels not found ({model_path}, {labels_path}). Run 2_train_letter_model.py first."
        )

    model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        index_to_label = {int(k): v for k, v in json.load(f).items()}

    speaker = None if args.no_speech else SpeechWorker()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    hands = create_hands(static_image_mode=False, min_detection_confidence=args.min_detection_confidence)

    recent_preds = deque(maxlen=args.smoothing_window)
    last_spoken_letter = None
    last_spoken_time = 0.0

    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = detect(frame, hands)
            raw_vector = extract_hand_landmarks(results)
            draw_hand_landmarks(frame, results)

            # No hand detected at all: the model was trained without these
            # (see 2_train_letter_model.py), so skip prediction rather than
            # feed it a degenerate all-zero input.
            if np.any(raw_vector):
                vector = normalize_landmarks(raw_vector)
                # A direct call, not model.predict(): predict() sets up a batching
                # pipeline on every call, which measured ~20x slower (52.8 ms vs
                # 2.6 ms per frame) for identical outputs on a single frame.
                probs = model(np.expand_dims(vector, axis=0), training=False).numpy()[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                if confidence >= args.confidence_threshold:
                    recent_preds.append(pred_idx)
                else:
                    recent_preds.clear()
            else:
                confidence = 0.0
                recent_preds.clear()

            display_text = ""
            if len(recent_preds) == args.smoothing_window:
                stable_idx, count = Counter(recent_preds).most_common(1)[0]
                if count >= args.smoothing_window * 0.75:
                    letter = index_to_label[stable_idx]
                    display_text = f"{letter} ({confidence:.2f})"

                    now = time.time()
                    can_speak = (letter != last_spoken_letter) or (now - last_spoken_time >= args.cooldown)
                    if speaker is not None and can_speak:
                        speaker.say(letter)
                        last_spoken_letter = letter
                        last_spoken_time = now

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(frame, display_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("ISL Letter Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        if speaker is not None:
            speaker.stop()


if __name__ == "__main__":
    main()
