"""Stage 3 - Real-Time Recognition.

Runs the trained LSTM sign classifier on a live webcam feed using a
sliding 40-frame window of MediaPipe landmarks, and optionally speaks the
predicted word through a non-blocking TTS worker thread so speech synthesis
never stalls the camera loop.
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from utils.mp_utils import SEQ_LEN, create_holistic, detect, draw_landmarks, extract_landmarks
from utils.tts_utils import SpeechWorker


def main():
    parser = argparse.ArgumentParser(description="Real-time ISL recognition from webcam.")
    parser.add_argument("--model", default="models/sign_model.keras")
    parser.add_argument("--labels", default="models/sign_model.labels.json")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--cooldown", type=float, default=2.0, help="Seconds between repeated spoken predictions")
    parser.add_argument("--no-speech", action="store_true", help="Disable text-to-speech output")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    args = parser.parse_args()

    model_path = Path(args.model)
    labels_path = Path(args.labels)
    if not model_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Model/labels not found ({model_path}, {labels_path}). Run 2_train_model.py first."
        )

    model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        index_to_label = {int(k): v for k, v in json.load(f).items()}

    speaker = None if args.no_speech else SpeechWorker()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    buffer = deque(maxlen=SEQ_LEN)
    last_spoken_word = None
    last_spoken_time = 0.0

    holistic = create_holistic(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = detect(frame, holistic)
            buffer.append(extract_landmarks(results))
            draw_landmarks(frame, results)

            display_text = ""
            if len(buffer) == SEQ_LEN:
                sequence = np.expand_dims(np.stack(buffer), axis=0)
                probs = model.predict(sequence, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])

                if confidence >= args.confidence_threshold:
                    word = index_to_label[pred_idx]
                    display_text = f"{word} ({confidence:.2f})"

                    now = time.time()
                    can_speak = (word != last_spoken_word) or (now - last_spoken_time >= args.cooldown)
                    if speaker is not None and can_speak:
                        speaker.say(word)
                        last_spoken_word = word
                        last_spoken_time = now

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(frame, display_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("ISL Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        holistic.close()
        cap.release()
        cv2.destroyAllWindows()
        if speaker is not None:
            speaker.stop()


if __name__ == "__main__":
    main()
