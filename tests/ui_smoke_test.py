"""End-to-end check of the fingerspelling UI without a webcam.

Replaces the camera with real Testing-split photos: each letter of WORD is
"held" for 20 frames, followed by 12 blank frames (no hand, i.e. the signer
relaxes). If the hold-to-type / release-to-re-arm logic works, the UI types
exactly WORD - once per letter, including doubled letters like the LL in
HELLO.

Run from the project root (a window opens for a few seconds):
    .venv\\Scripts\\python tests\\ui_smoke_test.py HELLO
    .venv\\Scripts\\python tests\\ui_smoke_test.py HELP --screenshot ui.png

Needs models/letter_model.keras, data/letters_processed/ and
data/letters_raw/Testing/ (the photos themselves).
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

import utils.custom_layers  # noqa: E402,F401 - registers custom layers for loading
from utils.hand_utils import normalize_landmarks  # noqa: E402

HOLD_FRAMES, GAP_FRAMES = 20, 12


def pick_image(letter, model, index_to_label, rows):
    """A Testing photo of `letter` the model gets right with >97% confidence,
    so a failure means the UI logic is wrong, not the classifier."""
    for r in rows:
        if r["split"] != "Testing" or r["label"] != letter:
            continue
        raw = np.load(Path("data/letters_processed") / r["path"])
        if not np.any(raw):
            continue
        probs = model(normalize_landmarks(raw)[None], training=False).numpy()[0]
        if index_to_label[int(probs.argmax())] != letter or probs.max() < 0.97:
            continue
        stem = Path(r["path"]).stem
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            img_path = Path("data/letters_raw/Testing") / letter / f"{stem}{ext}"
            if img_path.exists():
                return cv2.imread(str(img_path))
    raise SystemExit(f"No confidently-classified Testing photo found for letter {letter}")


def main():
    parser = argparse.ArgumentParser(description="Fingerspelling UI smoke test with a fake camera.")
    parser.add_argument("word", nargs="?", default="HELLO")
    parser.add_argument("--screenshot", default=None, help="Save a PNG of the window at the end (Windows)")
    args = parser.parse_args()
    word = args.word.upper()

    if args.screenshot and sys.platform == "win32":
        import ctypes
        try:  # make window coordinates match screen pixels under display scaling
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass

    model = tf.keras.models.load_model("models/letter_model.keras")
    index_to_label = {int(k): v for k, v in json.loads(Path("models/letter_model.labels.json").read_text()).items()}
    with open("data/letters_processed/manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    frames = []
    for letter in word:
        image = pick_image(letter, model, index_to_label, rows)
        frames += [image] * HOLD_FRAMES
        frames += [np.full_like(image, 255)] * GAP_FRAMES

    class FakeCapture:
        def __init__(self, *_):
            self.i = 0

        def isOpened(self):
            return True

        def set(self, *_):
            return True

        def read(self):
            if self.i >= len(frames):
                return False, None
            self.i += 1
            return True, frames[self.i - 1].copy()

        def release(self):
            pass

    spec = importlib.util.spec_from_file_location("letter_ui", "4_letter_recognition_ui.py")
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    ui.cv2.VideoCapture = FakeCapture

    ui_args = argparse.Namespace(
        model="models/letter_model.keras", labels="models/letter_model.labels.json",
        word_list="data/word_list.txt", camera=0, video_width=560, video_height=420,
        confidence_threshold=0.80, smoothing_window=8, release_frames=6,
        min_detection_confidence=0.5, no_speech=True,
    )
    root = ui.tk.Tk()
    app = ui.LetterRecognitionUI(root, ui_args)
    started = time.perf_counter()
    result = {}

    def finish():
        if app.cap.i < len(frames):
            root.after(50, finish)
            return
        result["typed"] = app.current_word
        result["suggestions"] = list(app.current_suggestions)
        result["seconds"] = time.perf_counter() - started
        if args.screenshot:
            root.update()
            time.sleep(0.3)
            from PIL import ImageGrab
            x, y = root.winfo_rootx(), root.winfo_rooty()
            ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height()), all_screens=True).save(args.screenshot)
        app._on_close()

    root.after(100, finish)
    root.mainloop()

    print(f"expected : {word}")
    print(f"typed    : {result['typed']}")
    print(f"suggest  : {result['suggestions']}")
    print(f"speed    : {len(frames) / result['seconds']:.1f} frames/second")
    ok = result["typed"] == word
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
