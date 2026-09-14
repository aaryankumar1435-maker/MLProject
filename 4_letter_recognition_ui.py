"""Stage 4 (letters) - Desktop UI for fingerspelling-to-text.

Wraps the same webcam -> MediaPipe Hands -> classifier pipeline as
3_realtime_letter_recognition.py in a Tkinter window that shows the live
feed alongside the recognized letter, the word currently being spelled,
autocomplete suggestions drawn from a static frequency-ranked word list
(utils/autocomplete.py), and the full typed sentence.

Letter commit model ("hold to type, release to re-arm"):
    Each frame's prediction only counts once a `--smoothing-window` run of
    frames agrees (>=75%) on the same letter at >=`--confidence-threshold`
    - this is the same majority-vote smoothing 3_realtime_letter_recognition.py
    uses to reject single-frame flicker. On top of that, this UI adds a
    "armed" latch: a stable letter is only appended to the current word
    once per hold, and appending again requires `--release-frames`
    consecutive not-stable frames first (hand moved away or changed shape).
    Without this latch, holding a letter shape for the ~1 second it takes
    to read the result would spam that letter dozens of times; with it,
    signing is "hold a shape -> it types once -> relax/move -> hold the
    next shape".

See LETTER_UI.md for the full user guide and design notes.
"""
import argparse
import json
import tkinter as tk
from collections import Counter, deque
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

from utils.autocomplete import WordCompleter
from utils.custom_layers import AddPositionEmbedding, RandomLandmarkAugment  # noqa: F401 - registers
# custom layers so tf.keras.models.load_model can deserialize architectures
# that use them (e.g. the canonical mlp+augment model).
from utils.hand_utils import create_hands, detect, draw_hand_landmarks, extract_hand_landmarks, normalize_landmarks
from utils.tts_utils import SpeechWorker

NUM_SUGGESTIONS = 5


class LetterRecognitionUI:
    def __init__(self, root, args):
        self.args = args
        self.root = root
        self.root.title("ISL Fingerspelling -> Text")

        model_path = Path(args.model)
        labels_path = Path(args.labels)
        if not model_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Model/labels not found ({model_path}, {labels_path}). Run 2_train_letter_model.py first."
            )
        self.model = tf.keras.models.load_model(model_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            self.index_to_label = {int(k): v for k, v in json.load(f).items()}

        word_list_path = Path(args.word_list)
        if not word_list_path.exists():
            raise FileNotFoundError(f"Word list not found: {word_list_path}")
        self.completer = WordCompleter(word_list_path)

        self.cap = cv2.VideoCapture(args.camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {args.camera}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.video_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.video_height)
        self.hands = create_hands(static_image_mode=False, min_detection_confidence=args.min_detection_confidence)

        self.speaker = SpeechWorker() if not args.no_speech else None

        # Recognition state
        self.recent_preds = deque(maxlen=args.smoothing_window)
        self.unstable_count = args.release_frames  # start "armed"
        self.armed = True
        self.last_letter = ""
        self.last_confidence = 0.0
        self.hand_present = False

        # Text-building state
        self.current_word = ""
        self.sentence = ""
        self.current_suggestions = []

        self._build_layout()
        self._refresh_suggestions()
        self._update_frame()

    # ---- UI construction ----------------------------------------------

    def _build_layout(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(main)
        self.video_label.grid(row=0, column=0, rowspan=8, padx=(0, 12))

        status_font = ("Segoe UI", 12)
        big_font = ("Segoe UI", 28, "bold")

        self.letter_var = tk.StringVar(value="-")
        ttk.Label(main, text="Detected letter", font=status_font).grid(row=0, column=1, sticky="w")
        ttk.Label(main, textvariable=self.letter_var, font=big_font).grid(row=1, column=1, sticky="w")

        self.hand_status_var = tk.StringVar(value="No hand detected")
        ttk.Label(main, textvariable=self.hand_status_var, font=status_font).grid(row=2, column=1, sticky="w", pady=(0, 8))

        self.word_var = tk.StringVar(value="(spell a word by holding letter shapes)")
        ttk.Label(main, text="Spelling", font=status_font).grid(row=3, column=1, sticky="w")
        ttk.Label(main, textvariable=self.word_var, font=("Segoe UI", 18, "bold"), foreground="#1a5fb4").grid(
            row=4, column=1, sticky="w", pady=(0, 8)
        )

        ttk.Label(main, text="Autocomplete (click, or press 1-5)", font=status_font).grid(row=5, column=1, sticky="w")
        suggestions_frame = ttk.Frame(main)
        suggestions_frame.grid(row=6, column=1, sticky="w", pady=(0, 8))
        self.suggestion_buttons = []
        for i in range(NUM_SUGGESTIONS):
            btn = ttk.Button(suggestions_frame, text="", width=12,
                              command=lambda i=i: self._accept_suggestion(i))
            btn.grid(row=0, column=i, padx=2)
            self.suggestion_buttons.append(btn)

        ttk.Label(main, text="Sentence", font=status_font).grid(row=7, column=1, sticky="w")
        text_frame = ttk.Frame(main)
        text_frame.grid(row=8, column=1, sticky="nsew", pady=(0, 8))
        self.sentence_text = tk.Text(text_frame, width=40, height=6, wrap="word", font=("Segoe UI", 13))
        self.sentence_text.grid(row=0, column=0)
        self.sentence_text.configure(state="disabled")

        controls = ttk.Frame(main)
        controls.grid(row=9, column=1, sticky="w")
        ttk.Button(controls, text="Space", command=self._finalize_word).grid(row=0, column=0, padx=2)
        ttk.Button(controls, text="Backspace", command=self._backspace).grid(row=0, column=1, padx=2)
        ttk.Button(controls, text="Clear word", command=self._clear_word).grid(row=0, column=2, padx=2)
        ttk.Button(controls, text="Clear all", command=self._clear_all).grid(row=0, column=3, padx=2)
        ttk.Button(controls, text="Speak sentence", command=self._speak_sentence).grid(row=0, column=4, padx=2)

        speech_controls = ttk.Frame(main)
        speech_controls.grid(row=10, column=1, sticky="w", pady=(4, 0))
        self.speak_letters_var = tk.BooleanVar(value=False)
        self.speak_words_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(speech_controls, text="Speak each letter", variable=self.speak_letters_var).grid(row=0, column=0, padx=(0, 8))
        ttk.Checkbutton(speech_controls, text="Speak each word", variable=self.speak_words_var).grid(row=0, column=1)
        if self.speaker is None:
            for child in speech_controls.winfo_children():
                child.state(["disabled"])

        ttk.Label(
            main,
            text="Hold a letter shape steadily to type it; move your hand between letters.\n"
                 "Space/Backspace/Clear also work as keyboard keys; 1-5 pick a suggestion.",
            font=("Segoe UI", 9), foreground="#666666",
        ).grid(row=11, column=1, sticky="w", pady=(8, 0))

        self.root.bind("<space>", lambda e: self._finalize_word())
        self.root.bind("<BackSpace>", lambda e: self._backspace())
        self.root.bind("<Escape>", lambda e: self._clear_word())
        for i in range(NUM_SUGGESTIONS):
            self.root.bind(str(i + 1), lambda e, i=i: self._accept_suggestion(i))

    # ---- Recognition loop ----------------------------------------------

    def _update_frame(self):
        ok, frame = self.cap.read()
        if ok:
            results = detect(frame, self.hands)
            raw_vector = extract_hand_landmarks(results)
            draw_hand_landmarks(frame, results)

            self.hand_present = bool(np.any(raw_vector))
            stable_letter = None
            if self.hand_present:
                vector = normalize_landmarks(raw_vector)
                probs = self.model.predict(np.expand_dims(vector, axis=0), verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                self.last_confidence = confidence
                if confidence >= self.args.confidence_threshold:
                    self.recent_preds.append(pred_idx)
                else:
                    self.recent_preds.clear()
            else:
                self.last_confidence = 0.0
                self.recent_preds.clear()

            if len(self.recent_preds) == self.args.smoothing_window:
                stable_idx, count = Counter(self.recent_preds).most_common(1)[0]
                if count >= self.args.smoothing_window * 0.75:
                    stable_letter = self.index_to_label[stable_idx]

            if stable_letter is not None:
                self.unstable_count = 0
                self.last_letter = stable_letter
                if self.armed:
                    self._commit_letter(stable_letter)
                    self.armed = False
            else:
                self.unstable_count += 1
                if self.unstable_count >= self.args.release_frames:
                    self.armed = True

            self._render_video(frame)
            self._render_status()

        self.root.after(15, self._update_frame)

    def _render_video(self, frame_bgr):
        # Some webcams ignore the requested capture resolution, so clamp the
        # *displayed* size too - otherwise a high-res frame can render wider
        # than the screen and push the sidebar off-screen.
        h, w = frame_bgr.shape[:2]
        if w > self.args.video_width:
            scale = self.args.video_width / w
            frame_bgr = cv2.resize(frame_bgr, (self.args.video_width, int(h * scale)))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # keep a reference, else Tk garbage-collects it

    def _render_status(self):
        if self.hand_present:
            self.letter_var.set(f"{self.last_letter or '?'}  ({self.last_confidence:.2f})")
            ready = "ready to type next" if self.armed else "hold released to type next"
            self.hand_status_var.set(f"Hand detected - {ready}")
        else:
            self.letter_var.set("-")
            self.hand_status_var.set("No hand detected")

    # ---- Text building ---------------------------------------------------

    def _commit_letter(self, letter):
        self.current_word += letter
        self.word_var.set(self.current_word + "_")
        self._refresh_suggestions()
        if self.speaker is not None and self.speak_letters_var.get():
            self.speaker.say(letter)

    def _refresh_suggestions(self):
        self.current_suggestions = self.completer.suggest(self.current_word, limit=NUM_SUGGESTIONS) if self.current_word else []
        for i, btn in enumerate(self.suggestion_buttons):
            if i < len(self.current_suggestions):
                btn.configure(text=self.current_suggestions[i], state="normal")
            else:
                btn.configure(text="", state="disabled")

    def _accept_suggestion(self, index):
        if index >= len(self.current_suggestions):
            return
        self._finalize_word(self.current_suggestions[index])

    def _finalize_word(self, word=None):
        word = word if word is not None else self.current_word
        if not word:
            return
        self.sentence += word + " "
        self.current_word = ""
        self.word_var.set("(spell a word by holding letter shapes)")
        self._refresh_suggestions()
        self._render_sentence()
        if self.speaker is not None and self.speak_words_var.get():
            self.speaker.say(word)

    def _backspace(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
            self.word_var.set((self.current_word + "_") if self.current_word else "(spell a word by holding letter shapes)")
            self._refresh_suggestions()
        elif self.sentence.strip():
            words = self.sentence.strip().split(" ")
            self.current_word = words[-1]
            self.sentence = (" ".join(words[:-1]) + " ") if len(words) > 1 else ""
            self.word_var.set(self.current_word + "_")
            self._refresh_suggestions()
            self._render_sentence()

    def _clear_word(self):
        self.current_word = ""
        self.word_var.set("(spell a word by holding letter shapes)")
        self._refresh_suggestions()

    def _clear_all(self):
        self.current_word = ""
        self.sentence = ""
        self.word_var.set("(spell a word by holding letter shapes)")
        self._refresh_suggestions()
        self._render_sentence()

    def _speak_sentence(self):
        text = self.sentence.strip()
        if text and self.speaker is not None:
            self.speaker.say(text)

    def _render_sentence(self):
        self.sentence_text.configure(state="normal")
        self.sentence_text.delete("1.0", "end")
        self.sentence_text.insert("1.0", self.sentence)
        self.sentence_text.configure(state="disabled")

    # ---- Teardown -----------------------------------------------------

    def _on_close(self):
        self.hands.close()
        self.cap.release()
        if self.speaker is not None:
            self.speaker.stop()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Desktop UI: real-time ISL fingerspelling with word autocomplete.")
    parser.add_argument("--model", default="models/letter_model.keras")
    parser.add_argument("--labels", default="models/letter_model.labels.json")
    parser.add_argument("--word-list", default="data/word_list.txt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video-width", type=int, default=560, help="Requested capture width and max displayed width")
    parser.add_argument("--video-height", type=int, default=420, help="Requested capture height")
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--smoothing-window", type=int, default=8, help="Frames of majority vote before accepting a prediction")
    parser.add_argument("--release-frames", type=int, default=6,
                         help="Consecutive not-stable frames required before the same/next letter can be typed again")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--no-speech", action="store_true", help="Disable text-to-speech entirely")
    args = parser.parse_args()

    root = tk.Tk()
    LetterRecognitionUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
