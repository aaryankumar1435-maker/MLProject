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

# ---- Design tokens -------------------------------------------------------
BG_MAIN = "#eef2f7"
HEADER_BG = "#111827"
HEADER_FG = "#f9fafb"
HEADER_SUB = "#9ca3af"
CARD_BG = "#ffffff"
BORDER = "#e2e8f0"
TEXT_PRIMARY = "#0f172a"
TEXT_SECONDARY = "#64748b"
ACCENT = "#2563eb"
ACCENT_DARK = "#1d4ed8"
ACCENT_LIGHT = "#dbeafe"
SUCCESS = "#16a34a"
SUCCESS_LIGHT = "#dcfce7"
WARNING = "#d97706"
MUTED_BTN_BG = "#f1f5f9"
MUTED_BTN_BG_HOVER = "#e2e8f0"
MUTED_BTN_FG = "#334155"
IDLE_DOT = "#cbd5e1"

FONT_TITLE = ("Segoe UI", 17, "bold")
FONT_SUBTITLE = ("Segoe UI", 10)
FONT_SECTION = ("Segoe UI", 10, "bold")
FONT_BIG_LETTER = ("Consolas", 40, "bold")
FONT_WORD = ("Segoe UI", 22, "bold")
FONT_STATUS = ("Segoe UI", 10)
FONT_HINT = ("Segoe UI", 9)


class LetterRecognitionUI:
    def __init__(self, root, args):
        self.args = args
        self.root = root
        self.root.title("ISL Fingerspelling -> Text")
        self.root.configure(bg=BG_MAIN)
        self.root.minsize(980, 640)

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

        self._setup_styles()
        self._build_layout()
        self._refresh_suggestions()
        self._render_sentence()
        self._update_frame()

    # ---- UI construction ----------------------------------------------

    def _setup_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor=BORDER, background=ACCENT, bordercolor=BORDER,
            lightcolor=ACCENT, darkcolor=ACCENT, thickness=8,
        )
        style.configure("Card.TCheckbutton", background=CARD_BG, foreground=TEXT_PRIMARY, font=FONT_STATUS)
        style.map("Card.TCheckbutton", background=[("active", CARD_BG)])

    def _card(self, parent, title=None, **pack_opts):
        """A bordered white panel with an optional uppercase section title."""
        outer = tk.Frame(parent, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER)
        outer.pack(fill="x", pady=(0, 12), **pack_opts)
        body = tk.Frame(outer, bg=CARD_BG, padx=18, pady=16)
        body.pack(fill="both", expand=True)
        if title:
            tk.Label(body, text=title.upper(), font=FONT_SECTION, bg=CARD_BG, fg=TEXT_SECONDARY).pack(
                anchor="w", pady=(0, 12)
            )
        return body

    def _flat_button(self, parent, text, command, bg=MUTED_BTN_BG, fg=MUTED_BTN_FG,
                      hover_bg=MUTED_BTN_BG_HOVER, font=("Segoe UI", 10), **kwargs):
        btn = tk.Button(
            parent, text=text, command=command, bg=bg, fg=fg, activebackground=hover_bg,
            activeforeground=fg, relief="flat", bd=0, font=font, cursor="hand2",
            padx=kwargs.pop("padx", 12), pady=kwargs.pop("pady", 7), **kwargs,
        )
        btn.bind("<Enter>", lambda e: btn.configure(bg=hover_bg) if str(btn["state"]) == "normal" else None)
        btn.bind("<Leave>", lambda e: btn.configure(bg=bg) if str(btn["state"]) == "normal" else None)
        return btn

    def _build_layout(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # ---- Header -----------------------------------------------------
        header = tk.Frame(self.root, bg=HEADER_BG)
        header.grid(row=0, column=0, sticky="ew")
        header_inner = tk.Frame(header, bg=HEADER_BG, padx=20, pady=14)
        header_inner.pack(fill="x")
        tk.Label(header_inner, text="ISL Fingerspelling → Text", font=FONT_TITLE,
                 bg=HEADER_BG, fg=HEADER_FG).pack(anchor="w")
        tk.Label(header_inner, text="Hold a letter shape steadily to type it · relax your hand between letters",
                 font=FONT_SUBTITLE, bg=HEADER_BG, fg=HEADER_SUB).pack(anchor="w", pady=(2, 0))

        # ---- Body ---------------------------------------------------------
        body = tk.Frame(self.root, bg=BG_MAIN, padx=20, pady=16)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # -- Camera card
        video_card = tk.Frame(body, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER)
        video_card.grid(row=0, column=0, sticky="n", padx=(0, 16))
        video_inner = tk.Frame(video_card, bg=CARD_BG, padx=12, pady=12)
        video_inner.pack()
        tk.Label(video_inner, text="LIVE CAMERA", font=FONT_SECTION, bg=CARD_BG, fg=TEXT_SECONDARY).pack(
            anchor="w", pady=(0, 8)
        )
        self.video_label = tk.Label(video_inner, bg="#000000")
        self.video_label.pack()

        # -- Sidebar
        sidebar = tk.Frame(body, bg=BG_MAIN)
        sidebar.grid(row=0, column=1, sticky="nsew")

        # Recognition card
        rec = self._card(sidebar, "Live Recognition")
        status_row = tk.Frame(rec, bg=CARD_BG)
        status_row.pack(fill="x", pady=(0, 14))
        self.status_dot = tk.Canvas(status_row, width=12, height=12, bg=CARD_BG, highlightthickness=0)
        self._status_dot_id = self.status_dot.create_oval(1, 1, 11, 11, fill=IDLE_DOT, outline="")
        self.status_dot.pack(side="left", padx=(0, 8))
        self.hand_status_var = tk.StringVar(value="No hand detected")
        tk.Label(status_row, textvariable=self.hand_status_var, font=FONT_STATUS, bg=CARD_BG,
                 fg=TEXT_PRIMARY).pack(side="left")

        hero = tk.Frame(rec, bg=CARD_BG)
        hero.pack(fill="x")
        self.letter_box = tk.Frame(hero, bg=ACCENT_LIGHT, width=84, height=84, highlightthickness=0)
        self.letter_box.pack(side="left", padx=(0, 16))
        self.letter_box.pack_propagate(False)
        self.letter_var = tk.StringVar(value="–")
        self.letter_label = tk.Label(self.letter_box, textvariable=self.letter_var, font=FONT_BIG_LETTER,
                                      bg=ACCENT_LIGHT, fg=ACCENT_DARK)
        self.letter_label.place(relx=0.5, rely=0.5, anchor="center")

        conf_col = tk.Frame(hero, bg=CARD_BG)
        conf_col.pack(side="left", fill="x", expand=True, anchor="s", pady=(0, 4))
        tk.Label(conf_col, text="Confidence", font=FONT_HINT, bg=CARD_BG, fg=TEXT_SECONDARY).pack(anchor="w")
        conf_row = tk.Frame(conf_col, bg=CARD_BG)
        conf_row.pack(fill="x", pady=(4, 0))
        self.confidence_var = tk.DoubleVar(value=0.0)
        self.confidence_bar = ttk.Progressbar(
            conf_row, style="Accent.Horizontal.TProgressbar", orient="horizontal",
            mode="determinate", maximum=100, variable=self.confidence_var, length=140,
        )
        self.confidence_bar.pack(side="left", fill="x", expand=True)
        self.confidence_label_var = tk.StringVar(value="0%")
        tk.Label(conf_row, textvariable=self.confidence_label_var, font=FONT_HINT, bg=CARD_BG,
                 fg=TEXT_SECONDARY, width=5, anchor="e").pack(side="left", padx=(8, 0))

        # Spelling card
        spell = self._card(sidebar, "Spelling")
        self.word_var = tk.StringVar(value="Hold letter shapes to spell a word")
        self.word_label = tk.Label(spell, textvariable=self.word_var, font=FONT_WORD, bg=CARD_BG,
                                    fg=ACCENT_DARK, anchor="w")
        self.word_label.pack(fill="x", pady=(0, 12))

        tk.Label(spell, text="Autocomplete · click or press 1-5", font=FONT_HINT, bg=CARD_BG,
                 fg=TEXT_SECONDARY).pack(anchor="w", pady=(0, 6))
        suggestions_frame = tk.Frame(spell, bg=CARD_BG)
        suggestions_frame.pack(fill="x")
        self.suggestion_buttons = []
        for i in range(NUM_SUGGESTIONS):
            btn = self._flat_button(
                suggestions_frame, text="", command=lambda i=i: self._accept_suggestion(i),
                bg=ACCENT_LIGHT, fg=ACCENT_DARK, hover_bg=ACCENT, font=("Segoe UI", 10, "bold"),
                padx=10, pady=8,
            )
            btn.grid(row=0, column=i, padx=(0, 6) if i < NUM_SUGGESTIONS - 1 else 0, sticky="ew")
            suggestions_frame.columnconfigure(i, weight=1)
            self.suggestion_buttons.append(btn)

        # Sentence card
        sent = self._card(sidebar, "Sentence")
        text_wrap = tk.Frame(sent, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER)
        text_wrap.pack(fill="x", pady=(0, 12))
        self.sentence_text = tk.Text(text_wrap, height=3, wrap="word", font=("Segoe UI", 12),
                                      relief="flat", bd=0, padx=10, pady=8, bg="#fbfcfe", fg=TEXT_PRIMARY)
        self.sentence_text.pack(fill="both", expand=True)
        self.sentence_text.tag_configure("placeholder", foreground=TEXT_SECONDARY, font=("Segoe UI", 12, "italic"))
        self.sentence_text.configure(state="disabled")

        controls = tk.Frame(sent, bg=CARD_BG)
        controls.pack(fill="x", pady=(0, 12))
        self._flat_button(controls, "Space", self._finalize_word).pack(side="left", padx=(0, 6))
        self._flat_button(controls, "⌫ Backspace", self._backspace).pack(side="left", padx=(0, 6))
        self._flat_button(controls, "Clear Word", self._clear_word).pack(side="left", padx=(0, 6))
        self._flat_button(controls, "Clear All", self._clear_all).pack(side="left", padx=(0, 6))
        self._flat_button(
            controls, "\U0001F50A Speak", self._speak_sentence,
            bg=ACCENT, fg="#ffffff", hover_bg=ACCENT_DARK,
        ).pack(side="left")

        speech_controls = tk.Frame(sent, bg=CARD_BG)
        speech_controls.pack(fill="x")
        self.speak_letters_var = tk.BooleanVar(value=False)
        self.speak_words_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(speech_controls, text="Speak each letter", variable=self.speak_letters_var,
                         style="Card.TCheckbutton").pack(side="left", padx=(0, 16))
        ttk.Checkbutton(speech_controls, text="Speak each word", variable=self.speak_words_var,
                         style="Card.TCheckbutton").pack(side="left")
        if self.speaker is None:
            for child in speech_controls.winfo_children():
                child.state(["disabled"])

        # ---- Footer hint bar ----------------------------------------------
        footer = tk.Frame(self.root, bg="#e2e8f0")
        footer.grid(row=2, column=0, sticky="ew")
        tk.Label(
            footer,
            text="Space / Backspace / Esc (clear word) work as keyboard keys · 1-5 pick a suggestion",
            font=FONT_HINT, bg="#e2e8f0", fg=TEXT_SECONDARY, pady=8,
        ).pack()

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
                # Direct call, not model.predict(): ~20x faster per single frame
                # (52.8 ms vs 2.6 ms measured) with identical outputs.
                probs = self.model(np.expand_dims(vector, axis=0), training=False).numpy()[0]
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
            self.letter_var.set(self.last_letter or "?")
            pct = round(self.last_confidence * 100)
            self.confidence_var.set(pct)
            self.confidence_label_var.set(f"{pct}%")
            if self.armed:
                dot_color, status_text = SUCCESS, "Hand detected – ready to type next"
            else:
                dot_color, status_text = WARNING, "Hand detected – hold released to type next"
            self.letter_box.configure(bg=ACCENT_LIGHT)
            self.letter_label.configure(bg=ACCENT_LIGHT, fg=ACCENT_DARK)
        else:
            self.letter_var.set("–")
            self.confidence_var.set(0)
            self.confidence_label_var.set("0%")
            dot_color, status_text = IDLE_DOT, "No hand detected"
            self.letter_box.configure(bg="#f1f5f9")
            self.letter_label.configure(bg="#f1f5f9", fg=TEXT_SECONDARY)
        self.status_dot.itemconfig(self._status_dot_id, fill=dot_color)
        self.hand_status_var.set(status_text)

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
                btn.configure(text=self.current_suggestions[i], state="normal",
                              bg=ACCENT_LIGHT, fg=ACCENT_DARK)
            else:
                btn.configure(text="", state="disabled", bg=MUTED_BTN_BG, fg=MUTED_BTN_BG)

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
        self.word_var.set("Hold letter shapes to spell a word")
        self._refresh_suggestions()
        self._render_sentence()
        if self.speaker is not None and self.speak_words_var.get():
            self.speaker.say(word)

    def _backspace(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
            self.word_var.set((self.current_word + "_") if self.current_word else "Hold letter shapes to spell a word")
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
        self.word_var.set("Hold letter shapes to spell a word")
        self._refresh_suggestions()

    def _clear_all(self):
        self.current_word = ""
        self.sentence = ""
        self.word_var.set("Hold letter shapes to spell a word")
        self._refresh_suggestions()
        self._render_sentence()

    def _speak_sentence(self):
        text = self.sentence.strip()
        if text and self.speaker is not None:
            self.speaker.say(text)

    def _render_sentence(self):
        self.sentence_text.configure(state="normal")
        self.sentence_text.delete("1.0", "end")
        if self.sentence.strip():
            self.sentence_text.insert("1.0", self.sentence)
        else:
            self.sentence_text.insert("1.0", "Your sentence will appear here as you spell words…", "placeholder")
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
