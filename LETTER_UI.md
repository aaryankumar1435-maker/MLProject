# Fingerspelling-to-Text UI

`4_letter_recognition_ui.py` is a desktop UI on top of the existing letter
pipeline (`1_extract_letter_landmarks.py` → `2_train_letter_model.py` →
`models/letter_model.keras`). Where `3_realtime_letter_recognition.py`
prints one recognized letter onto the video frame, this UI turns a
sequence of held letter shapes into typed text: it accumulates recognized
letters into a word, offers autocomplete suggestions from a static English
word list as you go, and builds up a full sentence you can speak aloud.

## Running it

```
.venv\Scripts\python 4_letter_recognition_ui.py
```

Requires `models/letter_model.keras` / `.labels.json` to already exist
(run `2_train_letter_model.py` first if not - see `README.md`).

Flags (all optional, same defaults as `3_realtime_letter_recognition.py`
plus a few new ones):

| Flag | Default | Meaning |
|---|---|---|
| `--model`, `--labels` | `models/letter_model.keras`, `.labels.json` | Which trained model to load |
| `--word-list` | `data/word_list.txt` | Autocomplete dictionary (see below) |
| `--camera` | `0` | OpenCV camera index |
| `--video-width` / `--video-height` | `560` / `420` | Requested capture size, and the max size the feed is displayed at (frames wider than this are downscaled before rendering, since some webcams ignore the requested capture resolution and a full-res frame can push the sidebar off-screen) |
| `--confidence-threshold` | `0.80` | Minimum softmax confidence for a frame's prediction to count at all |
| `--smoothing-window` | `8` | Frames of majority vote required before a prediction is considered stable |
| `--release-frames` | `6` | Consecutive not-stable frames required before the next letter can be typed (see "Letter commit model" below) |
| `--min-detection-confidence` | `0.5` | MediaPipe Hands detection threshold |
| `--no-speech` | off | Disable text-to-speech entirely (speech checkboxes are greyed out) |

## Layout

```
+---------------------------------------------------------------------------+
| ISL Fingerspelling -> Text                                  (dark header) |
| Hold a letter shape steadily to type it - relax your hand between letters |
+---------------------------------------------------------------------------+
| LIVE CAMERA            | LIVE RECOGNITION                                 |
| +--------------------+ |  (o) Hand detected - ready to type next          |
| | webcam feed with   | |  +-----+  Confidence                             |
| | hand landmarks     | |  |  L  |  [#################-----]  94%          |
| | drawn on it        | |  +-----+                                         |
| +--------------------+ | SPELLING                                         |
|                        |  HEL_                                            |
|                        |  Autocomplete - click or press 1-5               |
|                        |  [HELP] [HELD] [HELLO] [HELPFUL] [HELPS]         |
|                        | SENTENCE                                         |
|                        |  +--------------------------------------------+  |
|                        |  | I NEED                                     |  |
|                        |  +--------------------------------------------+  |
|                        |  [Space] [Backspace] [Clear Word] [Clear All]    |
|                        |  [Speak]   [ ] Speak each letter [x] Speak word  |
+---------------------------------------------------------------------------+
| Space / Backspace / Esc (clear word) work as keyboard keys - 1-5 picks     |
+---------------------------------------------------------------------------+
```

The status dot is **green** when the UI is armed (the next held letter
will be typed), **amber** while it waits for you to release the current
shape, and **grey** when no hand is in view. The confidence bar shows the
model's confidence in the current frame's letter.

The window is about 1,260 x 960 pixels at 100% scaling with the default
`--video-width 560`, so it fits above the taskbar on a 1080p screen.

## Letter commit model: "hold to type, release to re-arm"

The underlying recognizer classifies every frame independently (letters
are held poses, not gestures - see `README.md`), so naively appending
"whatever the model just said" would type the same letter dozens of times
per second while a shape is held. The UI adds a small state machine on top
of the existing majority-vote smoothing:

1. Each frame's prediction only counts if softmax confidence clears
   `--confidence-threshold`.
2. A prediction is **stable** once `--smoothing-window` consecutive
   counted frames agree on the same letter at least 75% of the time
   (identical to `3_realtime_letter_recognition.py`).
3. The UI is **armed** (ready to type) or not. A stable letter is only
   appended to the current word while armed; appending immediately
   disarms it.
4. The UI re-arms only after `--release-frames` consecutive *not-stable*
   frames - i.e. the signer visibly changed or dropped the hand shape.

In practice: hold a letter shape steadily for a few frames → it's typed
once → relax or move your hand → hold the next shape → it's typed. There
is deliberately no "same letter twice in a row without releasing" path;
if a word needs a doubled letter (e.g. "HELLO"), briefly relax the hand
between the two L's.

The status line and its coloured dot ("Hand detected - ready to type
next" in green, "hold released to type next" in amber) reflect `armed`
directly, so you can see when it's safe to hold the next shape.

## Autocomplete

`utils/autocomplete.py` (`WordCompleter`) loads `data/word_list.txt` once
at startup: ~9,900 English words, one per line, **ordered by real-world
frequency, most common first** (source: the public-domain
[first20hours/google-10000-english](https://github.com/first20hours/google-10000-english)
list, filtered to alphabetic entries only). `suggest(prefix, limit=5)`
returns the first `limit` words starting with `prefix` - because the list
is already frequency-sorted, "first N matches" is already "N most likely
matches" with no extra ranking/scoring step.

Every time a letter is committed, the UI recomputes suggestions for the
current in-progress word and shows up to 5 as buttons (click, or press the
`1`-`5` keys). Clicking a suggestion **replaces** whatever prefix you'd
spelled so far with the full word, appends a space, and starts a new word
- so you don't have to keep fingerspelling once autocomplete has found the
word you meant. If nothing in the list matches (e.g. a name or a typo),
the buttons stay empty/disabled and you can keep spelling manually or hit
Space/Backspace as-is.

This is a static offline dictionary, not a language model: suggestions
depend only on the current word's prefix, not on the words already in the
sentence (no grammar/context awareness), and anything outside the ~9,900-
word list (proper nouns, technical terms, less common words) will never
autocomplete - you can still spell it out fully and hit Space to accept
the raw spelling as typed.

## Editing controls

| Action | Effect |
|---|---|
| `Space` (button or key) | Finalizes the current word as spelled (or does nothing if nothing's been spelled) and starts a new word |
| Suggestion button / `1`-`5` | Finalizes the current word as that suggestion instead of the raw spelling |
| `Backspace` (button or key) | Removes the last letter of the word in progress; if the word in progress is empty, pulls the last *finalized* word out of the sentence back into the in-progress word so it can be corrected |
| `Clear Word` (`Esc`) | Discards the word in progress without touching the finished sentence |
| `Clear All` | Resets both the word in progress and the whole sentence |
| `Speak` | Speaks the full sentence so far via TTS (works even with the per-letter/per-word checkboxes off) |
| `Speak each letter` checkbox | Speaks every committed letter immediately (off by default - fast spelling gets noisy) |
| `Speak each word` checkbox | Speaks each word once it's finalized via Space or a suggestion (on by default) |

Both speech checkboxes are disabled and inert when the app is launched
with `--no-speech`.

## Testing without a camera

`tests/ui_smoke_test.py` runs this UI with the webcam replaced by real
photos from the Testing split. Each letter of a word is held for 20
frames, then 12 blank frames play (the "relaxed hand" gap). The test
passes only if the UI types the word exactly, once per letter:

```
.venv\Scripts\python tests\ui_smoke_test.py HELLO
.venv\Scripts\python tests\ui_smoke_test.py HELP --screenshot ui.png
```

`HELLO` is the useful case, because the doubled L only types twice if the
release-to-re-arm latch works.

## Speed

Each frame is classified with a direct model call,
`model(x, training=False)`, not `model.predict(x)`. `predict()` builds a
batching pipeline on every call, which measured about 20x slower on single
frames (52.8 ms vs 2.6 ms) with identical outputs. With the direct call,
the UI runs at roughly 18-21 frames per second on this laptop's CPU,
where MediaPipe Hands is now most of the per-frame cost.

## Known limitations

- **Single dictionary, no context.** Suggestions are prefix-only against a
  fixed 9,900-word list; no bigram/sentence-level language model, so it
  won't, e.g., prefer "SEE" over "SEA" based on prior words.
- **One camera, one hand-shape vocabulary.** Same constraints as the
  underlying classifier: it's the 26 static ISL letters from
  `models/letter_model.labels.json`, not numbers or dynamic (moving)
  letters like J/Z's motion variants some sign systems use.
- **Release-based commit, not timing-based.** `--release-frames` assumes
  the signer visibly relaxes/moves between letters. A signer who never
  fully releases a shape (e.g. transitioning directly between two very
  similar hand poses) may under- or over-type; raise/lower
  `--release-frames` and `--smoothing-window` to tune for a given signer
  and camera.
- **No signer-independent validation of the UI's timing choices.**
  `--confidence-threshold` / `--smoothing-window` come from
  `3_realtime_letter_recognition.py` (already tuned by feel, not a formal
  study); `--release-frames` is new to this UI and similarly untuned
  against real users beyond the smoke-test in this repo's history.
