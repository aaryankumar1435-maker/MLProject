# Sign Language Recognition (ISL)

Two independent pipelines live here:

- **Letters (current focus)** — static ISL fingerspelling alphabet (A-Z),
  MediaPipe Hands → feed-forward classifier. See
  `letter_recognition_architecture.docx` for the full design and why it
  deliberately differs from the word pipeline.
- **Words (on hold)** — isolated ISL words from the INCLUDE dataset,
  MediaPipe Holistic landmarks → 2-layer LSTM → real-time webcam
  prediction with optional speech. See
  `sign_language_project_documentation.docx`. Untouched and ready to resume
  later; the sections below cover this pipeline specifically.

## Letter pipeline (A-Z, static hand shapes)

| Stage | Script | Input | Output |
|---|---|---|---|
| 1 | `1_extract_letter_landmarks.py` | ISL letter images | 126-d hand-landmark vectors (`.npy`) |
| 2 | `2_train_letter_model.py` | Landmark vectors | Trained classifier + label map |
| 3 | `3_realtime_letter_recognition.py` | Webcam frames | Predicted letter + optional speech |

```
.venv\Scripts\python 1_extract_letter_landmarks.py
.venv\Scripts\python 2_train_letter_model.py --arch mlp --augment
.venv\Scripts\python 3_realtime_letter_recognition.py
```

`2_train_letter_model.py` supports four architectures (`--arch mlp|wide_mlp|pointnet|transformer`)
and an optional `--augment` flag (random rotation/scale/translation jitter
applied to training data only). All were compared under a fixed
`--seed` on the same held-out Testing split; `mlp --augment` won
(0.9607 test accuracy / 0.9596 macro F1) and is the canonical
`models/letter_model.keras`. Full comparison, rationale for each
architecture, and a serialization gotcha for anyone adding a new custom
layer: see `TRAINING_LOG.md`.

Dataset: [RealSign ISL alphabet dataset](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset)
(CC0-1.0), already downloaded into `data/letters_raw/{Training,Validation,Testing}/<letter>/*.jpg`.
That folder's own split is used as-is (see `letter_recognition_architecture.docx`
§2) — `Testing` is a genuine held-out test set, not a random re-split.

MediaPipe Holistic (used by the word pipeline) needs pose/body context to
locate hands and detects nothing on these close-up hand images, so the
letter pipeline uses MediaPipe Hands directly instead (`utils/hand_utils.py`).
Letters are held poses, not movements, so there's no 40-frame LSTM window —
each frame/image is classified independently, with a majority-vote
smoothing window at inference time to avoid flicker. Landmarks are also
normalized per-hand to that hand's own bounding box
(`utils/hand_utils.py:normalize_landmarks`), so predictions are invariant
to hand size/distance from the camera — see
`letter_recognition_architecture.docx` §6 for the research this design is
grounded in (3 cited papers) and the exact technique.

## Word pipeline (INCLUDE dataset, on hold)

| Stage | Script | Input | Output |
|---|---|---|---|
| 1 | `1_extract_landmarks.py` | INCLUDE videos | `40 × 258` landmark sequences (`.npy`) |
| 2 | `2_train_model.py` | Landmark sequences | Trained LSTM + label map |
| 3 | `3_realtime_recognition.py` | Webcam frames | Predicted word + optional speech |

Each frame is encoded as 258 features: pose (33 landmarks × x,y,z,visibility
= 132) + left hand (21 × x,y,z = 63) + right hand (21 × x,y,z = 63). The
face mesh is excluded. Every video is uniformly resampled to 40 frames.
`SEQ_LEN` and `FEATURE_DIM` live in `utils/mp_utils.py` and are imported by
all three stages, so extraction/training/inference can't drift out of sync.

## Setup

A working environment is already built at `.venv` (Python 3.11). To
recreate it elsewhere:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

`mediapipe`'s legacy `solutions.holistic` API (used here) was removed in
mediapipe 1.0+, and it requires `numpy<2` / `protobuf<5`. `requirements.txt`
pins a set that has been verified to install and run together on Windows +
Python 3.11 — don't casually bump individual packages without re-checking
this.

## Dataset layout

Download the [INCLUDE dataset](https://zenodo.org/record/4010759) (or your
own ISL video set) and arrange it as:

```
data/raw/<category>/<word>/<video>.mp4
```

e.g. `data/raw/Greetings/hello/MVI_0001.mp4`. The word/class label is taken
from the immediate parent folder of each video.

## Running it

```
.venv\Scripts\python 1_extract_landmarks.py
.venv\Scripts\python 2_train_model.py
.venv\Scripts\python 3_realtime_recognition.py
```

Useful flags:

- `1_extract_landmarks.py --raw-dir data/raw --out-dir data/processed` —
  writes `<word>/<video>.npy`, `labels.json`, `manifest.csv`, and
  `diagnostics.csv` (per-video pose/hand landmark detection rates — check
  this after extraction; low rates mean a video needs re-recording or
  should be excluded).
- `2_train_model.py --epochs 100 --batch-size 16 --val-split 0.2` — 80/20
  stratified train/validation split, Adam + categorical cross-entropy,
  early stopping and checkpointing on `val_categorical_accuracy`. Saves
  `models/sign_model.keras` and `models/sign_model.labels.json`.
- `3_realtime_recognition.py --confidence-threshold 0.80 --cooldown 2.0` —
  sliding 40-frame window; predicts only once the buffer fills; speaks a
  word once softmax confidence clears the threshold, at most once per
  cooldown window per repeated word. `--no-speech` disables TTS. TTS runs
  on a background thread so it never blocks the camera loop.

## Known limitations (see the documentation for the full list)

This is a working baseline, not a validated real-world system:

- The train/validation split is random, not signer-independent — it can
  overstate accuracy if the same signer appears in both splits.
- There's no held-out test set, no landmark normalization (coordinates are
  raw, camera/scale-dependent), no "no-sign"/background class, and no
  temporal smoothing across predictions.
- Confidence threshold (0.80) is a fixed cutoff, not a calibrated
  probability.

These are the "Phase 2+" items in the documentation's Improvement Plan and
are intentionally not implemented here yet — check the doc before deciding
which to tackle next.

## Project files

Word pipeline:
- `utils/mp_utils.py` — MediaPipe Holistic setup, landmark extraction/
  drawing, shared `SEQ_LEN`/`FEATURE_DIM` constants.
- `1_extract_landmarks.py`, `2_train_model.py`, `3_realtime_recognition.py`
  — the three pipeline stages.
- `data/raw/` — put the INCLUDE dataset here (not included).
- `data/processed/` — extraction output (`.npy` sequences, `labels.json`,
  `manifest.csv`, `diagnostics.csv`).
- `models/sign_model.keras`, `models/sign_model.labels.json`.

Letter pipeline:
- `utils/hand_utils.py` — MediaPipe Hands setup, 126-d feature extraction/
  drawing.
- `utils/custom_layers.py` — custom Keras layers (`RandomLandmarkAugment`,
  `AddPositionEmbedding`) shared between training and inference; kept
  separate and `register_keras_serializable`-decorated so
  `3_realtime_letter_recognition.py` can load models that use them (see
  `TRAINING_LOG.md` for why this matters).
- `utils/tts_utils.py` — shared non-blocking speech worker (used by both
  pipelines' real-time scripts).
- `1_extract_letter_landmarks.py`, `2_train_letter_model.py`,
  `3_realtime_letter_recognition.py` — the three pipeline stages.
- `data/letters_raw/` — RealSign ISL alphabet dataset
  (`Training`/`Validation`/`Testing`/`<letter>`).
- `data/letters_processed/` — extraction output (`.npy` vectors,
  `labels.json`, `manifest.csv`, `diagnostics.csv`).
- `models/letter_model.keras`, `models/letter_model.labels.json`.
- `letter_recognition_architecture.docx` — full design doc for this
  pipeline.
