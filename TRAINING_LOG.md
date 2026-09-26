# Letter Pipeline — Training Log

Running record of dataset versions, bugs found, and model experiments for
the ISL letter classifier. Design rationale and citations live in
`letter_recognition_architecture.docx`; this file is the lab-notebook
history of what was actually run and why, in order.

## Dataset versions

### v1 — RealSign only
- Source: RealSign ISL alphabet dataset (CC0-1.0), 26 letters.
- Training 700/class, Validation 100/class, Testing 200/class (26,000 images).
- 25,975 usable after 2 unreadable files.
- **Detection failure rate**: 2,783/25,975 (10.7%) images had no hand
  detected by MediaPipe Hands at all — highly uneven across letters, from
  0.3% (R, T) to 28.4% (O). See `data/letters_processed/diagnostics.csv`.

### v2 — RealSign + ayesha-hannure (current)
- Added: [ayesha-hannure/Indian-Sign-Language-dataset](https://github.com/ayesha-hannure/Indian-Sign-Language-dataset)
  (Apache-2.0), 12,637 additional images, ~486/letter, self-recorded by a
  different team (per the dataset's README) — added for signer/condition
  diversity, not just volume.
- Merged into `data/letters_raw/Training/<letter>/` with a `src2_` filename
  prefix to avoid collisions; RealSign's `Validation`/`Testing` folders are
  untouched, so the held-out test set is unchanged across dataset versions
  and model comparisons stay apples-to-apples.
- `1_extract_letter_landmarks.py` was made incremental (skips images that
  already have a saved `.npy`) specifically so this addition didn't require
  redoing the ~50 minutes of already-extracted v1 images.

## Bug found: zero-vector collapse onto letter "O"

First model (v1 dataset, `mlp` architecture, trained without filtering):
**84% test accuracy**, but letter O had precision 0.28 / recall 0.97 —
every other letter's confusion matrix row had a chunk misrouted into the O
column (e.g. B: 49/200, D: 49/200, Z: 49/200 misclassified as O).

Root cause: `utils/hand_utils.py:extract_hand_landmarks` zero-fills a
missing hand, so any image where MediaPipe detected no hand at all becomes
an all-zero 126-d vector. O happened to have the dataset's highest
detection-failure rate (28.4%), so it got the most zero-vector training
examples, and the model learned "all-zero input -> predict O" - which then
misfires on every other letter's own zero-vector (detection-failure) test
images.

Fix: `2_train_letter_model.py:load_split` now drops all-zero vectors from
Training/Validation/Testing before use. Re-running the *same* v1 dataset
and architecture with this fix: **93% test accuracy**, O's precision back
to 0.89. This is a data-quality fix, not a model change - kept separate
from the architecture comparison below so the two aren't conflated.

## Model architecture comparison

Four architectures, same data pipeline (v2 dataset, zero-vector filtering,
per-hand bounding-box landmark normalization), same held-out Testing split,
compared by test accuracy and macro F1. All numbers below are from seeded
runs (`--seed 42`, the default — see "Reproducibility" below); an earlier
unseeded pass gave the same ranking but different absolute numbers, which
is why every run must be quoted with its seed.

| Arch | Description | Test accuracy | Macro F1 |
|---|---|---|---|
| `mlp` | Dense(128)→Dropout→Dense(64)→Dropout→Softmax (baseline, used for the v1 bug fix above) | 0.9351 | 0.9325 |
| `wide_mlp` | Dense(256)→BatchNorm→Dense(128)→BatchNorm→Dense(64)→Softmax | 0.9349 | 0.9348 |
| `pointnet` | Per-landmark shared Conv1D(1x1) → global max-pool → Dense→Softmax; treats the 42 landmarks as a point cloud, per Thomas et al. (arXiv:2408.09311) | 0.8632 | 0.8544 |
| `transformer` | Self-attention over the 42 landmarks (2 pre-norm transformer-encoder blocks + learned positional embedding) → mean-pool → Softmax; see `2_train_letter_model.py:build_transformer` | 0.9031 | 0.8942 |

Each trained via `2_train_letter_model.py --arch <name>` (saves to
`models/letter_model_<name>.keras` so all four survive for comparison).

Surprising result: `pointnet`'s permutation-invariant max-pool design,
which the cited paper uses for ASL fingerspelling, underperforms plain
dense layers here — max-pooling over per-landmark features throws away
exactly the relative-position information (e.g. "index finger tip is
*above* the knuckle") that distinguishes visually similar ISL letters,
and this dataset doesn't need permutation invariance since MediaPipe
always emits landmarks in the same canonical order. `mlp`, which sees all
126 numbers jointly from the first layer, doesn't have that limitation.
`transformer` (self-attention, order-aware via a learned positional
embedding) was built to test that theory further but still trailed `mlp`
— for a 126-d input this small, a plain dense layer already captures
cross-landmark relationships just fine, and attention's extra capacity
mostly adds variance, not signal.

### Augmentation experiment — the actual accuracy win

`--augment` (see `RandomLandmarkAugment` in `utils/custom_layers.py`)
applies random small rotation/scale/translation jitter to landmarks during
training only (never at validation/test time), to check whether the model
is data-limited on pose variation, not just architecture-limited.

| Run | Test accuracy | Macro F1 |
|---|---|---|
| `mlp` (no augment) | 0.9351 | 0.9325 |
| **`mlp --augment`** | **0.9607** | **0.9596** |
| `transformer` (no augment) | 0.9031 | 0.8942 |
| `transformer --augment` | 0.9299 | 0.9255 |

Augmentation helped both architectures it was tried on (+2.6pp for `mlp`,
+2.7pp for `transformer`), confirming the model was more data-limited than
architecture-limited: the raw dataset only shows each letter at whatever
tilt/scale the original photographer happened to use, and jittering
rotation/scale/translation at training time exposes the model to pose
variation closer to what a live webcam will actually produce. `mlp
--augment` is the best model found across the whole comparison and is
what's copied to the canonical `models/letter_model.keras` /
`.labels.json` that `3_realtime_letter_recognition.py` loads by default.

### Reproducibility: seeding, and a serialization bug found along the way

`2_train_letter_model.py` now takes `--seed` (default 42, applied via
`tf.keras.utils.set_random_seed`) so architecture comparisons aren't
confounded by run-to-run noise. This was added *because* of what happened
without it: an early unseeded `mlp --augment` run scored 0.9435, but
re-running the exact same command (after an unrelated code fix, see
below) scored 0.9277 — a 1.6pp swing from random weight init /
augmentation draws alone, large enough to flip a close architecture
comparison. All numbers in the tables above are from the seeded re-run.

Separately, saving/loading caught a real bug: `RandomLandmarkAugment` and
`AddPositionEmbedding` were originally defined inline in
`2_train_letter_model.py` with no `register_keras_serializable`
decoration. `2_train_letter_model.py --arch mlp --augment` saved fine, but
loading that `.keras` file back in a fresh process (i.e. exactly what
`3_realtime_letter_recognition.py` does) failed with `TypeError: Could not
locate class 'RandomLandmarkAugment'` — Keras 3's `.keras` format doesn't
embed custom layer code, only its registered name, so any script loading
the model needs that class registered first. Fixed by moving both classes
to `utils/custom_layers.py`, decorating them with
`@register_keras_serializable(package="isl_letters")`, and importing that
module (for its registration side effect) at the top of
`3_realtime_letter_recognition.py`. Verified by loading
`models/letter_model.keras` fresh and running a real prediction. Anyone
adding another custom layer to an architecture must follow the same
pattern (define in `utils/custom_layers.py`, decorate, import in both
training and inference scripts) or real-time inference will break the
moment that architecture is promoted to canonical.

### Final decision

`mlp --augment` (seed 42): **0.9607 test accuracy, 0.9596 macro F1**.
Copied to `models/letter_model.keras` / `models/letter_model.labels.json`.
All six candidate models (one per arch, plus the two `--augment` variants)
remain in `models/letter_model_<tag>.keras` for future comparison if the
dataset or feature representation changes.

## Held-out confusion matrix (`4_evaluate_letter_model.py`)

Re-running the canonical model on the Testing split independently of
training (`4_evaluate_letter_model.py`, same `load_split` zero-vector
filtering as training) reproduces the 0.9607 accuracy exactly, and breaks
down where the remaining ~4% of errors go: they're concentrated in a
handful of visually similar hand shapes rather than spread evenly across
all 26 letters. Recall drops most on **S** (0.67 - 33% of S's called Q),
**W** (0.75 - 15% called M), and **X** (0.92 - 4% called T); every other
letter is at 95-100% recall. See
`models/letter_model_confusion_matrix.png`/`.csv` for the full matrix.

## Fingerspelling UI (`4_letter_recognition_ui.py`)

Built a Tkinter desktop UI on top of the same model/pipeline as
`3_realtime_letter_recognition.py`, to turn single-letter predictions into
typed words/sentences instead of a one-line video overlay. Design and full
usage doc: `LETTER_UI.md`. Two things worth recording here because they
weren't obvious going in:

- **Naive per-frame appending doesn't work.** The classifier runs on every
  frame independently (no notion of "key up"), so directly appending
  whatever letter is currently stable would type it dozens of times per
  second while a shape is held. Fixed with an "armed" latch: a stable
  letter can only be appended once, and re-arming requires
  `--release-frames` (default 6) consecutive *not-stable* frames first -
  i.e. the signer must visibly relax/change the hand shape before the next
  letter can be typed. This is new, untuned state on top of the
  already-existing confidence-threshold/smoothing-window logic.
- **Webcam capture resolution can exceed the screen.** First run rendered
  the live feed at the camera's native resolution with no cap, which on
  this machine's webcam produced a frame wide enough to push the entire
  sidebar (letter/word/autocomplete/sentence panel) off the right edge of
  the screen - the window looked like it was just a bare video feed with
  no controls. `cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT)` alone wasn't
  sufficient (some webcam drivers ignore the request), so the UI also
  hard-downscales any frame wider than `--video-width` (default 560)
  before rendering it, independent of what the camera actually delivers.
  Verified visually via a full-screen screenshot after the fix - sidebar
  fits alongside the video at the default size.

Autocomplete (`utils/autocomplete.py:WordCompleter`) is backed by a static
~9,900-word frequency-ranked list (`data/word_list.txt`, filtered from the
public-domain first20hours/google-10000-english corpus) rather than a live
dictionary service or an installed NLP package (nltk/wordfreq/etc. were
all unavailable offline in this environment) - keeps the whole UI usable
with no network dependency at runtime. Since the list is already
frequency-sorted, ranking suggestions is just "first N prefix matches",
no separate scoring step.

## Finishing pass: UI redesign, speed, tests

- **UI redesign committed and verified.** The card-based redesign of
  `4_letter_recognition_ui.py` (header, live-recognition card with a
  status dot and confidence bar, spelling card, sentence card, footer
  hints) was checked end to end with `tests/ui_smoke_test.py`: real
  Testing photos played in as a fake webcam spell HI, HELP and HELLO, and
  the UI types each exactly (the doubled L in HELLO confirms the
  release-to-re-arm latch). A screenshot showed the window was 1,019 px
  tall, which hid the footer behind the taskbar on a 1080p screen;
  shrinking the sentence box from 5 to 3 lines brought it to 963 px.
- **20x faster per-frame inference.** Both live scripts called
  `model.predict()` on every frame. On 300 real Testing vectors that took
  52.8 ms per frame, against 2.6 ms for a direct `model(x,
  training=False)` call, with identical probabilities (max difference 0).
  `predict()` alone had capped the live loop below 20 frames per second
  before MediaPipe ran. Both scripts now use the direct call.
- **One shared `load_split`.** Training and evaluation each had their own
  copy; both now import `utils/letter_data.py`. Re-running
  `4_evaluate_letter_model.py` afterwards gave 0.9607 with a confusion
  matrix identical, count for count, to the saved one.
- **Tests added** (`tests/`): unit tests for normalization, autocomplete,
  data loading, the augmentation layer and a model-accuracy check, plus
  the UI smoke test above.
