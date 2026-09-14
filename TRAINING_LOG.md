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
