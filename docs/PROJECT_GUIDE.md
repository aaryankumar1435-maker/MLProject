# ISL Sign Language Recognition: The Complete Project Guide

> A plain-language guide to the whole project: what it does, how to run it, how it works, what every
> file does, which tools it uses, and why things are built the way they are.
>
> For deeper detail: `README.md` is the overview, `TRAINING_LOG.md` records every experiment and bug,
> and `LETTER_UI.md` covers the desktop app in depth.

---

## Contents

1. [What this project is](#1-what-this-project-is)
2. [Key words you need to know](#2-key-words-you-need-to-know)
3. [Tech stack](#3-tech-stack)
4. [How to run the project](#4-how-to-run-the-project)
5. [The big picture](#5-the-big-picture)
6. [Letter pipeline, step 1: extract landmarks](#6-letter-pipeline-step-1-extract-landmarks)
7. [Letter pipeline, step 2: train the model](#7-letter-pipeline-step-2-train-the-model)
8. [Letter pipeline, step 3: measure accuracy](#8-letter-pipeline-step-3-measure-accuracy)
9. [Letter pipeline, step 4: recognize live](#9-letter-pipeline-step-4-recognize-live)
10. [The desktop app: letters to words to sentences](#10-the-desktop-app-letters-to-words-to-sentences)
11. [The word pipeline (paused)](#11-the-word-pipeline-paused)
12. [Folder map and what every file does](#12-folder-map-and-what-every-file-does)
13. [Results](#13-results)
14. [Known problems](#14-known-problems)
15. [Troubleshooting](#15-troubleshooting)
16. [Quick Q&A for revision and interviews](#16-quick-qa-for-revision-and-interviews)

---

## 1. What this project is

**In one sentence:** you show a hand sign to your webcam, and the computer recognizes which letter of
the Indian Sign Language (ISL) alphabet it is, types it, and can speak it out loud.

It has two parts:

| Part | Status | What it recognizes |
|---|---|---|
| **Letter pipeline** | **Finished and working** | The 26 letters A–Z, held as still hand shapes (fingerspelling) |
| **Word pipeline** | Paused, code kept | Whole ISL words, which are hand *movements* over time |

On top of the letter pipeline sits a **desktop app** that lets you spell words letter by letter, offers
autocomplete suggestions, builds sentences, and reads them aloud.

**Headline result:** on 4,636 test photos the model had never seen, it names the right letter
**96.07%** of the time.

**The main idea:** the model never looks at the picture itself. A tool called MediaPipe first finds 21
points on each hand (fingertips, knuckles, wrist), and the model only sees those points' positions.
That makes it small, fast, and much less bothered by lighting or background.

Everything runs **offline on your own computer**. No cloud service is called.

---

## 2. Key words you need to know

| Word | Simple meaning |
|---|---|
| **ISL** | Indian Sign Language. Some of its letters use **two hands**, unlike American Sign Language. |
| **Fingerspelling** | Spelling a word one letter at a time with hand shapes. |
| **Landmark** | One tracked point on the hand, like the tip of the index finger. MediaPipe gives **21 per hand**, each with x, y and z. |
| **Feature vector** | The list of numbers fed to the model. Here: 2 hands × 21 points × 3 coordinates = **126 numbers**. |
| **Normalization** | Rescaling the numbers so a hand close to the camera and the same hand far away give the same values. |
| **Model / classifier** | The trained neural network that turns 126 numbers into "this is the letter L". |
| **MLP** | Multi-Layer Perceptron: the simplest kind of neural network, a few layers of connected "neurons". It won here. |
| **Softmax / confidence** | The model's last step gives 26 probabilities that add up to 1. The biggest is its guess; its value (e.g. 0.94) is the confidence. |
| **Training / Validation / Testing split** | Photos used to learn, photos used to decide when to stop learning, and photos kept completely hidden until the final score. |
| **Augmentation** | Making extra training examples by slightly rotating, resizing and shifting the hand points. |
| **Accuracy** | The share of test photos where the model's guess was right. |
| **Precision / recall** | For one letter: precision = "when it says S, how often is it really S?"; recall = "of all real S's, how many did it catch?" |
| **Macro F1** | One score combining precision and recall, averaged equally over all 26 letters. |
| **Confusion matrix** | A 26 × 26 table of "real letter vs guessed letter". It shows exactly which letters get mixed up. |
| **Majority vote / smoothing** | Only trusting a letter once most of the last few frames agree, so one flickery frame can't change the answer. |
| **TTS** | Text-to-speech: the computer reading text aloud. |

---

## 3. Tech stack

| Job | Tool | Why this one |
|---|---|---|
| Finding hand points | **MediaPipe Hands** (Google) | Turns a camera frame into 21 3D points per hand, fast, on the CPU. The model then only needs to learn from 126 numbers, not millions of pixels. |
| Camera and drawing | **OpenCV** (`opencv-python`) | Reads the webcam, converts colours, draws the hand skeleton on the video. |
| The neural network | **TensorFlow / Keras** | Builds, trains, saves and loads the models. |
| Scoring | **scikit-learn** | Accuracy, precision, recall, F1 and the confusion matrix. |
| Charts | **Matplotlib** | Draws the confusion-matrix heatmap image. |
| Desktop app | **Tkinter** (built into Python) + **Pillow** | The window, buttons and live video. Pillow converts camera frames into images Tkinter can show. |
| Speech | **pyttsx3** | Offline text-to-speech, run on a background thread so the camera never freezes while it talks. |
| Data files | **NumPy**, **pandas** | Hand points saved as `.npy` files; lists of files saved as CSV. |
| Holistic body tracking (words only) | **MediaPipe Holistic** | Tracks the body pose and both hands together, needed for word signs. |
| Language | **Python 3.11**, on Windows | |

**Why the exact versions matter:** the code uses MediaPipe's older `solutions` API, which was removed
in MediaPipe 1.0. That older MediaPipe needs NumPy below 2 and protobuf below 5, which in turn limits
TensorFlow and OpenCV versions. `requirements.txt` pins a set that is known to install and work
together, so don't upgrade single packages casually.

---

## 4. How to run the project

> Commands are for **Windows PowerShell**, run from the project folder `D:\MLProject`.

### 4.1 One-time setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

The environment in this folder is already set up, so you can skip this on this computer.

**Datasets** (only needed to re-extract or retrain; the trained model is already in `models/`):

- [RealSign ISL alphabet](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset) → put it in
  `data/letters_raw/Training|Validation|Testing/<letter>/*.jpg`
- [ayesha-hannure ISL dataset](https://github.com/ayesha-hannure/Indian-Sign-Language-dataset) → its photos
  were added into `data/letters_raw/Training/<letter>/` with a `src2_` name prefix.

### 4.2 Use it (the trained model is already included)

```powershell
# The desktop app: spell words and sentences
.venv\Scripts\python 4_letter_recognition_ui.py

# The simple version: one letter shown on the video
.venv\Scripts\python 3_realtime_letter_recognition.py
```

Hold a letter shape in front of the webcam for about half a second. Relax your hand between letters.
Press **q** to quit the simple version; close the window to quit the app.

Useful flags (both scripts):

| Flag | Default | What it changes |
|---|---|---|
| `--camera 1` | `0` | Use a different webcam |
| `--no-speech` | off | Turn off speaking |
| `--confidence-threshold` | `0.80` | How sure the model must be before a frame counts |
| `--smoothing-window` | `8` | How many frames must agree before a letter is accepted |
| `--release-frames` (app only) | `6` | How long you must relax before the next letter can be typed |

### 4.3 Rebuild everything from the photos (optional)

```powershell
.venv\Scripts\python 1_extract_letter_landmarks.py                 # photos -> hand points (~50 min the first time)
.venv\Scripts\python 2_train_letter_model.py --arch mlp --augment    # hand points -> trained model
.venv\Scripts\python 4_evaluate_letter_model.py                      # score it + confusion-matrix image
```

`2_train_letter_model.py` saves to `models/letter_model_mlp_aug.keras`. The live scripts load
`models/letter_model.keras`, so copy the new model over it (with its `.labels.json`) if it's better.

### 4.4 Check that everything works

```powershell
.venv\Scripts\python -m unittest discover -s tests -t . -v   # 10 tests, a few seconds
.venv\Scripts\python tests\ui_smoke_test.py HELLO             # runs the app with test photos instead of a camera
```

---

## 5. The big picture

Every part follows the same shape: **picture → hand points → tidy up → model → steady answer → do
something with it.**

```
 OFFLINE (done once)                                          LIVE (every camera frame)

 ISL letter photos                                            webcam frame
      │                                                            │
      ▼  1_extract_letter_landmarks.py                             ▼  MediaPipe Hands
 126 numbers per photo (.npy files)                           126 numbers
      │                                                            │
      ▼  2_train_letter_model.py                                   ▼  normalize (same as training)
 trained model  ──────────────────── models/letter_model.keras ──► model: 26 probabilities
      │                                                            │
      ▼  4_evaluate_letter_model.py                                ▼  confidence ≥ 0.80?
 accuracy + confusion matrix                                   majority vote over 8 frames
                                                                   │
                                                  ┌────────────────┴────────────────┐
                                                  ▼                                 ▼
                                   3_realtime_letter_recognition.py      4_letter_recognition_ui.py
                                   letter shown on video + spoken        letter → word → sentence,
                                                                         autocomplete, speech
```

The most important rule: **the live scripts must prepare the numbers exactly the way training did.**
That's why the shared steps live in `utils/` and every script imports them from there.

---

## 6. Letter pipeline, step 1: extract landmarks

**File:** `1_extract_letter_landmarks.py` · **Helpers:** `utils/hand_utils.py`

### What it does

1. Finds every photo under `data/letters_raw/<split>/<letter>/`.
2. For each photo, MediaPipe Hands looks for up to 2 hands and returns 21 points per hand.
3. Those points become a list of **126 numbers**: the left hand's 63, then the right hand's 63.
   A hand that isn't there is filled with zeros, so every photo gives exactly 126 numbers.
4. Saves each list as a small `.npy` file under `data/letters_processed/<split>/<letter>/`.
5. Writes three summary files:
   - `labels.json`: letter → number (A=0 … Z=25);
   - `manifest.csv`: every `.npy` file with its letter and split;
   - `diagnostics.csv`: how many hands were found in each photo.

### Why MediaPipe **Hands** and not **Holistic**

Holistic (used by the word pipeline) finds hands by first finding the body. The letter photos are
close-ups of hands with no body visible, so Holistic found nothing: 0 hands in 5 of 5 test photos.
MediaPipe Hands finds hands directly, so it works on both close-up photos and normal webcam frames.

### Why both hands

Many ISL letters use two hands, so both are kept. MediaPipe labels each hand "Left" or "Right", and
that label decides which half of the 126 numbers it goes into.

### Why it's "incremental"

If a photo's `.npy` file already exists, it's reused instead of running MediaPipe again. When the second
dataset was added, only the new photos had to be processed, which saved about 50 minutes.

### Why the original split is kept

The dataset already comes split into Training / Validation / Testing. That split is used as-is, so the
Testing photos are truly unseen. Mixing and re-splitting at random could put near-identical photos of
the same person on both sides and make the score look better than it really is.

---

## 7. Letter pipeline, step 2: train the model

**File:** `2_train_letter_model.py` · **Helpers:** `utils/letter_data.py`, `utils/hand_utils.py`, `utils/custom_layers.py`

### 7.1 Loading the data (`utils/letter_data.py` → `load_split`)

For each split it:

1. **Drops photos where no hand was found at all** (all 126 numbers are zero). See 7.2 for why.
2. **Normalizes** each remaining vector (7.3).
3. Returns the numbers `X` and the letter indices `y`.

Training and evaluation both use this same function, so they always prepare data identically.

### 7.2 The "everything becomes O" bug (and its fix)

The first model scored only **84%**, and the letter **O** looked strange: precision 0.28, recall 0.97.
Every other letter had a chunk of its test photos wrongly called "O".

**Why:** photos where MediaPipe found no hand became 126 zeros. O had the most such photos (28.4% of O
photos failed detection), so the model learned "all zeros means O". Then every other letter's
failed-detection test photos, also all zeros, got called O.

**Fix:** drop all-zero vectors from every split. The same model on the same data jumped to **93%**.
The live scripts also skip prediction when no hand is visible, for the same reason.

### 7.3 Normalization (`utils/hand_utils.py` → `normalize_landmarks`)

MediaPipe gives each point's position as a fraction of the image (0 to 1). So the same hand shape gives
different numbers depending on where the hand is and how close it is. Normalization fixes that,
**separately for each hand**:

1. Find the hand's bounding box: the smallest and largest x, y and z of its 21 points.
2. Subtract the smallest corner, so the hand starts at 0.
3. Divide by the box's biggest side, so the hand fits between 0 and 1.

**Small example**, looking at x only:

| | Wrist x | Fingertip x |
|---|---|---|
| Hand far away | 0.40 | 0.50 |
| Same hand, closer | 0.30 | 0.50 |
| **After normalizing, both** | **0.0** | **1.0** |

A missing hand stays all zeros. This follows the method in Thomas et al. (arXiv:2408.09311).

### 7.4 The four model designs that were compared

| Design (`--arch`) | Idea in simple words |
|---|---|
| `mlp` | Feed all 126 numbers into two small layers (128, then 64 neurons), then 26 outputs. The simplest. |
| `wide_mlp` | The same, but bigger (256 → 128 → 64) with extra stabilizing layers. Tests "is the simple one too small?" |
| `pointnet` | Treats the 42 points as a cloud: learns features for each point, then keeps the strongest of each. |
| `transformer` | Lets every point "look at" every other point (self-attention), like modern language models. |

All four were trained with the same seed on the same data and scored on the same Testing photos.

### 7.5 Augmentation (`utils/custom_layers.py` → `RandomLandmarkAugment`)

During training only, each hand's points are randomly:

- rotated by up to **±15°**;
- resized by up to **±8%**;
- shifted by up to **±0.05**.

A missing hand stays zeros. This shows the model the small tilts and sizes a real webcam produces, which
the photos alone don't cover. It's **switched off** for validation, testing and live use, so it never
changes a real prediction. It was the single biggest improvement (see [Results](#13-results)).

### 7.6 How training runs

| Setting | Value | Meaning |
|---|---|---|
| Optimizer | Adam | The standard way to adjust the network's weights |
| Loss | Categorical cross-entropy | Punishes confident wrong answers the most |
| Epochs | up to 50 | Full passes over the training data |
| Batch size | 64 | Photos processed together per weight update |
| Early stopping | patience 8 | Stop when validation accuracy hasn't improved for 8 epochs, and keep the best weights |
| Seed | 42 | Makes runs repeatable, so design comparisons are fair |

At the end it saves the model and its label map and prints the Testing scores.

**Why the seed matters:** without it, running the same command twice gave 0.9435 and then 0.9277. That
1.6-point swing from luck alone was big enough to flip which design looked best.

### 7.7 Saving models with custom layers

Keras's `.keras` file stores a custom layer's *name*, not its code. So any script that loads a model
must first import the layer's code to register it. That's why both custom layers live in
`utils/custom_layers.py`, marked with `@register_keras_serializable`, and every loading script imports
that file. Forgetting this gives `TypeError: Could not locate class 'RandomLandmarkAugment'`.

---

## 8. Letter pipeline, step 3: measure accuracy

**File:** `4_evaluate_letter_model.py`

1. Loads the Testing split with the same `load_split` as training.
2. Runs the model on all 4,636 usable Testing photos.
3. Prints overall accuracy and each letter's precision, recall and F1.
4. Saves the confusion matrix as a heatmap (`models/letter_model_confusion_matrix.png`) and as raw
   counts (`.csv`).

**How to read the heatmap:** rows are the real letter and columns are the guess. Each row is shown as
fractions that add up to 1. A perfect model is a bright diagonal line. Any bright square off the
diagonal is a mix-up, for example the row S, column Q.

**Worked example of the scores**, for letter S:
- **recall 0.67:** of all real S photos, 67% were called S (33% were called Q);
- a letter with **precision 0.99** is almost never guessed wrongly.

---

## 9. Letter pipeline, step 4: recognize live

**File:** `3_realtime_letter_recognition.py`

For **every camera frame**:

1. MediaPipe Hands finds the hands; the skeleton is drawn on the frame.
2. **No hand at all?** Skip the model (it never learned all-zero input) and clear the history.
3. Otherwise, normalize the 126 numbers exactly like training and run the model.
4. If the top confidence is **below 0.80**, ignore this frame and clear the history.
5. Otherwise, add the guess to a list of the **last 8 frames**.
6. When those 8 frames are full and **at least 6 of the 8** agree, that letter is **stable**: show it,
   and speak it (at most once every 1.5 seconds for the same letter).

**Why the vote:** a single frame can wobble between two similar shapes. Requiring 6 of 8 frames
(under half a second) makes the displayed letter steady.

**Example:** last 8 guesses `L L L M L L L L` → L appears 7 times ≥ 6 → **L** is stable.
`L M L M L M L M` → L appears only 4 times → nothing shown yet.

### Speed

The model is called directly, `model(x, training=False)`, not with `model.predict(x)`. `predict()` sets up
a batch pipeline on every call. Measured on 300 real frames: **52.8 ms per frame with `predict()`, 2.6 ms
with the direct call**, with exactly the same probabilities. That makes the model 20× faster, and now
MediaPipe is most of the time per frame.

---

## 10. The desktop app: letters to words to sentences

**File:** `4_letter_recognition_ui.py` · **Helpers:** `utils/autocomplete.py`, `utils/tts_utils.py` · **Full guide:** `LETTER_UI.md`

### 10.1 What you see

- **Live camera** on the left, with the hand skeleton drawn on it.
- **Live Recognition:** the current letter in a big box, a confidence bar, and a coloured dot:
  - green = ready to type the next letter;
  - amber = waiting for you to relax your hand;
  - grey = no hand in view.
- **Spelling:** the word you're building (e.g. `HEL_`) and up to 5 autocomplete buttons.
- **Sentence:** the finished words, with Space, Backspace, Clear Word, Clear All and Speak buttons,
  and checkboxes to speak each letter or each word automatically.
- **Footer:** keyboard shortcuts.

### 10.2 How a held letter gets typed exactly once

The model judges each frame on its own, with no idea of "the key was released". If the app simply
typed whatever letter was stable, holding L for one second would type about 20 L's.

So the app adds an **"armed" switch** on top of the 6-of-8 vote:

```
  ARMED (green) ── stable letter seen ──►  type it once, become NOT ARMED (amber)
       ▲                                                  │
       └────── 6 frames in a row with no stable letter ◄──┘
               (you relaxed or moved your hand)
```

- Hold a shape: it's typed once.
- Relax your hand for about a third of a second: armed again.
- Hold the next shape: it's typed.
- **Double letters**, like the LL in HELLO: relax briefly between the two L's.

The app checks the camera every 15 milliseconds using Tkinter's `root.after`, so the window stays
responsive without extra threads.

### 10.3 Autocomplete (`utils/autocomplete.py` → `WordCompleter`)

- `data/word_list.txt` holds **9,893 English words, most common first** (from the public-domain
  google-10000-english list).
- After each typed letter, the app finds words starting with what you've spelled and shows the **first
  5**. Because the list is already sorted by how common each word is, the first matches are the most
  likely ones.
- Click a suggestion, or press **1–5**, to finish the word instantly.

Example: spelled `HEL` → `HELP`, `HELD`, `HELLO`, `HELPFUL`, `HELPS`.

### 10.4 Editing controls

| Control | What it does |
|---|---|
| **Space** (button or key) | Adds the current word to the sentence |
| **1–5** or a suggestion button | Adds that suggested word instead |
| **Backspace** (button or key) | Deletes the last letter; if the word is empty, pulls the last word back out of the sentence to fix it |
| **Clear Word** (**Esc**) | Throws away the word in progress |
| **Clear All** | Clears the word and the whole sentence |
| **Speak** | Reads the whole sentence aloud |

### 10.5 Speech without freezing (`utils/tts_utils.py` → `SpeechWorker`)

`pyttsx3` stops the program until it finishes speaking. If that happened on the camera loop, the video
would freeze mid-sentence. So speech runs on its own **background thread**: the app drops text into a
queue, and the thread speaks each item in turn.

### 10.6 Small UI details that fix real problems

- **Video width is capped at 560 pixels.** Some webcams ignore the requested size and send huge
  frames, which once pushed the whole sidebar off the screen.
- **The window is about 960 pixels tall,** so the footer isn't hidden behind the taskbar on a 1080p
  screen.

---

## 11. The word pipeline (paused)

This part recognizes whole ISL **words**. Words are **movements**, not held shapes, so it works differently:

| | Letters | Words |
|---|---|---|
| Tracker | MediaPipe Hands | MediaPipe **Holistic** (body pose + both hands) |
| Numbers per frame | 126 | **258** = pose 33 points × 4 (x, y, z, visibility) + 2 hands × 21 × 3 |
| Input to the model | one frame | **40 frames** in a row |
| Model | MLP | **LSTM** (a network that reads sequences in order) |
| Dataset | photos | **INCLUDE** videos: `data/raw/<category>/<word>/<video>.mp4` |

**Files:**

| File | What it does |
|---|---|
| `utils/mp_utils.py` | Holistic setup, the 258-number feature, `SEQ_LEN = 40`, and picking 40 evenly spaced frames from any video |
| `1_extract_landmarks.py` | Each video → a 40 × 258 array saved as `.npy`, plus labels, manifest and detection-rate diagnostics |
| `2_train_model.py` | LSTM(128) → LSTM(64) → Dense(64) → one output per word; random 80/20 train/validation split; saves `models/sign_model.keras` |
| `3_realtime_recognition.py` | Keeps the last 40 webcam frames; predicts once the buffer is full; speaks words above 0.80 confidence, at most every 2 seconds per word |

**Why it's paused:** the dataset folder `data/raw/` is empty, so there's nothing to train on. The
INCLUDE dataset is tens of gigabytes of video. The design doc
(`sign_language_project_documentation.docx`, "Improvement Plan") also lists fixes to make before
restarting:

1. Split by **signer**, so the same person never appears in both training and testing.
2. Add a real **held-out test set**.
3. **Normalize** the points, like the letter pipeline does.
4. Add a **"no sign"** class, so resting hands aren't forced into some word.
5. **Smooth** predictions over time, like the letters' majority vote.

---

## 12. Folder map and what every file does

```
MLProject/
├── 1_extract_letter_landmarks.py    letters step 1: photos → 126-number .npy files
├── 2_train_letter_model.py          letters step 2: train and compare mlp / wide_mlp / pointnet / transformer
├── 3_realtime_letter_recognition.py letters step 4: webcam → letter on video + speech
├── 4_evaluate_letter_model.py       letters step 3: accuracy, per-letter scores, confusion matrix
├── 4_letter_recognition_ui.py       the desktop app: letter → word → sentence, autocomplete, speech
│
├── 1_extract_landmarks.py           words step 1: videos → 40×258 sequences   (paused)
├── 2_train_model.py                 words step 2: train the LSTM              (paused)
├── 3_realtime_recognition.py        words step 3: webcam → spoken word        (paused)
│
├── utils/
│   ├── hand_utils.py      MediaPipe Hands: find hands, make the 126 numbers, normalize, draw
│   ├── letter_data.py     load_split: load one split, drop no-hand photos, normalize
│   ├── custom_layers.py   RandomLandmarkAugment (training jitter) + AddPositionEmbedding (transformer)
│   ├── autocomplete.py    WordCompleter: most-common-first prefix suggestions
│   ├── tts_utils.py       SpeechWorker: text-to-speech on a background thread
│   └── mp_utils.py        MediaPipe Holistic for the word pipeline: 258 numbers, SEQ_LEN = 40
│
├── tests/
│   ├── test_letter_pipeline.py   10 unit tests (normalization, autocomplete, loading, augmentation, model accuracy)
│   └── ui_smoke_test.py          runs the app with test photos as a fake camera, checks it types the word
│
├── models/
│   ├── letter_model.keras + .labels.json     THE model the app uses (mlp + augment)
│   ├── letter_model_<design>[_aug].keras     the other 5 trained designs, kept for comparison
│   └── letter_model_confusion_matrix.png/.csv  the Testing-set mix-up table
│
├── data/
│   ├── word_list.txt           9,893 English words, most common first (in git)
│   ├── letters_raw/            the photo datasets (not in git)
│   ├── letters_processed/      the extracted .npy files + labels.json, manifest.csv, diagnostics.csv (not in git)
│   ├── raw/                    INCLUDE videos for words (empty; not in git)
│   └── processed/              extracted word sequences (empty; not in git)
│
├── docs/PROJECT_GUIDE.md        this guide
├── README.md                    project overview and results
├── TRAINING_LOG.md              lab notebook: datasets, bugs, every experiment, in order
├── LETTER_UI.md                 the desktop app in depth
├── letter_recognition_architecture.docx      design document for letters (with cited papers)
├── sign_language_project_documentation.docx  design document + improvement plan for words
└── requirements.txt             exact package versions known to work together
```

### Functions worth knowing

| Function | File | What it does |
|---|---|---|
| `create_hands` | `utils/hand_utils.py` | Starts MediaPipe Hands (photo mode for extraction, video mode for live) |
| `detect` | `utils/hand_utils.py` | Runs MediaPipe on one frame (converts BGR to RGB first) |
| `extract_hand_landmarks` | `utils/hand_utils.py` | MediaPipe result → 126 numbers, left hand then right, zeros if missing |
| `normalize_landmarks` | `utils/hand_utils.py` | Per-hand bounding-box scaling (7.3) |
| `load_split` | `utils/letter_data.py` | Loads a split, drops no-hand photos, normalizes |
| `build_mlp` / `build_wide_mlp` / `build_pointnet` / `build_transformer` | `2_train_letter_model.py` | The four model designs |
| `build_model` | `2_train_letter_model.py` | Wraps a design with augmentation if `--augment`, and compiles it |
| `RandomLandmarkAugment` | `utils/custom_layers.py` | Training-only rotate/resize/shift (7.5) |
| `WordCompleter.suggest` | `utils/autocomplete.py` | Top-N words starting with a prefix |
| `SpeechWorker.say` | `utils/tts_utils.py` | Queue text to be spoken without blocking |
| `LetterRecognitionUI._update_frame` | `4_letter_recognition_ui.py` | The per-frame loop: detect → predict → vote → armed switch → draw |

---

## 13. Results

### 13.1 What each change bought

All on the same untouched Testing photos:

| Step | Test accuracy | What changed |
|---|---|---|
| First model | 0.84 | Plain MLP, no filtering (the "everything becomes O" bug) |
| + drop no-hand photos | 0.93 | Data fix (7.2) |
| + second dataset, seeded `mlp` | 0.9351 | More signers and conditions |
| `wide_mlp` | 0.9349 | Bigger model: no gain, so size wasn't the problem |
| `pointnet` | 0.8632 | Worse: keeping only the strongest feature per point loses "which finger is above which" |
| `transformer` | 0.9031 | Worse: too much capacity for 126 numbers |
| `transformer` + augment | 0.9299 | Augmentation helps it too (+2.7 points) |
| **`mlp` + augment** | **0.9607** | **The winner** (+2.6 points over plain `mlp`); macro F1 0.9596 |

**Lesson:** the fancy designs lost to the simplest one, and the biggest win was a better **training data
trick** (augmentation), not a bigger model.

### 13.2 Where the remaining 4% of errors are

Nearly every letter is recognized 95–100% of the time. The mistakes are concentrated in a few pairs that
look alike:

| Letter | Recall | Mostly confused with |
|---|---|---|
| S | 0.67 | Q |
| W | 0.75 | M |
| X | 0.92 | T |

### 13.3 Verified in the finishing pass

- The evaluation still gives **0.9607**, with a confusion matrix identical count for count.
- **Live prediction is 20× faster** (52.8 → 2.6 ms per frame) with identical outputs.
- The app, fed real test photos as a fake camera, types **HI, HELP and HELLO** exactly, and runs at
  about 18–21 frames per second.

---

## 14. Known problems

- **S/Q, W/M and X/T still get confused** (13.2). More training photos of those letters, or features
  such as angles between fingers, would be the next thing to try.
- **The tuning numbers were picked by feel.** The confidence threshold (0.80), vote window (8) and
  release frames (6) haven't been tested with many different signers.
- **Tested on photos, not many real people.** The Testing split is photos from the datasets. Accuracy on
  a new person's webcam, lighting and camera angle may be lower.
- **Letters only; no motion letters.** Letters that some sign systems sign with movement are treated as
  still shapes.
- **Autocomplete has no context.** It only looks at the letters of the current word, not the words before
  it, and it can't suggest names or words outside its 9,893-word list (you can still spell them fully).
- **A double letter needs a pause.** HELLO needs you to relax between the two L's.
- **MediaPipe may label both hands "Right".** Then the second hand overwrites the first in the 126
  numbers. This happens rarely.
- **The word pipeline is paused** (section 11).

---

## 15. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open camera index 0` | Close other apps using the camera (Zoom, Teams, the browser), or try `--camera 1`. |
| `Model/labels not found` | `models/letter_model.keras` is missing. Run `2_train_letter_model.py`, then copy the result to `letter_model.keras` + `.labels.json`. |
| `Could not locate class 'RandomLandmarkAugment'` | A script loaded the model without `import utils.custom_layers` first (7.7). |
| `module 'mediapipe' has no attribute 'solutions'` | MediaPipe was upgraded past 0.10. Reinstall from `requirements.txt`. |
| NumPy / protobuf version errors | Something installed NumPy 2 or protobuf 5+. Reinstall from `requirements.txt`. |
| The same letter types many times | Raise `--release-frames` (e.g. 10). |
| Letters are hard to type / nothing types | Improve lighting, keep your whole hand in view, or lower `--confidence-threshold` to 0.7. |
| Wrong letters flicker | Raise `--smoothing-window` (e.g. 12). |
| The app window doesn't fit on screen | Use `--video-width 440`. |
| No speech | Check your Windows voices and volume, or run with `--no-speech`. |
| The model test says "skipped" | The model or `data/letters_processed/` isn't there (normal on a fresh clone). |

---

## 16. Quick Q&A for revision and interviews

- **Why hand points instead of the raw image?**
  126 numbers are much smaller than millions of pixels, so the model is tiny and fast and mostly ignores
  lighting and background. MediaPipe has already done the hard vision work.
- **Why normalize each hand separately?**
  So the same hand shape gives the same numbers wherever the hand is and however close it is.
- **Why drop photos with no detected hand?**
  They're all zeros, carry no information, and taught the model "zeros means O", which cost 9 points
  of accuracy.
- **Why did the simple MLP beat PointNet and the Transformer?**
  PointNet's max-pooling throws away which point is where relative to the others. The Transformer has
  far more capacity than 126 numbers need, so it mostly adds noise.
- **What gave the biggest accuracy gain?**
  Augmentation: small random rotations, resizes and shifts during training (+2.6 points).
- **Why use the dataset's own Testing split?**
  A random re-split can put near-identical photos on both sides and inflate the score.
- **Why a majority vote at live time?**
  One frame can wobble between similar shapes; 6 of 8 frames agreeing makes the answer steady.
- **Why the "armed" switch in the app?**
  The model has no concept of releasing a key, so without it, holding a letter would type it about 20
  times a second.
- **Why call the model directly instead of `model.predict`?**
  `predict()` has setup overhead on every call: 52.8 ms against 2.6 ms per frame, with the same answers.
- **Why is speech on a separate thread?**
  Speaking blocks until it's finished; on the camera loop, that would freeze the video.
- **Why an LSTM for words but not letters?**
  Letters are still shapes, so one frame is enough. Words are movements, so the model must read frames
  in order.
