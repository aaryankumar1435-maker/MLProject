"""Tests for the letter pipeline.

Run from the project root:
    .venv\\Scripts\\python -m unittest discover -s tests -v

The model test is skipped automatically if the trained model or the
processed Testing split isn't present (e.g. on a fresh clone).
"""
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from utils.autocomplete import WordCompleter
from utils.hand_utils import FEATURE_DIM, HAND_LANDMARKS, normalize_landmarks
from utils.letter_data import load_split

ROOT = Path(__file__).resolve().parent.parent


def fake_hand(seed=0):
    """21 random (x, y, z) points, like one MediaPipe hand."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.6, size=(HAND_LANDMARKS, 3)).astype(np.float32)


class NormalizeLandmarksTest(unittest.TestCase):
    def test_same_hand_closer_or_farther_gives_same_features(self):
        hand = fake_hand()
        near = np.concatenate([np.zeros(63, np.float32), hand.flatten()])
        # Same shape, twice as big and shifted: what moving closer to the camera does.
        far = np.concatenate([np.zeros(63, np.float32), (hand * 2.0 + 0.1).flatten()])
        np.testing.assert_allclose(normalize_landmarks(near), normalize_landmarks(far), atol=1e-5)

    def test_missing_hand_stays_zero(self):
        vec = np.concatenate([np.zeros(63, np.float32), fake_hand().flatten()])
        out = normalize_landmarks(vec)
        self.assertFalse(np.any(out[:63]))

    def test_output_shape_and_range(self):
        vec = np.concatenate([fake_hand(1).flatten(), fake_hand(2).flatten()])
        out = normalize_landmarks(vec)
        self.assertEqual(out.shape, (FEATURE_DIM,))
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0 + 1e-6)


class AutocompleteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        self.tmp.write("the\nhello\nhelp\nhelpful\nhello\nworld\n")
        self.tmp.close()
        self.completer = WordCompleter(Path(self.tmp.name))

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_prefix_matches_keep_frequency_order(self):
        self.assertEqual(self.completer.suggest("HEL"), ["HELLO", "HELP", "HELPFUL"])

    def test_case_insensitive_and_deduplicated(self):
        self.assertEqual(self.completer.suggest("hello"), ["HELLO"])

    def test_limit(self):
        self.assertEqual(len(self.completer.suggest("HEL", limit=2)), 2)

    def test_is_word(self):
        self.assertTrue(self.completer.is_word("world"))
        self.assertFalse(self.completer.is_word("wor"))


class LoadSplitTest(unittest.TestCase):
    def test_drops_no_hand_samples_and_keeps_split(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            good = np.concatenate([fake_hand().flatten(), np.zeros(63, np.float32)])
            np.save(d / "a.npy", good)
            np.save(d / "b.npy", np.zeros(FEATURE_DIM, np.float32))  # detection failure
            np.save(d / "c.npy", good)
            rows = [
                {"path": "a.npy", "label": "A", "split": "Testing"},
                {"path": "b.npy", "label": "A", "split": "Testing"},
                {"path": "c.npy", "label": "B", "split": "Training"},
            ]
            X, y = load_split(d, rows, {"A": 0, "B": 1}, "Testing")
        self.assertEqual(X.shape, (1, FEATURE_DIM))
        self.assertEqual(y.tolist(), [0])


class AugmentLayerTest(unittest.TestCase):
    def test_inactive_at_inference_and_keeps_missing_hand_zero(self):
        from utils.custom_layers import RandomLandmarkAugment

        layer = RandomLandmarkAugment()
        vec = np.concatenate([np.zeros(63, np.float32), fake_hand().flatten()])[None]
        np.testing.assert_array_equal(layer(vec, training=False).numpy(), vec)
        jittered = layer(vec, training=True).numpy()
        self.assertFalse(np.any(jittered[0, :63]), "a missing hand must stay all-zero")
        self.assertFalse(np.allclose(jittered[0, 63:], vec[0, 63:]), "a present hand should be jittered")


class TrainedModelTest(unittest.TestCase):
    MODEL = ROOT / "models" / "letter_model.keras"
    LABELS = ROOT / "models" / "letter_model.labels.json"
    PROCESSED = ROOT / "data" / "letters_processed"

    @unittest.skipUnless(
        (ROOT / "models" / "letter_model.keras").exists()
        and (ROOT / "data" / "letters_processed" / "manifest.csv").exists(),
        "trained model or processed data not present",
    )
    def test_model_is_accurate_on_a_testing_sample(self):
        import tensorflow as tf

        import utils.custom_layers  # noqa: F401 - registers custom layers for loading

        model = tf.keras.models.load_model(self.MODEL)
        index_to_label = {int(k): v for k, v in json.loads(self.LABELS.read_text()).items()}
        label_to_index = {v: k for k, v in index_to_label.items()}
        with open(self.PROCESSED / "manifest.csv", newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == "Testing"]
        # Every 10th Testing sample: quick, and spread across all 26 letters.
        X, y = load_split(self.PROCESSED, rows[::10], label_to_index, "Testing")
        pred = model(X, training=False).numpy().argmax(axis=1)
        accuracy = float((pred == y).mean())
        self.assertEqual(len(index_to_label), 26)
        self.assertGreater(accuracy, 0.93, f"accuracy {accuracy:.3f} on {len(y)} samples")


if __name__ == "__main__":
    unittest.main()
