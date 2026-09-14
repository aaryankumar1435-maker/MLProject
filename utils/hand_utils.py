"""MediaPipe Hands utilities for static hand-shape (letter) recognition.

The word pipeline (mp_utils.py) uses MediaPipe Holistic, which locates
hands using pose/body context. ISL letter images are close-up hand crops
with no visible body, so Holistic detects nothing on them (verified while
building this: 0/5 test images got any Holistic landmarks). MediaPipe
Hands detects hands directly without needing body context, and works on
both the cropped dataset images and normal webcam frames, so it's used for
the whole letter pipeline instead.
"""
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

HAND_LANDMARKS = 21
FEATURE_DIM = HAND_LANDMARKS * 3 * 2  # left hand + right hand, x/y/z each = 126


def create_hands(static_image_mode=False, max_num_hands=2,
                  min_detection_confidence=0.5, min_tracking_confidence=0.5):
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def detect(frame_bgr, hands):
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    return hands.process(image)


def extract_hand_landmarks(results):
    """Fixed 126-d vector: left hand (63) + right hand (63), by MediaPipe's
    handedness label. A missing hand is zero-filled so the shape is always
    constant. ISL fingerspelling uses two-handed letters, unlike ASL, so
    both hands are kept (not just the dominant one).
    """
    left = np.zeros(HAND_LANDMARKS * 3, dtype=np.float32)
    right = np.zeros(HAND_LANDMARKS * 3, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            vec = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            ).flatten()
            if handedness.classification[0].label == "Left":
                left = vec
            else:
                right = vec

    return np.concatenate([left, right])


def hands_present(results):
    """Number of hands detected in the frame, for extraction diagnostics."""
    return len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0


def _normalize_single_hand(vec63: np.ndarray) -> np.ndarray:
    points = vec63.reshape(HAND_LANDMARKS, 3)
    if not np.any(points):
        return vec63  # zero-padded (hand not detected) - leave as-is

    min_corner = points.min(axis=0)
    extent = points.max(axis=0) - min_corner
    scale = float(np.max(extent))
    if scale < 1e-8:
        return vec63

    normalized = (points - min_corner) / scale
    return normalized.flatten().astype(np.float32)


def normalize_landmarks(vector: np.ndarray) -> np.ndarray:
    """Rescale each hand's landmarks relative to that hand's own bounding
    box, independently per hand, so the features are invariant to hand size
    and distance from the camera rather than raw image-relative coordinates.

    Applied as a preprocessing step at both training and inference time
    (not baked into extraction) so raw landmarks stay available and
    extraction doesn't need to be redone if the normalization changes.

    Method follows Thomas et al., "An Open-Source American Sign Language
    Fingerspell Recognition and Semantic Pose Retrieval Interface"
    (arXiv:2408.09311), which normalizes each point "relative to the bounds
    of the hand itself" for the same reason.
    """
    left = _normalize_single_hand(vector[:HAND_LANDMARKS * 3])
    right = _normalize_single_hand(vector[HAND_LANDMARKS * 3:])
    return np.concatenate([left, right]).astype(np.float32)


def draw_hand_landmarks(frame_bgr, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
            )
    return frame_bgr
