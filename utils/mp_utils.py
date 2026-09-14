"""Shared MediaPipe Holistic utilities used by both the offline extraction
stage and the real-time webcam stage, so the feature representation used
during training always matches the one used at inference time.
"""
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21

# 33*4 (pose x,y,z,visibility) + 21*3 (left hand) + 21*3 (right hand)
FEATURE_DIM = POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 + HAND_LANDMARKS * 3  # 258

# Frames per sequence. Extraction and inference must use the same value.
SEQ_LEN = 40


def create_holistic(static_image_mode=False, model_complexity=1,
                     min_detection_confidence=0.5, min_tracking_confidence=0.5):
    return mp_holistic.Holistic(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def detect(frame_bgr, holistic):
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    return holistic.process(image)


def extract_landmarks(results):
    """Flatten pose + left hand + right hand landmarks into a 258-d vector.
    Missing landmark groups (e.g. a hand out of frame) are zero-filled so
    the output shape is always constant.
    """
    pose = np.zeros(POSE_LANDMARKS * 4, dtype=np.float32)
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()

    left_hand = np.zeros(HAND_LANDMARKS * 3, dtype=np.float32)
    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()

    right_hand = np.zeros(HAND_LANDMARKS * 3, dtype=np.float32)
    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()

    return np.concatenate([pose, left_hand, right_hand])


def landmarks_present(results):
    """(pose_present, left_hand_present, right_hand_present) booleans, for
    per-frame detection-rate diagnostics during extraction.
    """
    return (
        results.pose_landmarks is not None,
        results.left_hand_landmarks is not None,
        results.right_hand_landmarks is not None,
    )


def draw_landmarks(frame_bgr, results):
    mp_drawing.draw_landmarks(
        frame_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    mp_drawing.draw_landmarks(
        frame_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
    )
    mp_drawing.draw_landmarks(
        frame_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
    )
    return frame_bgr


def sample_uniform_indices(total_frames, seq_len):
    """Pick seq_len evenly spaced frame indices out of total_frames.
    Shorter videos are stretched (repeated indices); longer videos are
    subsampled.
    """
    if total_frames <= 0:
        return np.zeros(seq_len, dtype=int)
    return np.linspace(0, total_frames - 1, seq_len).astype(int)
