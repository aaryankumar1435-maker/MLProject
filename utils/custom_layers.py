"""Custom Keras layers shared between letter-model training and inference.

Kept in one module (rather than defined inline in 2_train_letter_model.py)
so 3_realtime_letter_recognition.py can import it too: `tf.keras.models
.load_model` needs every custom layer class registered *before* it
deserializes a saved model that uses one, and Python can't `import` a
module whose filename starts with a digit (2_train_letter_model.py), so a
shared utils module is the only way both scripts see the same registered
classes. Each class is decorated with `register_keras_serializable` so the
registration is keyed by name and survives independently of which script
imported it first.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="isl_letters")
class RandomLandmarkAugment(layers.Layer):
    """Training-only random rotation/scale/translation jitter, applied
    independently to each hand's landmarks.

    Per-hand bounding-box normalization (utils/hand_utils.py) already
    removes scale/position dependence on distance-from-camera, but the
    dataset still only shows each letter at whatever tilt/scale the
    photographer happened to capture. This augments training data with
    small random in-plane rotations, scale jitter, and translation jitter
    so the model sees more pose variation than the raw dataset provides,
    which should improve generalization to webcam hands at inference.
    Inactive whenever training=False (validation/test/real-time
    inference), so it never affects evaluation metrics.
    """

    def __init__(self, num_landmarks_per_hand=21, max_rotation_deg=15.0,
                 scale_jitter=0.08, translate_jitter=0.05, **kwargs):
        super().__init__(**kwargs)
        self.n = num_landmarks_per_hand
        self.max_rot = max_rotation_deg * np.pi / 180.0
        self.scale_jitter = scale_jitter
        self.translate_jitter = translate_jitter

    def call(self, inputs, training=None):
        if not training:
            return inputs
        batch = tf.shape(inputs)[0]
        left, right = inputs[:, :self.n * 3], inputs[:, self.n * 3:]
        return tf.concat([self._augment_hand(left, batch), self._augment_hand(right, batch)], axis=1)

    def _augment_hand(self, hand, batch):
        pts = tf.reshape(hand, (batch, self.n, 3))
        present = tf.reduce_any(tf.not_equal(pts, 0.0), axis=[1, 2], keepdims=True)

        angle = tf.random.uniform((batch, 1), -self.max_rot, self.max_rot)
        cos_a, sin_a = tf.cos(angle), tf.sin(angle)
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        pts = tf.stack([x * cos_a - y * sin_a, x * sin_a + y * cos_a, z], axis=-1)

        scale = tf.random.uniform((batch, 1, 1), 1 - self.scale_jitter, 1 + self.scale_jitter)
        shift = tf.random.uniform((batch, 1, 3), -self.translate_jitter, self.translate_jitter)
        pts = pts * scale + shift

        pts = tf.where(tf.broadcast_to(present, tf.shape(pts)), pts, tf.zeros_like(pts))
        return tf.reshape(pts, (batch, self.n * 3))

    def get_config(self):
        return {**super().get_config(), "num_landmarks_per_hand": self.n,
                "max_rotation_deg": float(self.max_rot * 180.0 / np.pi),
                "scale_jitter": self.scale_jitter, "translate_jitter": self.translate_jitter}


@register_keras_serializable(package="isl_letters")
class AddPositionEmbedding(layers.Layer):
    """Learned per-landmark positional embedding.

    The PointNet baseline is order-invariant by design (global max-pool
    over per-point features), which throws away landmark identity (e.g.
    "this point is the thumb tip") entirely. Self-attention has no
    inherent notion of position either, so without this the transformer
    architecture would be just as blind to which landmark is which. Adding
    a learned embedding per landmark index (the standard transformer
    positional-embedding trick) lets attention condition on landmark
    identity, not just landmark features.
    """

    def build(self, input_shape):
        num_points, dim = input_shape[1], input_shape[2]
        self.pos_emb = self.add_weight(
            name="pos_emb", shape=(1, num_points, dim), initializer="random_normal", trainable=True,
        )

    def call(self, inputs):
        return inputs + self.pos_emb
