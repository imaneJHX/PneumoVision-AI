"""Serializable training metrics shared by experiment models."""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="PneumoVision")
class BinaryF1(tf.keras.metrics.Metric):
    """Thresholded positive-class F1 for epoch-level monitoring."""

    def __init__(self, threshold: float = 0.5, name: str = "f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.bool)
        y_pred = tf.reshape(y_pred, [-1]) >= self.threshold
        weights = None
        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), self.dtype)

        self.true_positives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(y_true, y_pred), self.dtype) * (weights if weights is not None else 1.0))
        )
        self.false_positives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_true), y_pred), self.dtype) * (weights if weights is not None else 1.0))
        )
        self.false_negatives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(y_true, tf.logical_not(y_pred)), self.dtype) * (weights if weights is not None else 1.0))
        )

    def result(self):
        return tf.math.divide_no_nan(
            2.0 * self.true_positives,
            2.0 * self.true_positives + self.false_positives + self.false_negatives,
        )

    def reset_state(self):
        for variable in self.variables:
            variable.assign(0.0)

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}
