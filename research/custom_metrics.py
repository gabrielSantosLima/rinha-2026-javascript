import tensorflow as tf


class BinaryFBetaScore(tf.keras.metrics.FBetaScore):
    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_true.shape) == 1:
            y_true = tf.expand_dims(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)
