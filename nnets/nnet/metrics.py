import keras.metrics as km
import tensorflow as tf


def categorical_accuracy(y_true, class_score):
    """Standard accuracy for class_score output.

    Unwraps class_score (model output) -> evidence.

    To be used with `add_evidence_block()` that returns
    class_score (y_pred) instead of evidence, so it works
    backwards to get the evidence from class_score.
    """
    assert class_score.shape[-1] == y_true.shape[-1]
    uncertainty = 1 - tf.reduce_sum(class_score, axis=-1, keepdims=True)
    S = y_true.shape[-1] / uncertainty
    alpha = S * class_score
    evidence = alpha - 1
    return km.categorical_accuracy(y_true, evidence)
