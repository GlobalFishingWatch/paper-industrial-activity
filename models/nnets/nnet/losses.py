import numpy as np
import keras.backend as K
import keras.callbacks
import tensorflow as tf
import tensorflow_probability as tfp


def evidential_loss_orig(y_true, evidence_pred, lambda_t, weights=None):
    """Evidential multiclass loss function.

    evidence_pred (y_pred) is produced by softplus or relu, then

    Class score (array) = alpha / S
    Uncertainty (scalar) = y_true.shape[-1] / S

    with alpha = evidence + 1, and S = sum(alpha)

    paper: https://tinyurl.com/2p86y5ur

    See also
    --------
    add_evidence_block()

    """
    # First compute the base loss L_i (Eq 5, interpretable form)
    assert evidence_pred.shape[-1] == y_true.shape[-1]

    if not weights:
        weights = tf.ones_like(y_true)
    else:
        assert weights.shape == y_true.shape

    # For 2 classes: y_true = true|false (single value)
    # one-hot encodes two classes: [true, false]
    # y_true = tf.expand_dims(y_true, axis=3)  
    # y = tf.concat([1 - y_true, y_true], axis=-1)

    y = y_true
    alpha = evidence_pred + 1
    S = tf.reduce_sum(alpha, axis=-1, keepdims=True)
    p = alpha / S

    arg = (y - p) ** 2 + p * (1 - p) / (S + 1)
    core = tf.reduce_sum(arg, axis=-1)

    alpha_hat = y + (1 - y) * alpha
    Dahat = tfp.distributions.Dirichlet(alpha_hat)
    Dunif = tfp.distributions.Dirichlet(tf.ones_like(alpha_hat))
    reg = Dahat.kl_divergence(Dunif)

    return K.sum(weights * (core + lambda_t * reg), axis=(1, 2))


def evidential_loss(y_true, class_score, lambda_t):
    """Evidential multiclass loss function.

    To be used with `add_evidence_block()` that returns
    class_score (y_pred) instead of evidence, so it works
    backwards to get the evidence from class_score.

    evidence is produced by softplus or relu, then

    class_score (array) = alpha / S
    uncertainty (scalar) = y_true.shape[-1] / S

    with alpha = evidence + 1, and S = sum(alpha)

    paper: https://tinyurl.com/2p86y5ur

    See:
        OutputBlock
        add_evidence_block
        evidential_loss_orig
    """
    assert class_score.shape[-1] == y_true.shape[-1]

    y = y_true
    p = class_score

    uncertainty = 1 - tf.reduce_sum(p, axis=-1, keepdims=True)
    S = y_true.shape[-1] / uncertainty
    alpha = S * p
    # evidence = alpha - 1

    arg = (y - p) ** 2 + p * (1 - p) / (S + 1)
    core = tf.reduce_sum(arg, axis=-1, keepdims=False)

    alpha_hat = y + (1 - y) * alpha
    Dahat = tfp.distributions.Dirichlet(alpha_hat)
    Dunif = tfp.distributions.Dirichlet(tf.ones_like(alpha_hat))
    reg = Dahat.kl_divergence(Dunif)

    return core + lambda_t * reg
