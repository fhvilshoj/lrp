import tensorflow as tf

import lrp.forward_lrp as forward_lrp
from lrp.configuration import LAYER, RULE
from lrp.lrp_util import sum_relevances


def _alpha_beta_softmax(R, current_operation, config):
    alpha = config.alpha
    beta = config.beta

    # Output tensor shape (batch_size, .. , classes)
    output_tensor = current_operation.outputs[0]

    # R_new shape (batch_size, predictions_per_sample, ... , classes)
    R_activator = alpha * R

    # Inhibitor relevance
    classes_size = tf.shape(output_tensor)[-1]
    output_rank = tf.rank(output_tensor)
    upper_dimension_tiles = tf.ones((output_rank,), tf.int32)
    tiles = tf.concat([upper_dimension_tiles, [classes_size]], axis=0)

    output_reshaped = tf.expand_dims(output_tensor, axis=-1)

    # Fractions shape: (batch_size, .. , classes, classes)
    fractions = tf.tile(output_reshaped, tiles)

    R_transposed = tf.transpose(R, tf.concat([[0], tf.range(2, output_rank + 1), [1]], axis=0))

    # R_inhibitor shape (batch_size, .. , classes , predictions_per_sample)
    R_inhibitor = tf.matmul(fractions, R_transposed)

    # R_inhibitor shape (batch_size, predictions_per_sample, .. , classes)
    R_inhibitor = tf.transpose(R_inhibitor, tf.concat([[0, output_rank], tf.range(1, output_rank)], axis=0))

    # Scale R_inhibitor by beta
    R_inhibitor *= beta

    # R_new shape: (batch_size, .. , classes)
    R_new = R_activator + R_inhibitor

    return R_new

def softmax(router, R):
    """
    Handeling softmax layers by passing the relevance along to the input
    :param router: the router object to report changes to
    :param R: the list of tensors containing the relevances from the upper layers
    """
    R_summed = sum_relevances(R)

    current_operation = router.get_current_operation()
    config = router.get_configuration(LAYER.SOFTMAX)

    if config.type == RULE.ALPHA_BETA:
        R_new = _alpha_beta_softmax(R_summed, current_operation, config)
        router.mark_operation_handled(current_operation)
        router.forward_relevance_to_operation(R_new, current_operation, current_operation.inputs[0].op)
    else:
        forward_lrp.forward(router, R)
