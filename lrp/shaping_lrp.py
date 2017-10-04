from lrp import lrp
import tensorflow as tf

from lrp import lrp_util


def shaping(tensor, R):
    # Assert that shape of tensor, R are the same
    assert tensor.shape == R.shape

    # Reshape R to the same shape as the input to the operation that created tensor
    reshape_operation = tensor.op
    input_to_reshape = reshape_operation.inputs[0]
    R_reshaped = tf.reshape(R, input_to_reshape.shape)

    # Recursively find the relevance of the next layer in the network
    return lrp._lrp(input_to_reshape, R_reshaped)
