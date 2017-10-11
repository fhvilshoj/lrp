from lrp import lrp
import tensorflow as tf

from lrp import lrp_util


def shaping(path, R):
    # Assert that shape of tensor, R are the same
    tensor = path[0].outputs[0]
    assert tensor.shape == R.shape

    # Reshape R to the same shape as the input to the reshaping operation that created the tensor
    reshape_operation = path[0]
    # input_to_reshape = reshape_operation.inputs[0]
    input_to_reshape = path[1].outputs[0]
    R_reshaped = tf.reshape(R, input_to_reshape.shape)

    # Return the reshaped relevances
    return path[1:], R_reshaped
