import tensorflow as tf


def shaping(path, R):
    # Reshape R to the same shape as the input to the
    # reshaping operation that created the tensor
    # input_to_reshape = reshape_operation.inputs[0]
    input_to_reshape = path[1].outputs[0]
    R_reshaped = tf.reshape(R, tf.shape(input_to_reshape))

    # Return the reshaped relevances
    return path[1:], R_reshaped
