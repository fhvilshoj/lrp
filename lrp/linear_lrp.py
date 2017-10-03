from lrp import lrp
import tensorflow as tf

from lrp import lrp_util


def linear(tensor, R):
    """
    linear lrp
    :param tensor: tensor should be the activation (i.e. the output of the linear layer before an eventual non-linearity)
    :param R: the tensor containing the relevance from the upper layer
    :return: lower layer relevance (i.e. relevance distributed to the input to the linear layer)
    """
    # Start by assuming the activation tensor is the output of a matrix multiplication (i.e. not an addition with a bias)
    # Tensor shape: (1, upper layer size)
    matmultensor = tensor
    with_bias = False

    # If the activation tensor is the output of an addition (i.e. the above assumption does not hold), move through the graph to find the output of the nearest matrix multiplication.
    if tensor.op.type == 'Add':
        matmultensor = lrp_util.find_first_tensor_from_type(tensor, 'MatMul')
        with_bias = True

    # Find the inputs to the matrix multiplication, transpose the weights and perform elementwise multiplication to find the z_ij's
    (inp1, inp2) = matmultensor.op.inputs
    inp2 = tf.transpose(inp2)
    zs = tf.multiply(inp1, inp2)

    # Replace the negative elements with zeroes to only have the positive z's left (i.e. z_ij^+)
    zp = lrp_util.replace_negatives_with_zeros(zs)

    # Take the sum of each row of z_ij^+'s
    zp_sum = tf.reduce_sum(zp, axis=1, keep_dims=True)

    # Find the positive parts of an eventual bias that is added to the results of the matrix multiplication (i.e. b^+).
    # Use zeroes if there is no bias addition after the matrix multiplication.
    bias_positive = tf.zeros_like(zp_sum)
    if with_bias:
        bias = lrp_util.get_input_bias_from_add(tensor)

        # Replace the negative elements in the bias with zeroes and transpose to the right form
        bias_positive = tf.transpose(lrp_util.replace_negatives_with_zeros(bias))

    # Add the sum of the z_ij^+'s and the positive bias (i.e. find the z_j^+'s)
    zp_sum_with_bias = tf.add(zp_sum, bias_positive)

    # Calculate the lower layer relevances (a combination of equation 60 and 62 in Bach 2015)
    R_new = tf.matmul(R, tf.divide(zp, zp_sum_with_bias))

    # Recursively find the relevance of the next layer in the network
    return lrp._lrp(lrp_util.find_path_towards_input(matmultensor), R_new)
