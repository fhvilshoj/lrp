from lrp import lrp
import tensorflow as tf

from lrp import lrp_util


def linear(tensor, R):
    """
    linear lrp
    :param tensor: tensor should be the activation (before the none linearity)
    :param R: the tensor containing the relevance from the upper layer
    :return: the relevance distributed on the lower layer
    """
    # Tensor shape: (1, upper layer size)
    matmultensor = tensor
    with_bias = False
    if tensor.op.type == 'Add':
        matmultensor = lrp_util.find_first_tensor_from_type(tensor, 'MatMul')
        with_bias = True

    (inp1, inp2) = matmultensor.op.inputs
    inp2 = tf.transpose(inp2)
    zs = tf.multiply(inp1, inp2)

    # take positive part of zs
    zp = lrp_util.replace_negatives_with_zeros(zs)

    zp_sum = tf.expand_dims(tf.reduce_sum(zp, axis=1), axis=-1)

    bias_positive = tf.zeros_like(zp_sum)
    if with_bias:
        # Find bias tensor if using bias
        if tensor.op.inputs[0].op.type in ['Identity', 'Const']:
            bias = tensor.op.inputs[0]
        else:
            bias = tensor.op.inputs[1]
        bias_positive = tf.transpose(lrp_util.replace_negatives_with_zeros(bias))

    zp_sum_with_bias = tf.add(zp_sum, bias_positive)
    div = tf.divide(zp, zp_sum_with_bias)
    R_new = tf.matmul(R, div)

    R_new = tf.Print(R_new, [R, zp, zp_sum_with_bias])

    return lrp._lrp(lrp_util.find_path_towards_input(matmultensor), R_new)
