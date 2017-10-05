from lrp import lrp, lrp_util
import tensorflow as tf

def nonlinearities(tensor, R):
    """
    Handeling all nonlinearities (Relu, Sigmoid, Tanh) by passing the relevance along
    :param tensor: the tensor of the upper activation of the nonlinearity
    :param R: The upper layer relevance
    :return: lower layer relevance
    """

    assert tensor.shape == R.shape, "Tensor and R should have same shape"

    # Recursively find the relevance of the next layer in the network
    return lrp._lrp(tensor.op.inputs[0], R)


