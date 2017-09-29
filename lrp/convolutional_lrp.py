from lrp import lrp
import tensorflow as tf


def convolutional(tensor, R):
    """
    Convolutional lrp
    :param tensor: the tensor of the upper activation (the output of the convolution)
    :param R: The upper layer relevance
    :return: lower layer relevance (i.e. relevance distributed to the input to the convolution)
    """
    raise NotImplemented('Convolutional is not yet implemented')
