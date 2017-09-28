import tensorflow as tf

from lrp import lrp_util
from lrp.convolutional_lrp import convolutional
from lrp.linear_lrp import linear
from lrp.lstm_lrp import lstm
from lrp.max_pooling_lrp import max_pooling


def _lrp(tensor, R):
    operation = tensor.op

    if not operation.inputs:
        return R

    if operation.type in ['MatMul', 'Add']:
        return linear(tensor, R)
    elif 'conv' in operation.type:
        return convolutional(tensor, R)
    elif 'lstm' in operation.type:
        return lstm(tensor, R)
    elif 'pool' in operation.type:
        return max_pooling(tensor, R)
    else:
        return _lrp(lrp_util.find_path_towards_input(tensor), R)


def lrp(prediction):
    """
    lrp main function
    :param prediction: Expecting Tensor of a single prediction [h1]
    :return: Tensor of network input size for distributed relevance
    """

    # Recurse on each layer
    return _lrp(prediction, prediction)
