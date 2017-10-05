import tensorflow as tf

from lrp import lrp_util
from lrp.convolutional_lrp import convolutional
from lrp.linear_lrp import linear
from lrp.max_pooling_lrp import max_pooling
from lrp.nonlinearities_lrp import nonlinearities
from lrp.shaping_lrp import shaping
from lrp.lstm_lrp import lstm


def _lrp(tensor, R):
    # Find the operation that created the tensor
    operation = tensor.op

    if not operation.inputs:
        return R

    operation_type = operation.type

    # Check if the operation is a matrix multiplication or an addition, which means that the current layer is a linear layer
    if operation_type in ['Add', 'BiasAdd']:
        operation_type = lrp_util.addition_associated_with(tensor)

    # Route responsibility to appropriate handler
    if operation_type in ['MatMul']:
        return linear(tensor, R)
    elif operation_type in ['Conv2D']:
        return convolutional(tensor, R)
    elif operation_type in ['ExpandDims', 'Squeeze', 'Reshape']:
        return shaping(tensor, R)
    elif operation_type in ['MaxPool']:
        return max_pooling(tensor, R)
    elif operation_type in ['Relu', 'Sigmoid', 'Tanh']:
        return nonlinearities(tensor, R)
    # elif 'lstm' in operation.type:
    #     return lstm(tensor, R)
    else:
        return _lrp(lrp_util.find_path_towards_input(tensor), R)


def lrp(prediction):
    """
    lrp main function
    :param prediction: Expecting a tensor containing a single prediction [h1]
    :return: Tensor of network input size containing a relevance score for each feature of the input
    """

    # Recurse on each layer with the relevance for the last layer set to the activation of the last layer (following the LRP convention)
    return _lrp(prediction, prediction)
