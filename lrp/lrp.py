import tensorflow as tf

from lrp import lrp_util
from lrp.convolutional_lrp import convolutional
from lrp.linear_lrp import linear
from lrp.lstm_lrp import lstm
from lrp.max_pooling_lrp import max_pooling

# Internal function that traverses the network layer by layer and applies LRP to each of them
def _lrp(tensor, R):

    # Find the operation that created the tensor
    operation = tensor.op

    # TODO why does this work? Will there not be an operation with an input even if we are in the input layer?
    if not operation.inputs:
        return R

    operation_type = operation.type

    # Check if the operation is a matrix multiplication or an addition, which means that the current layer is a linear layer
    if operation_type in ['Add', 'BiasAdd']:
      operation_type = lrp_util.addition_associated_with(tensor)

    if operation_type in ['MatMul']:
        return linear(tensor, R)
    elif operation_type in ['Conv2D']:
        return convolutional(tensor, R)
    # elif operation_type in ['Expand', 'Squeeze', 'Reshape']:
    #     return convolutional(tensor, R)
    # elif 'lstm' in operation.type:
    #     return lstm(tensor, R)
    # elif 'pool' in operation.type:
    #     return max_pooling(tensor, R)
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
