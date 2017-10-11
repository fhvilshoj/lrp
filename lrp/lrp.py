import tensorflow as tf

from lrp import lrp_util
from lrp.convolutional_lrp import convolutional
from lrp.linear_lrp import linear
from lrp.max_pooling_lrp import max_pooling
from lrp.nonlinearities_lrp import nonlinearities
from lrp.shaping_lrp import shaping
from lrp.lstm_lrp import lstm

router = {
    'MatMul': linear,
    'Conv2D': convolutional,
    'TensorArrayGatherV3': lstm,
    'MaxPool': max_pooling,
    'ExpandDims': shaping,
    'Squeeze': shaping,
    'Reshape': shaping,
    'Relu': nonlinearities,
    'Sigmoid': nonlinearities,
    'Tanh': nonlinearities
}


def _lrp_routing(path, R):
    # Find the operation that created the tensor

    while path:
        print(path)
        operation_type = path[0].type
        if operation_type in ['Add', 'BiasAdd']:
            # Check which operation a given addition is associated with
            # Note that it cannot be lstm because lstm has its own scope
            operation_type = lrp_util.addition_associated_with(path[0].outputs[0])

        if operation_type in router:
            # Route responsibility to appropriate function
            path, R = router[operation_type](path, R)
        else:
            path = path[1:]
    return R


def _lrp(input, output, R):
    path = lrp_util.get_operations_between_input_and_output(input, output)

    # Recurse on each layer with the relevance for the last layer set to the activation of the last layer (following the LRP convention)
    return _lrp_routing(path, R)


def lrp(input, output):
    """
    lrp main function
    :param input: Expecting a tensor containing a single input causing the output
    :param output: Expecting a tensor containing a single prediction [h1]
    :return: Tensor of input size containing a relevance score for each feature of the input
    """
    return _lrp(input, output, output)
