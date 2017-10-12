from lrp import lrp_util
from lrp.convolutional_lrp import convolutional
from lrp.linear_lrp import linear
from lrp.max_pooling_lrp import max_pooling
from lrp.nonlinearities_lrp import nonlinearities
from lrp.shaping_lrp import shaping
from lrp.lstm_lrp import lstm
import tensorflow as tf

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

    # Recurse on each layer with the relevance for the last layer set to the
    # activation of the last layer (following the LRP convention)
    return _lrp_routing(path, R)


def lrp(input, output):
    """
    lrp main function
    :param input: Expecting a tensor containing a single input causing the output
    :param output: Expecting the output to begin lrp from
    :return: Tensor of input size containing a relevance score for each feature of the input
    """

    relevances_to_use_as_starting_point = _find_starting_point_relevances(output)

    return _lrp(input, output, relevances_to_use_as_starting_point)

# Helper function that finds the relevance to be used as a starting point for lrp. For each sample
# in a batch, the function finds the class of interest by either a) taking
# the class with the largest prediction score or b) using the index of the class of interest if
# such indexes are provided by the user of the function.
# Expects class scores as a tensor of shape (batch_size, classes) and optionally a tensor with
# user chosen indices of shape (batch_size, )
def _find_starting_point_relevances(class_scores, user_chosen_indices=None):
        # Get the shape of the class scores
        batch_size, number_of_classes = class_scores.get_shape().as_list()

        # If the user has provided the indexes of the classes of interest, use those. If not, find
        # the indexes by finding the class with the largest prediction score for each sample
        max_score_indices = user_chosen_indices if user_chosen_indices else tf.argmax(class_scores, axis=1)

        # Create a tensor that for each sample has a one at the position of the class of interest and
        # zeros in all other positions
        max_score_indices_as_one_hot_vectors = tf.one_hot(max_score_indices, number_of_classes)

        # Create and return the relevance tensor, where all relevances except the one for the class of interest
        # for each sample have been set to zero
        relevances_with_only_chosen_class = tf.multiply(max_score_indices_as_one_hot_vectors, class_scores)

        return relevances_with_only_chosen_class
