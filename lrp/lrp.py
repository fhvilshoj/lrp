from lrp import lrp_util
from lrp.convolutional_lrp import convolutional
from lrp.linear_lrp import linear, element_wise_linear
from lrp.max_pooling_lrp import max_pooling
from lrp.nonlinearities_lrp import nonlinearities
from lrp.shaping_lrp import shaping
from lrp.lstm_lrp import lstm
import tensorflow as tf


class _LRPImplementation:

    _router = {
        'MatMul': linear,
        'Mul': element_wise_linear,
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

    def __init__(self):
        # Placeholders for input and output
        self.input = None
        self.output = None

        # Initialize empty structures

        # Relevances are used to hold lists of the relevances comming from the
        # upper layers
        self.relevances = []

        # In path indicators holds booleans indicating whether a node with
        # the id equal to the index is in the path list.
        self.in_path_indicators = []

        # Path holds the nodes in the path from output to input
        self.path = []

        # Handled operation holds booleans indicating whether the operation in
        # the path at the same index has been handled already
        self.handled_operations = []

        # Path index indicates which node in the path to consider at this point
        self.path_index = 0

    def lrp(self, input, output, R = None):
        # Remember input and output
        self.input = input
        self.output = output

        # Find relevance to distribute from the output if the relevance tensor
        # is not already defined
        if R is None:
            R = self._find_starting_point_relevances(output)

        # Fill structures
        g = output.op.graph
        self.relevances = [[] for _ in range(g._last_id + 1)]
        self.relevances[output.op._id].append({'producer': output.op._id, 'relevance': R})

        # Find the path between output and input and remember it
        self.in_path_indicators, self.path = lrp_util.get_operations_between_output_and_input(input, output)

        self.handled_operations = [False] * (g._last_id + 1)

        # Return the final relevances
        return self._lrp_routing()

    def mark_operation_handled(self, operation):
        self.handled_operations[operation._id] = True

    def forward_relevance_to_operation(self, relevance, relevance_producer, relevance_receiver):
        self.relevances[relevance_receiver._id].append({'producer': relevance_producer._id, 'relevance': relevance})

    def get_current_operation(self):
        current_operation = self.path[self.path_index]
        return current_operation

    # Run through the path between output and input and iteratively
    # compute relevances
    def _lrp_routing(self):
        while self.path_index < len(self.path):
            current_operation = self.path[self.path_index]

            # If the operation has already been taken care of, skip it
            # by jumping to next while iteration
            if self.handled_operations[current_operation._id]:
                self.path_index += 1
                continue

            # Find type of the operation in the front of the path
            operation_type = current_operation.type
            if operation_type in ['Add', 'BiasAdd']:
                # Check which operation a given addition is associated with
                # Note that it cannot be lstm because lstm has its own scope
                operation_type = lrp_util.addition_associated_with(current_operation.outputs[0])

            if operation_type in self._router:
                # Route responsibility to appropriate function
                # Send the recorded relevance for the current operation
                # along. This saves the confusion of finding relevances
                # for Add in the concrete implementations
                self._router[operation_type](self, self.relevances[current_operation._id])
            else:
                print("Router did not know the operation: ", operation_type)
                # If we don't know the operation, skip it

            # Go to next operation in path
            self.path_index += 1

        # Sum the potentially multiple relevances calculated for the input
        final_input_relevances = lrp_util.sum_relevances(self.relevances[self.input.op._id])

        return final_input_relevances


    # Helper function that finds the relevance to be used as a starting point for lrp. For each sample
    # in a batch, the function finds the class of interest by either a) taking
    # the class with the largest prediction score or b) using the index of the class of interest if
    # such indexes are provided by the user of the function.
    # Expects class scores as a tensor of shape (batch_size, classes) and optionally a tensor with
    # user chosen indices of shape (batch_size, )

    #TODO is it a fair constraint to only accept class scores of shape (batch_size, classes) and reject FX (batch_size, 1, classes)?
    def _find_starting_point_relevances(self, class_scores, user_chosen_indices=None):
            # Get the shape of the class scores
            number_of_classes = class_scores.get_shape().as_list()[-1]

            # If the user has provided the indexes of the classes of interest, use those. If not, find
            # the indexes by finding the class with the largest prediction score for each sample
            max_score_indices = user_chosen_indices if user_chosen_indices else tf.argmax(class_scores, axis=-1)

            # Create a tensor that for each sample has a one at the position of the class of interest and
            # zeros in all other positions
            max_score_indices_as_one_hot_vectors = tf.one_hot(max_score_indices, number_of_classes)

            # Create and return the relevance tensor, where all relevances except the one for the class of interest
            # for each sample have been set to zero
            relevances_with_only_chosen_class = tf.multiply(max_score_indices_as_one_hot_vectors, class_scores)

            return relevances_with_only_chosen_class


# The purpose of this method is to have a handle for test cases where
# the relevence is predefined
def _lrp(input, output, R = None):
    # Instantiate a LRP object
    impl = _LRPImplementation()
    # Return the relevances computed from the object
    return impl.lrp(input, output, R)


def lrp(input, output):
    """
    lrp main function
    :param input: Expecting a tensor containing a single input causing the output
    :param output: Expecting the output tensor to begin lrp from
    :return: Tensor of input size containing a relevance score for each feature of the input
    """
    return _lrp(input, output)
