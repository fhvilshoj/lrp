from lrp import lrp_util
from lrp.convolutional_lrp import convolutional
from lrp.linear_lrp import linear, element_wise_linear
from lrp.max_pooling_lrp import max_pooling
from lrp.nonlinearities_lrp import nonlinearities
from lrp.shaping_lrp import shaping
from lrp.concatenate_lrp import concatenate
from lrp.split_lrp import split
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
        'ConcatV2': concatenate,
        'Split': split,
        'Relu': nonlinearities,
        'Sigmoid': nonlinearities,
        'Tanh': nonlinearities
    }

    def __init__(self):
        # Placeholders for input and output
        self.input = None
        self.output = None

        # Initialize empty structures one by one.
        # Relevances are used to hold lists of the relevances comming from the
        # upper layers
        self.relevances = []

        # In path indicators holds booleans indicating whether a node with
        # the id equal to the index is in the path list.
        self.in_path_indicators = []

        # Path holds the nodes in the path from output to input, divided into contexts. Each context is either
        # all operations belonging to a LSTM or a part of the computational graph that isn't a LSTM
        self.contexts = []

        # Handled operation holds booleans indicating whether the operation in
        # the path at the same index has been handled already
        self.handled_operations = []

        # Context index indicates which context to consider at this point
        self.current_context_index = 0

        # Path index indicates which operation in the context to consider at this point
        self.current_path_index = 0

        # Remember if there has been added an dimension to the starting point relevances
        self.added_dimension_for_multiple_predictions_per_sample = False

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
        self.in_path_indicators, self.contexts = lrp_util.get_operations_between_output_and_input(input, output)

        self.handled_operations = [False] * (g._last_id + 1)

        # Return the final relevances
        return self._lrp_routing()

    def mark_operation_handled(self, operation):
        self.handled_operations[operation._id] = True

    def forward_relevance_to_operation(self, relevance, relevance_producer, relevance_receiver):
        self.relevances[relevance_receiver._id].append({'producer': relevance_producer._id, 'relevance': relevance})

    def get_current_operation(self):
        # Get the current operation
        current_operation = self.contexts[self.current_context_index][self.current_path_index]
        return current_operation

    def did_add_extra_dimension_for_multiple_predictions_per_sample(self):
        return self.added_dimension_for_multiple_predictions_per_sample

    # Run through the path between output and input and iteratively
    # compute relevances
    def _lrp_routing(self):
        while self.current_context_index < len(self.contexts):
            current_path = self.contexts[self.current_context_index]
            while self.current_path_index < len(current_path):
                current_operation = current_path[self.current_path_index]

                # If the operation has already been taken care of, skip it
                # by jumping to next while iteration
                if self.handled_operations[current_operation._id]:
                    self.current_path_index += 1
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
                self.current_path_index += 1

            self.current_context_index += 1


        # Sum the potentially multiple relevances calculated for the input
        final_input_relevances = lrp_util.sum_relevances(self.relevances[self.input.op._id])

        # If there was added an extra dimension to the starting point relevances,
        # remove it again before returning the calculated relevances
        if self.added_dimension_for_multiple_predictions_per_sample:
            final_input_relevances = tf.squeeze(final_input_relevances, 1)

        return final_input_relevances


    # Helper function that finds the relevance to be used as a starting point for lrp. For each sample
    # in a batch, the function finds the class of interest by either a) taking
    # the class with the largest prediction score or b) using the index of the class of interest if
    # such indexes are provided by the user of the function.
    # Expects class scores as a tensor of shape (batch_size, classes) and optionally a tensor with
    # user chosen indices of shape (batch_size, )

    def _find_starting_point_relevances(self, predictions, user_chosen_indices=None):
            # Get the shape of the predictions
            predictions_shape = predictions.get_shape().as_list()

            # Check if the predictions have the shape (batch_size, predictions_per_sample, number_of_classes) or
            # (batch_size, number_of_classes). In the case of the latter, add the predictions_per_sample dimension
            if len(predictions_shape) == 3:
                batch_size, predictions_per_sample, number_of_classes = predictions_shape
            elif len(predictions_shape) == 2:
                predictions = tf.expand_dims(predictions, 1)
                batch_size, predictions_per_sample, number_of_classes = predictions.get_shape().as_list()
                # Remember that there has been added an extra dimension, so it can be removed again later
                self.added_dimension_for_multiple_predictions_per_sample = True
            else:
                raise ValueError("Only accepts outputs of shape (batch_size, predictions_per_sample, number_of_classes) "
                                 "or (batch_size, number_of_classes) but got shape: " + predictions_shape)


            # If the user has provided the indexes of the number_of_classes of interest, use those. If not, find
            # the indexes by finding the class with the largest prediction score for each sample
            max_score_indices = user_chosen_indices if user_chosen_indices else tf.argmax(predictions, axis=2)

            # Create a tensor that for each sample has a one at the position of the class of interest and
            # zeros in all other positions
            max_score_indices_as_one_hot_vectors = tf.one_hot(max_score_indices, number_of_classes)

            # Create and return the relevance tensor, where all relevances except the one for the class of interest
            # for each sample have been set to zero
            relevances_with_only_chosen_class = tf.multiply(predictions, max_score_indices_as_one_hot_vectors)

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
