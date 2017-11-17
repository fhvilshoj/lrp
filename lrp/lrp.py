from lrp import lrp_util
from lrp.configuration import LRPConfiguration, LOG_LEVEL
from lrp.context_handler_switch import ContextHandlerSwitch
from lrp.constants import *

import tensorflow as tf

class _LRPImplementation:
    def __init__(self):
        # Placeholders for input and output
        self._input = None
        self._output = None

        # Initialize default configuration (alpha=1, beta=0 for all layers but LSTM: epsilon=1e-12)
        self._configuration = LRPConfiguration()

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

        # Remember if there has been added an dimension to the starting point relevances
        self.starting_point_relevances_had_predictions_per_sample_dimension = True

    def should_log(self):
        return self._configuration.log_level == LOG_LEVEL.VERBOSE

    def lrp(self, input, output, configuration=None, R=None):
        # Remember input and output
        self._input = input
        self._output = output

        if configuration is not None:
            self._configuration = configuration

        # Find relevance to distribute from the output if the relevance tensor
        # is not already defined
        if R is None:
            R = self._find_starting_point_relevances(output)

        # Fill structures
        g = output.op.graph
        self.relevances = [[] for _ in range(g._last_id + 1)]
        self.relevances[output.op._id].append({RELEVANCE_PRODUCER: output.op._id, RELEVANCE: R})

        # Find the path between output and input and remember it
        self.in_path_indicators, self.contexts = lrp_util.get_operations_between_output_and_input(input, output)

        self.handled_operations = [False] * (g._last_id + 1)

        # Return the final relevances
        return self._lrp_routing()

    def mark_operation_handled(self, operation):
        self.handled_operations[operation._id] = True

    def is_operation_handled(self, operation):
        return self.handled_operations[operation._id]

    def forward_relevance_to_operation(self, relevance, relevance_producer, relevance_receiver):
        if self.should_log():
            message = "\nRelevance sum from {} to {}: \n".format(relevance_producer.type, relevance_receiver.type)
            if isinstance(relevance, tf.SparseTensor):
                values = tf.Print(relevance.values, [relevance_producer._id, relevance_receiver._id, tf.sparse_reduce_sum(relevance)], message)
                relevance = tf.SparseTensor(relevance.indices, values, relevance.dense_shape)
            else :
                relevance = tf.Print(relevance, [relevance_producer._id, relevance_receiver._id, tf.reduce_sum(relevance)],
                                     message)

        self.relevances[relevance_receiver._id].append(
            {RELEVANCE_PRODUCER: relevance_producer._id, RELEVANCE: relevance})

    def get_relevance_for_operation(self, operation):
        return self.relevances[operation._id]

    def get_configuration(self, layer):
        return self._configuration.get(layer)

    # Run through the path between output and input and iteratively
    # compute relevances
    def _lrp_routing(self):
        # Create context handler switch which is used to forward the responsibility of the
        # different types of contexts to the appropriate handlers
        context_switch = ContextHandlerSwitch(self)

        # Handle each context separately by routing the context through the context switch
        for current_context in self.contexts:
            context_switch.handle_context(current_context)

        # Sum the potentially multiple relevances calculated for the input
        final_input_relevances = lrp_util.sum_relevances(self.relevances[self._input.op._id])

        # If the starting point relevances were shape (batch_size, classes), remove the extra
        # predictions_per_sample dimension that was added to the starting point relevances
        if not self.starting_point_relevances_had_predictions_per_sample_dimension:
            # Check if the relevances are sparse, in which case we need to use tf's sparse reshape operation
            # to remove the extra dimension
            if isinstance(final_input_relevances, tf.SparseTensor):
                # Get the shape of the final relevances
                final_input_relevances_shape = tf.shape(final_input_relevances)
                # Extract the batch_size dimension
                batch_size = tf.slice(final_input_relevances_shape, [0], [1])
                # Extract all the dimensions after the predictions_per_sample dimension
                sample_dimensions = tf.slice(final_input_relevances_shape, [2], [-1])
                # Create the new shape of the relevances, i.e. the shape where the predictions_per_sample
                # has been removed
                final_input_relevances_new_shape = tf.concat([batch_size, sample_dimensions], 0)
                # Remove the predictions_per_sample dimension
                final_input_relevances = tf.sparse_reshape(final_input_relevances, final_input_relevances_new_shape)
            # If the relevances are not sparse, i.e. they are dense, we can just squeeze the extra dimension
            else:
                final_input_relevances = tf.squeeze(final_input_relevances, 1)

        return final_input_relevances

    # Helper function that finds the relevance to be used as a starting point for lrp. For each sample
    # in a batch, the function finds the class of interest by either a) taking
    # the class with the largest prediction score or b) using the index of the class of interest if
    # such indexes are provided by the user of the function.
    # Expects class scores as a tensor of shape (batch_size, classes) and optionally a tensor with
    # user chosen indices of shape (batch_size, )

    def _find_starting_point_relevances(self, predictions, user_chosen_indices=None):
        # Get the shape of the prediction
        predictions_shape = tf.shape(predictions)

        has_pred_per_sample_dim = True

        # Check if the predictions have the shape (batch_size, predictions_per_sample, number_of_classes) or
        # (batch_size, number_of_classes). In the case of the latter, set the predictions_per_sample dimension
        # equal to one
        predictions_rank = len(predictions.get_shape())
        if predictions_rank == 3:
            predictions_per_sample = predictions_shape[1]
            number_of_classes = predictions_shape[2]
        elif predictions_rank == 2:
            predictions_per_sample = 1
            # Remember that the starting point relevances did not have a 'predictions_per_sample' dimension
            self.starting_point_relevances_had_predictions_per_sample_dimension = False
            number_of_classes = predictions_shape[1]
        else:
            raise ValueError("Only accepts outputs of shape (batch_size, predictions_per_sample, number_of_classes) "
                             "or (batch_size, number_of_classes)")

        # If the user has provided the indexes of the number_of_classes of interest, use those. If not, find
        # the indexes by finding the class with the largest prediction score for each sample
        max_score_indices = user_chosen_indices if user_chosen_indices else tf.argmax(predictions, axis=-1)

        # Create a tensor that for each sample has a one at the position of the class of interest and
        # zeros in all other positions
        max_score_indices_as_one_hot_vectors = tf.one_hot(max_score_indices, number_of_classes)

        # Create and return the relevance tensor, where all relevances except the one for the class of interest
        # for each sample have been set to zero
        relevances_with_only_chosen_class = tf.multiply(predictions, max_score_indices_as_one_hot_vectors)

        # Add an extra dimension for predictions_per_sample, so we can calculate 'predictions_per_sample'
        # independent paths through the network.
        # New shape: (predictions_per_sample, batch_size, predictions_per_sample, classes)
        relevances_with_only_chosen_class = tf.expand_dims(relevances_with_only_chosen_class, 0)

        # Stack 'predictions_per_sample' identical copies of the starting relevances on top of each other
        if self.starting_point_relevances_had_predictions_per_sample_dimension:
            stacked_relevances = tf.tile(relevances_with_only_chosen_class, [predictions_per_sample, 1, 1, 1])
            # Create a tensor of shape (predictions_per_sample, 1, predictions_per_sample, 1) that has ones in position
            # [i, 1, i, 1] and zeros everywhere else
            selections = tf.expand_dims(tf.expand_dims(tf.eye(predictions_per_sample), -1), 1)

            # For each prediction per sample, "select" the associated relevance and set all other relevances to zero
            relevances_new = tf.multiply(selections, stacked_relevances)

            # Transpose the starting point relevances to get the final shape
            # (batch_size, predictions_per_sample, predictions_per_sample, classes)
            relevances_new = tf.transpose(relevances_new, [1, 0, 2, 3])
        else:
            # Transpose the starting point relevances to get the final shape
            # (batch_size, predictions_per_sample=1, classes)
            relevances_new = tf.transpose(relevances_with_only_chosen_class, [1, 0, 2])

        return relevances_new


# The purpose of this method is to have a handle for test cases where
# the relevence is predefined
def _lrp(input, output, configuration, R=None):
    # Instantiate a LRP object
    impl = _LRPImplementation()
    # Return the relevances computed from the object
    return impl.lrp(input, output, configuration, R)


def lrp(input, output, configuration=None):
    """
    lrp main function
    :param input: Expecting a tensor containing a single input causing the output
    :param output: Expecting the output tensor to begin lrp from
    :param configuration: Expecting LRPConfiguration object
    :return: Tensor of input size containing a relevance score for each feature of the input
    """
    return _lrp(input, output, configuration)
