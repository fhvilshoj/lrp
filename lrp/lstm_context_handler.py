from context_handler import ContextHandler
from lstm_lrp import lstm
import lrp_util
from constants import *


class LSTMContextHandler(ContextHandler):
    def __init__(self, router):
        super().__init__(router)

    def handle_context(self, context):
        path_ = context[CONTEXT_PATH]

        # Get the relevances to move through the LSTM
        R = self.router.get_relevance_for_operation(path_[0])

        # Sum the potentially multiple relevances from the upper layers
        R = lrp_util.sum_relevances(R)

        # Get the path containing all operations in the LSTM
        path = context[CONTEXT_PATH]

        # Get the extra information related to the LSTM context
        extra_context_information = context[EXTRA_CONTEXT_INFORMATION]

        # Get the transpose operation that marks the beginning of the LSTM
        transpose_operation = extra_context_information[LSTM_BEGIN_TRANSPOSE_OPERATION]

        # Get the operation that produces the input to the LSTM (i.e. the operation right before
        # the transpose that marks the start of the LSTM)
        input_operation = extra_context_information[LSTM_INPUT_OPERATION]

        # Get the tensor that is the input to the LSTM (i.e. the input to the transpose operation
        # that marks the start of the LSTM)
        LSTM_input = transpose_operation.inputs[0]


        # TODO use configuration to set alpha_beta or epsilon rule
        # Calculate the relevances to distribute to the lower layers
        R_new = lstm(path, R, LSTM_input)

        # TODO this call should be done in the LSTM. Not in the context handler.
        # Mark all operations belonging to the LSTM as "handled"
        for op in path:
            self.router.mark_operation_handled(op)

        # TODO this call should be done in the LSTM. Not in the context handler.
        # Forward the relevances to the lower layers
        self.router.forward_relevance_to_operation(R_new,
                                                   transpose_operation,
                                                   input_operation)
