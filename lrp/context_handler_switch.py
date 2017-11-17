from lrp.constants import CONTEXT_TYPE, LSTM_CONTEXT_TYPE
from lrp.context_handler import ContextHandler
from lrp.lstm_context_handler import LSTMContextHandler
from lrp.standard_context_handler import StandardContextHandler


class ContextHandlerSwitch(ContextHandler):
    """
    The purpose of this class is simply to act as a switch. It forwards the responsibility
    of each context to the right handler.
    """

    # noinspection PyMissingConstructor
    def __init__(self, router):
        # Instantiate an LSTM context handler
        self.lstm_handler = LSTMContextHandler(router)
        # Instantiate a Standard context handler
        self.standard_handler = StandardContextHandler(router)
        # Set current context handler (it is overwritten before it is used the first time)
        self.current_context_handler = self.standard_handler

    def set_context_handler(self, context_handler):
        # Set the current context handler
        self.current_context_handler = context_handler

    def handle_context(self, context):
        # Pass the responsibility on to the right handler
        if context[CONTEXT_TYPE] == LSTM_CONTEXT_TYPE:
            self.lstm_handler.handle_context(context)
        else:
            self.standard_handler.handle_context(context)
