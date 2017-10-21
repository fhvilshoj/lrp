from context_handler import ContextHandler
from lstm_lrp import lstm
from constants import CONTEXT_PATH

class LSTMContextHandler(ContextHandler):
    def __init__(self, router):
        super().__init__(router)

    def handle_context(self, context):
        path_ = context[CONTEXT_PATH]
        lstm(self, context, self.router.get_relevance_for_operation(path_[0]))
