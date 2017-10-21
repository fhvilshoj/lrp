EPSILON = 1e-12
BIAS_DELTA = 1  # (0 if we shouldn't consider bias in epsilon rule) 1 if we should

# KEYS FOR CONTEXT DICTIONARIES
CONTEXT_TYPE = "context_type"
CONTEXT_PATH = "path"
LSTM_BEGIN_TRANSPOSE_OPERATION = 'lstm_begin_transpose_operation'
LSTM_INPUT_OPERATION = 'lstm_input_operation'
EXTRA_CONTEXT_INFORMATION = 'extra_context_information'

# CONTEXT TYPES
NON_LSTM_CONTEXT_TYPE = "non-LSTM"
LSTM_CONTEXT_TYPE = "LSTM"


# KEYS FOR RELEVANCE DICTIONARIES
RELEVANCE_PRODUCER = 'producer'
RELEVANCE = 'relevance'