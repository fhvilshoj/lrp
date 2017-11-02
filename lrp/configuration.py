# Layer types
LINEAR_LAYER = 'Linear'
CONVOLUTIONAL_LAYER = 'Convolution'
MAX_POOL_LAYER = 'MaxPool'
LSTM_LAYER = 'LSTM'
_EMPTY_LAYER = 'Empty'

# Rules
ALPHA_BETA_RULE = 'Alpha_Beta_Rule'
EPSILON_RULE = 'Epsilon_Rule'
FLAT_RULE = 'Flat_Rule'
WW_RULE = 'WW_Rule'

class LayerConfiguration:
    def __init__(self, layer):
        self._layer = layer

    @property
    def type(self):
        return self._layer


class AlphaBetaConfiguration(LayerConfiguration):
    def __init__(self, alpha=1, beta=0):
        super().__init__(ALPHA_BETA_RULE)
        assert alpha + beta == 1, "alpha + beta should be 1"
        self._alpha = alpha
        self._beta = beta

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta


class EpsilonConfiguration(LayerConfiguration):
    def __init__(self, epsilon=1e-12):
        super().__init__(EPSILON_RULE)
        self._epsilon = epsilon

    @property
    def epsilon(self):
        return self._epsilon


class FlatConfiguration(LayerConfiguration):
    def __init__(self):
        super().__init__(FLAT_RULE)


class WWConfiguration(LayerConfiguration):
    def __init__(self):
        super().__init__(WW_RULE)


class LRPConfiguration(object):
    def __init__(self):
        self._rules = {
            LINEAR_LAYER: AlphaBetaConfiguration(),
            CONVOLUTIONAL_LAYER: AlphaBetaConfiguration(),
            MAX_POOL_LAYER: AlphaBetaConfiguration(),
            LSTM_LAYER: EpsilonConfiguration()
        }

    def set(self, layer_type, configuration):
        if layer_type in self._rules:
            self._rules[layer_type] = configuration
        else:
            raise ValueError("Unknown configuration")

    def get(self, layer):
        if layer in self._rules:
            return self._rules[layer]
        else:
            return LayerConfiguration(_EMPTY_LAYER)
