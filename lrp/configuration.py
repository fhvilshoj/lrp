LINEAR_LAYER = 'Linear'
CONVOLUTIONAL_LAYER = 'Convolution'
MAX_POOL_LAYER = 'MaxPool'
LSTM_LAYER = 'LSTM'
_EMPTY_LAYER = 'Empty'

class LayerConfiguration:
    def __init__(self, layer):
        self._layer = layer

    @property
    def type(self):
        return self._layer


class AlphaBetaConfiguration(LayerConfiguration):
    def __init__(self, layer, alpha=1, beta=0):
        super().__init__(layer)
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
    def __init__(self, layer, epsilon=1e-12):
        super().__init__(layer)
        self._epsilon = epsilon

    @property
    def epsilon(self):
        return self._epsilon


class LRPConfiguration(object):
    def __init__(self):
        self._rules = {
            LINEAR_LAYER: AlphaBetaConfiguration(LINEAR_LAYER),
            CONVOLUTIONAL_LAYER: AlphaBetaConfiguration(CONVOLUTIONAL_LAYER),
            MAX_POOL_LAYER: AlphaBetaConfiguration(MAX_POOL_LAYER),
            LSTM_LAYER: EpsilonConfiguration(LSTM_LAYER)
        }

    def set(self, configuration):
        configuration_type = configuration.type()
        if configuration_type in self._rules:
            self._rules[configuration_type] = configuration
        else:
            raise ValueError("Unknown configuration")

    def get(self, layer):
        if layer in self._rules:
            return self._rules[layer]
        else:
            return LayerConfiguration(_EMPTY_LAYER)
