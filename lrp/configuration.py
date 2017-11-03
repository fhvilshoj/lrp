from enum import Enum

# Layer types
LINEAR_LAYER = 'Linear'
CONVOLUTIONAL_LAYER = 'Convolution'
MAX_POOL_LAYER = 'MaxPool'
LSTM_LAYER = 'LSTM'
_EMPTY_LAYER = 'Empty'

# Layers
class LAYER:
    CONVOLUTIONAL = 0
    MAX_POOL = 1
    LINEAR = 2
    EMPTY = 3
    LSTM = 4


# Rules
class RULE:
    ALPHA_BETA = 0
    EPSILON = 1
    FLAT = 2
    WW = 3

# Bias strategy
class BIAS_STRATEGY:
    ACTIVE = 0
    NONE = 1
    ALL = 2

class LayerConfiguration:
    def __init__(self, layer, bias_strategy=BIAS_STRATEGY.NONE):
        self._layer = layer
        self._bias_strategy = bias_strategy

    @property
    def type(self):
        return self._layer

    @property
    def bias_strategy(self):
        return self._bias_strategy


class AlphaBetaConfiguration(LayerConfiguration):
    def __init__(self, alpha=1, beta=0, bias_strategy=BIAS_STRATEGY.NONE):
        super().__init__(RULE.ALPHA_BETA, bias_strategy)
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
    def __init__(self, epsilon=1e-12, bias_strategy=BIAS_STRATEGY.ALL):
        super().__init__(RULE.EPSILON, bias_strategy)
        self._epsilon = epsilon

    @property
    def epsilon(self):
        return self._epsilon


class FlatConfiguration(LayerConfiguration):
    def __init__(self):
        super().__init__(RULE.FLAT)


class WWConfiguration(LayerConfiguration):
    def __init__(self):
        super().__init__(RULE.WW)


class LRPConfiguration(object):
    def __init__(self):
        self._rules = {
            LAYER.LINEAR: AlphaBetaConfiguration(),
            LAYER.CONVOLUTIONAL: AlphaBetaConfiguration(),
            LAYER.MAX_POOL: AlphaBetaConfiguration(),
            LAYER.LSTM: EpsilonConfiguration()
        }

    def set(self, layer_type, configuration):
        if layer_type in self._rules:
            self._rules[layer_type] = configuration
        else:
            raise ValueError("Unknown configuration " + layer_type)

    def get(self, layer):
        if layer in self._rules:
            return self._rules[layer]
        else:
            return LayerConfiguration(LAYER.EMPTY)
