# Layers
class LAYER:
    ELEMENTWISE_LINEAR = 0
    CONVOLUTIONAL = 1
    SPARSE_LINEAR = 2
    MAX_POOLING = 3
    SOFTMAX = 4
    LINEAR = 5
    EMPTY = 6
    LSTM = 7


# Rules
class RULE:
    WINNERS_TAKE_ALL = 0
    ALPHA_BETA = 1
    EPSILON = 2
    FLAT = 3
    WW = 4


# Bias strategy
class BIAS_STRATEGY:
    IGNORE = 0
    ACTIVE = 1
    NONE = 2
    ALL = 3


class LOG_LEVEL:
    VERBOSE = 0
    NONE = 1


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

    def __str__(self) -> str:
        if self._bias_strategy == BIAS_STRATEGY.ACTIVE:
            bias_strategy = 'ac'
        elif self._bias_strategy == BIAS_STRATEGY.ALL:
            bias_strategy = 'al'
        elif self._bias_strategy == BIAS_STRATEGY.IGNORE:
            bias_strategy = 'ig'
        else:
            bias_strategy = 'no'
        return bias_strategy


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

    def __str__(self) -> str:
        return "a{}b{}_{}".format(self.alpha, self.beta, super().__str__())


class EpsilonConfiguration(LayerConfiguration):
    def __init__(self, epsilon=1e-12, bias_strategy=BIAS_STRATEGY.ALL):
        super().__init__(RULE.EPSILON, bias_strategy)
        self._epsilon = epsilon

    @property
    def epsilon(self):
        return self._epsilon

    def __str__(self) -> str:
        return "e{:.2f}_{}".format(self.epsilon, super().__str__())


class FlatConfiguration(LayerConfiguration):
    def __init__(self):
        super().__init__(RULE.FLAT)

    def __str__(self) -> str:
        return 'flat'


class WWConfiguration(LayerConfiguration):
    def __init__(self):
        super().__init__(RULE.WW)

    def __str__(self) -> str:
        return 'ww'


class BaseConfiguration(LayerConfiguration):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)

    def __str__(self) -> str:
        rule = ""
        if self.type == RULE.WINNERS_TAKE_ALL:
            rule = "win"
        elif self.type == RULE.FLAT:
            rule = "flat"
        else:
            rule = "unknown"
        return rule


class LRPConfiguration(object):
    def __init__(self):
        self._log_level = LOG_LEVEL.NONE
        self._rules = {
            LAYER.LINEAR: AlphaBetaConfiguration(),
            LAYER.SPARSE_LINEAR: AlphaBetaConfiguration(),
            LAYER.ELEMENTWISE_LINEAR: EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.ACTIVE),
            LAYER.CONVOLUTIONAL: AlphaBetaConfiguration(),
            LAYER.LSTM: EpsilonConfiguration(),
            LAYER.MAX_POOLING: LayerConfiguration(RULE.WINNERS_TAKE_ALL),
            LAYER.SOFTMAX: EpsilonConfiguration()
        }

    @property
    def log_level(self):
        return self._log_level

    @log_level.setter
    def log_level(self, level):
        self._log_level = level

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

    def __str__(self) -> str:
        return "LIN_{0}_ELE_{1}_SPA_{2}_CONV_{3}_MAX_{4}_LSTM_{5}_SM_{6}".format(self._rules[LAYER.LINEAR],
                                                                                 self._rules[LAYER.ELEMENTWISE_LINEAR],
                                                                                 self._rules[LAYER.SPARSE_LINEAR],
                                                                                 self._rules[LAYER.CONVOLUTIONAL],
                                                                                 self._rules[LAYER.MAX_POOLING],
                                                                                 self._rules[LAYER.LSTM],
                                                                                 self._rules[LAYER.SOFTMAX])
