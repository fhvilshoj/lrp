from lrp.configuration import *

linear_configurations = [
    AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.ACTIVE),
    # AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.ALL),
    # AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.NONE),
    # AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.ACTIVE),
    AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.ALL),
    # AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.NONE),
    # EpsilonConfiguration(1e-3, BIAS_STRATEGY.ALL),
    # EpsilonConfiguration(1e-3, BIAS_STRATEGY.ACTIVE),
    EpsilonConfiguration(1e-3, BIAS_STRATEGY.NONE),
    # EpsilonConfiguration(100, BIAS_STRATEGY.ALL),
    # EpsilonConfiguration(100, BIAS_STRATEGY.ACTIVE),
    # EpsilonConfiguration(100, BIAS_STRATEGY.NONE),
    WWConfiguration()
]
elementwise_linear_configurations = [
    # AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.ACTIVE),
    # AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.NONE),
    # AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.ACTIVE),
    # AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.NONE),
    EpsilonConfiguration(1e-3, BIAS_STRATEGY.ACTIVE),
    # EpsilonConfiguration(1e-3, BIAS_STRATEGY.NONE),
    # EpsilonConfiguration(100, BIAS_STRATEGY.ACTIVE),
    # EpsilonConfiguration(100, BIAS_STRATEGY.NONE),
]
sparse_linear_configurations = [
    AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.ACTIVE),
    # AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.NONE),
    AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.ACTIVE),
    # AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.NONE),
    EpsilonConfiguration(1e-3, BIAS_STRATEGY.ACTIVE),
    # EpsilonConfiguration(1e-3, BIAS_STRATEGY.NONE),
    # EpsilonConfiguration(100, BIAS_STRATEGY.ACTIVE),
    # EpsilonConfiguration(100, BIAS_STRATEGY.NONE),
]
conv_configurations = [
    AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.ACTIVE),
    # AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.ALL),
    # AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.NONE),
    AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.ACTIVE),
    # AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.ALL),
    # AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.NONE),
    EpsilonConfiguration(1e-3, BIAS_STRATEGY.ALL),
    # EpsilonConfiguration(1e-3, BIAS_STRATEGY.ACTIVE),
    # EpsilonConfiguration(1e-3, BIAS_STRATEGY.NONE),
    # EpsilonConfiguration(100, BIAS_STRATEGY.ALL),
    EpsilonConfiguration(100, BIAS_STRATEGY.ACTIVE),
    # EpsilonConfiguration(100, BIAS_STRATEGY.NONE),
    # WWConfiguration()
]
lstm_configurations = [
    # EpsilonConfiguration(1e-3, BIAS_STRATEGY.ALL),
    EpsilonConfiguration(1e-3, BIAS_STRATEGY.ACTIVE),
    # EpsilonConfiguration(1e-3, BIAS_STRATEGY.NONE),
    # EpsilonConfiguration(100, BIAS_STRATEGY.ALL),
    # EpsilonConfiguration(100, BIAS_STRATEGY.ACTIVE),
    # EpsilonConfiguration(100, BIAS_STRATEGY.NONE),
    # WWConfiguration()
]


def get_configurations():
    configurations = []

    for lin_conf in linear_configurations:
        for ele_conf in elementwise_linear_configurations:
            for spa_conf in sparse_linear_configurations:
                for conv_conf in conv_configurations:
                    for lstm_conf in lstm_configurations:
                        config = LRPConfiguration()
                        config.set(LAYER.LINEAR, lin_conf)
                        config.set(LAYER.ELEMENTWISE_LINEAR, ele_conf)
                        config.set(LAYER.SPARSE_LINEAR, spa_conf)
                        config.set(LAYER.CONVOLUTIONAL, conv_conf)
                        config.set(LAYER.LSTM, lstm_conf)

                        configurations.append(config)
    for c in configurations:
        print(c)
    return configurations


