from lrp.configuration import *
from copy import deepcopy

linear_configurations = [
    EpsilonConfiguration(1e-12, BIAS_STRATEGY.NONE),
    EpsilonConfiguration(1e-12, BIAS_STRATEGY.IGNORE),
    EpsilonConfiguration(1e-12, BIAS_STRATEGY.ACTIVE),

    EpsilonConfiguration(0.01, BIAS_STRATEGY.NONE),
    EpsilonConfiguration(0.01, BIAS_STRATEGY.IGNORE),
    EpsilonConfiguration(0.01, BIAS_STRATEGY.ACTIVE),

    EpsilonConfiguration(10, BIAS_STRATEGY.NONE),
    EpsilonConfiguration(10, BIAS_STRATEGY.IGNORE),
    EpsilonConfiguration(10, BIAS_STRATEGY.ACTIVE),

    EpsilonConfiguration(100, BIAS_STRATEGY.NONE),
    EpsilonConfiguration(100, BIAS_STRATEGY.IGNORE),
    EpsilonConfiguration(100, BIAS_STRATEGY.ACTIVE),

    AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.NONE),
    AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.IGNORE),
    AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.ACTIVE),

    AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.NONE),
    AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.IGNORE),
    AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.ACTIVE),

    WWConfiguration()
]

conv_configurations = [
    EpsilonConfiguration(1e-12, BIAS_STRATEGY.NONE),
    EpsilonConfiguration(1e-12, BIAS_STRATEGY.IGNORE),
    EpsilonConfiguration(1e-12, BIAS_STRATEGY.ACTIVE),

    EpsilonConfiguration(0.01, BIAS_STRATEGY.NONE),
    EpsilonConfiguration(0.01, BIAS_STRATEGY.IGNORE),
    EpsilonConfiguration(0.01, BIAS_STRATEGY.ACTIVE),

    EpsilonConfiguration(10, BIAS_STRATEGY.NONE),
    EpsilonConfiguration(10, BIAS_STRATEGY.IGNORE),
    EpsilonConfiguration(10, BIAS_STRATEGY.ACTIVE),

    EpsilonConfiguration(100, BIAS_STRATEGY.NONE),
    EpsilonConfiguration(100, BIAS_STRATEGY.IGNORE),
    EpsilonConfiguration(100, BIAS_STRATEGY.ACTIVE),

    AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.NONE),
    AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.IGNORE),
    AlphaBetaConfiguration(2, -1, BIAS_STRATEGY.ACTIVE),

    AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.NONE),
    AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.IGNORE),
    AlphaBetaConfiguration(1, 0, BIAS_STRATEGY.ACTIVE),

    WWConfiguration()
]

lstm_configurations = [
    EpsilonConfiguration(1e-12, BIAS_STRATEGY.ALL),
    EpsilonConfiguration(1e-12, BIAS_STRATEGY.IGNORE),
]

max_pooling_configurations = [
    BaseConfiguration(RULE.WINNERS_TAKE_ALL),
    BaseConfiguration(RULE.WINNER_TAKES_ALL),
    # This second rule doesn't matter since implementation looks for WINNERS_TAKE_ALL or not
    BaseConfiguration(RULE.FLAT),
]

def get_configurations_for_layers(linear=False, convolution=False, lstm=False, maxpool=False, batchnorm=False):
    cf = []
    
    def _c(configs, layers, rules):
        if configs:
            new_configs = []
            for rule in rules:
                for c in configs:
                    new_c = deepcopy(c)
                    for l in layers:
                        new_c.set(l, rule)
                    new_configs.append(new_c)
            return new_configs
        else:
            for rule in rules:
                config = LRPConfiguration()
                for l in layers:
                    config.set(l, rule)
                configs.append(config)
            return configs

    if linear:
        cf = _c(cf, [LAYER.LINEAR, LAYER.SPARSE_LINEAR], linear_configurations)
    if convolution:
        cf = _c(cf, [LAYER.CONVOLUTIONAL], conv_configurations)
    if lstm:
        cf = _c(cf, [LAYER.LSTM], lstm_configurations)
    if maxpool:
        cf = _c(cf, [LAYER.MAX_POOLING], max_pooling_configurations)

    if batchnorm:
        flat = FlatConfiguration()
        for c in cf:
            l = c.get(LAYER.LINEAR)
            c.set(LAYER.ELEMENTWISE_LINEAR, l)
        to_append = []
        for c in cf:
            c_new = deepcopy(c)
            c_new.set(LAYER.ELEMENTWISE_LINEAR, flat)
            to_append.append(c_new)
        cf.extend(to_append)
    
    return cf
            
        
def get_configurations():
    configurations = []

    for lin_conf in linear_configurations:
        for conv_conf in conv_configurations:
            for lstm_conf in lstm_configurations:
                for max_conf in max_pooling_configurations:
                    for i in range(2):
                        config = LRPConfiguration()

                        config.set(LAYER.LINEAR, lin_conf)
                        config.set(LAYER.SPARSE_LINEAR, lin_conf)
                        config.set(LAYER.CONVOLUTIONAL, conv_conf)
                        config.set(LAYER.MAX_POOLING, max_conf)
                        config.set(LAYER.LSTM, lstm_conf)

                        if i == 0:
                            config.set(LAYER.ELEMENTWISE_LINEAR, lin_conf)
                        else:
                            config.set(LAYER.ELEMENTWISE_LINEAR, FlatConfiguration())

                        configurations.append(config)
    return configurations
