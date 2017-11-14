import tensorflow as tf
import numpy as np
import unittest

from configuration import LRPConfiguration, LAYER, EpsilonConfiguration, BIAS_STRATEGY, AlphaBetaConfiguration, \
    WWConfiguration, FlatConfiguration
from lrp import lrp


class TestConvolutionConfigs(unittest.TestCase):
    def test_epsilon_all_bias(self):
        # config
        config = LRPConfiguration()
        config.set(LAYER.CONVOLUTIONAL, EpsilonConfiguration())

        # Shape (3, 2, 2, 4, 2)
        expected = [[[[[0., 0.],
                       [0., 0.],
                       [0.11989002, 0.32163827],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [-0.03191111, 0.48056054],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.11567159, 0.32295102]]]],
                    [[[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.3551029, 0.35204153]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [-0.02325702, 0.19252851]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.22175457, 0.21091581],
                       [0., 0.],
                       [0., 0.]]]],
                    [[[[0.00876684, 0.14695124],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0.02519954, 0.17054913],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.24505551, 0.46359828],
                       [0., 0.],
                       [0., 0.]]]]]

        self._do_convolutional_test_with_config(config, expected)

    def test_epsilon_active_bias(self):
        # config
        config = LRPConfiguration()
        config.set(LAYER.CONVOLUTIONAL, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.ACTIVE))

        # Shape (3, 2, 2, 4, 2)
        expected = [[[[[0., 0.],
                       [0., 0.],
                       [0.11989002, 0.32163827],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [-0.03191111, 0.48056054],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.12920713, 0.33648656]]]],
                    [[[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.3551029, 0.35204153]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [-0.02325702, 0.19252851]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.24398904, 0.23315028],
                       [0., 0.],
                       [0., 0.]]]],
                    [[[[0.00876684, 0.14695124],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0.02519954, 0.17054913],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.26513471, 0.48367747],
                       [0., 0.],
                       [0., 0.]]]]]

        self._do_convolutional_test_with_config(config, expected)

    def test_epsilon_no_bias(self):
        # config
        config = LRPConfiguration()
        config.set(LAYER.CONVOLUTIONAL, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.NONE))

        # Shape (3, 2, 2, 4, 2)
        expected = [[[[[0., 0.],
                       [0., 0.],
                       [0.10428416, 0.3060324],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [-0.04751698, 0.46495467],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.10213605, 0.30941548]]]],
                    [[[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.33713338, 0.33407201]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [-0.04122655, 0.17455899]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.1995201, 0.18868134],
                       [0., 0.],
                       [0., 0.]]]],
                    [[[[-0.00199118, 0.13619322],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0.01444152, 0.15979111],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.22497632, 0.44351909],
                       [0., 0.],
                       [0., 0.]]]]]

        self._do_convolutional_test_with_config(config, expected)

    def test_alpha_beta_no_bias(self):
        # config
        config = LRPConfiguration()
        config.set(LAYER.CONVOLUTIONAL, AlphaBetaConfiguration(alpha=2, beta=-1, bias_strategy=BIAS_STRATEGY.NONE))

        # Shape (3, 2, 2, 4, 2)
        expected = [[[[[0., 0.],
                       [0., 0.],
                       [0.19799927, 0.58104888],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [-0.89017772, 0.88278689],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.20427209, 0.61883095]]]],
                    [[[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.64397426, 0.63812658]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [-0.87641592, 0.33343329]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.3990402, 0.37736268],
                       [0., 0.],
                       [0., 0.]]]],
                    [[[[-0.35146675, 0.27085197],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0.02872034, 0.31778188],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.44995264, 0.88703817],
                       [0., 0.],
                       [0., 0.]]]]]

        self._do_convolutional_test_with_config(config, expected)

    def test_alpha_beta_all_bias(self):
        # config
        config = LRPConfiguration()
        config.set(LAYER.CONVOLUTIONAL, AlphaBetaConfiguration(alpha=2, beta=-1, bias_strategy=BIAS_STRATEGY.ALL))

        # Shape (3, 2, 2, 4, 2)
        expected = [[[[[0., 0.],
                       [0., 0.],
                       [0.22762937, 0.61067898],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [-0.86054762, 0.91241699],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.23134317, 0.64590203]]]],
                    [[[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.67829868, 0.67245101]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [-0.84209149, 0.36775772]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.44350914, 0.42183162],
                       [0., 0.],
                       [0., 0.]]]],
                    [[[[-0.33007192, 0.2922468],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0.05011517, 0.33917671],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.49011103, 0.92719656],
                       [0., 0.],
                       [0., 0.]]]]]

        self._do_convolutional_test_with_config(config, expected)

    def test_alpha_beta_active_bias(self):
        # config
        config = LRPConfiguration()
        config.set(LAYER.CONVOLUTIONAL, AlphaBetaConfiguration(alpha=2, beta=-1, bias_strategy=BIAS_STRATEGY.ACTIVE))

        # Shape (3, 2, 2, 4, 2)
        expected = [[[[[0., 0.],
                       [0., 0.],
                       [0.23750607, 0.62055568],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [-0.89017772, 0.92229369],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.25841425, 0.67297311]]]],
                    [[[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.68974016, 0.68389248]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [-0.87641592, 0.37919919]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.48797808, 0.46630056],
                       [0., 0.],
                       [0., 0.]]]],
                    [[[[-0.35146675, 0.29937841],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0.05724678, 0.34630832],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.53026941, 0.96735494],
                       [0., 0.],
                       [0., 0.]]]]]

        self._do_convolutional_test_with_config(config, expected)

    def test_ww(self):
        # config
        config = LRPConfiguration()
        config.set(LAYER.CONVOLUTIONAL, WWConfiguration())

        # Shape (3, 2, 2, 4, 2)
        expected = [[[[[0., 0.],
                       [0., 0.],
                       [0.2205872, 0.21228028],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [0.08030722, 0.37700302],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.1153995, 0.11105376]]]],
                    [[[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.217177, 0.20899851]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.0790657, 0.3711747]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.11823574, 0.1137832],
                       [0., 0.],
                       [0., 0.]]]],
                    [[[[0.08709392, 0.08381412],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0.03170751, 0.14885121],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.18555663, 0.1785689],
                       [0., 0.],
                       [0., 0.]]]]]

        self._do_convolutional_test_with_config(config, expected)

    def test_flat(self):
        # config
        config = LRPConfiguration()
        config.set(LAYER.CONVOLUTIONAL, FlatConfiguration())

        # Shape (3, 2, 2, 4, 2)
        expected = [[[[[0., 0.],
                       [0., 0.],
                       [0.22254443, 0.22254443],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [0.22254443, 0.22254443],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.11642342, 0.11642342]]]],
                    [[[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.21910398, 0.21910398]],
                      [[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0.21910398, 0.21910398]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.11928483, 0.11928483],
                       [0., 0.],
                       [0., 0.]]]],
                    [[[[0.08786669, 0.08786669],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0.08786669, 0.08786669],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]]],
                     [[[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.]],
                      [[0., 0.],
                       [0.18720304, 0.18720304],
                       [0., 0.],
                       [0., 0.]]]]]

        self._do_convolutional_test_with_config(config, expected)

    def _do_convolutional_test_with_config(self, config, expected_explanation, shape=None, input=None, filters=None,
                                           bias=None):
        # Set up default values for convolutional tests
        shape = shape if shape is not None else (3, 2, 4, 2)
        input = input if input is not None else [[[[1.80995856, 1.75300107],
                                                   [-0.23499947, 1.24451596],
                                                   [0.77244209, 2.310736],
                                                   [-1.72436699, 0.85537119]],
                                                  [[1.56641824, 0.74264226],
                                                   [2.09326164, -0.95402815],
                                                   [0.58332319, 2.63436498],
                                                   [0.87224583, 2.69362554]]],
                                                 [[[3.03400979, 0.3397696],
                                                   [1.07886937, 1.22382285],
                                                   [-0.72168244, -0.06016954],
                                                   [2.16870603, 2.19065678]],
                                                  [[0.29083574, -0.81528648],
                                                   [1.03727835, 0.99993768],
                                                   [0.95362287, 0.36160145],
                                                   [0.43953022, 0.85893218]]],
                                                 [[[-0.02139509, 1.49174273],
                                                   [0.66090231, -0.06420824],
                                                   [-1.20463713, 1.60492691],
                                                   [-0.81951792, -0.59385211]],
                                                  [[-0.25717514, 1.3133288],
                                                   [1.29516779, 2.60277641],
                                                   [-0.53964658, 1.44639959],
                                                   [-0.42642239, 1.81083638]]]]

        filters = filters if filters is not None else [[[[1.07621199, -1.37667063],
                                                         [1.05575343, 1.0899564]]],
                                                       [[[-0.64935916, 0.55298082],
                                                         [1.40695463, 0.18898547]]]]

        bias = bias if bias is not None else [0.49761477, -0.54884433]

        with tf.Graph().as_default():
            # Input of shape (batch_size, input_height, input_width, input_channels)
            inp = tf.placeholder(dtype=tf.float32, shape=shape)

            conv_filter = tf.placeholder(dtype=tf.float32, shape=(2, 1, 2, 2))
            conv_bias = tf.placeholder(dtype=tf.float32, shape=(2,))

            # Output has shape (3, 2, 4, 2)
            conv_out = tf.nn.conv2d(inp, conv_filter, [1, 1, 1, 1], 'SAME')

            # Add bias
            conv_out = tf.nn.bias_add(conv_out, conv_bias)

            # Reshape output to have shape (3, 2, 8) to let lrp interpret it as batch_size = 3, p/s=2 and 8 predictions
            conv_reshaped = tf.reshape(conv_out, (3, 2, -1))

            # Take the softmax to simulate something real
            out = tf.nn.softmax(conv_reshaped)

            # Get the explanation with given configuration
            expl = lrp.lrp(inp, out, config)

            with tf.Session() as s:
                before_soft, output, explanation = s.run([conv_reshaped, out, expl],
                                                         feed_dict={inp: input, conv_filter: filters, conv_bias: bias})
                self.assertTrue(np.allclose(expected_explanation, explanation, rtol=1.e-3, atol=1.e-3),
                                "expected indices did not equal actual indices")

                self.assertTrue(True)
