import tensorflow as tf
import numpy as np
import unittest

from configuration import LRPConfiguration, LAYER, EpsilonConfiguration, BIAS_STRATEGY
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
                print("before_soft: ", before_soft)
                print("Output: ", output)
                print("Explanation: ", explanation)
                self.assertTrue(np.allclose(expected_explanation, explanation, rtol=1.e-3, atol=1.e-3),
                                "expected indices did not equal actual indices")

                self.assertTrue(True)
