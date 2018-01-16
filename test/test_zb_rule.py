import tensorflow as tf
import numpy as np
import unittest

from configuration import LRPConfiguration, AlphaBetaConfiguration, LAYER, BIAS_STRATEGY, EpsilonConfiguration, \
    LOG_LEVEL
from lrp import lrp


def _c(val, size=18):
    return np.array([val] * size, dtype=np.float32)

def _l(val):
    return np.array([val] * 15, dtype=np.float32).reshape(3, 5, 1)

class TestZbRule(unittest.TestCase):
    def test_alpha_beta_ignore_bias(self):
        config = LRPConfiguration()
        config.set(LAYER.LINEAR, AlphaBetaConfiguration(bias_strategy=BIAS_STRATEGY.IGNORE))

        expected_result = [[[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],

                           [[0.05056302043, 0.001782476642, 0.4564389581, 0, 0.2856364171],
                            [0.0006553275046, 0.2230518547, 0.1905802792, 0.09459906389, 0.08207159722]],

                           [[0, 0, 0, 0, 0],
                            [0.1729084504, 0, 0.0773933278, 0.2900907282, 0.006641839018]]]
        self._do_linear_test(config, np.array(expected_result))

    def test_zb_ignore_bias(self):
        config = LRPConfiguration()
        config.set(LAYER.LINEAR, AlphaBetaConfiguration(bias_strategy=BIAS_STRATEGY.IGNORE))
        config.set_first_layer_zb(_l(-1), _l(1), BIAS_STRATEGY.IGNORE)

        expected_result = [[[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],
                           [[0.08941490911, 0.001847394735, 0.3857157564, 0.04130132575, 0.2761414863],
                            [0.06524084895, 0.2049935833, 0.1593686046, 0.08435578325, 0.07699930241]],
                           [[0, 0, 0, 0, 0],
                            [0.1842489119, 0.02888073415, 0.04590480826, 0.2341442052, 0.05385568583]]]

        self._do_linear_test(config, np.array(expected_result))

    def test_zb_no_bias(self):
        config = LRPConfiguration()
        config.set(LAYER.LINEAR, AlphaBetaConfiguration(bias_strategy=BIAS_STRATEGY.IGNORE))
        config.set_first_layer_zb(_l(-1), _l(1), BIAS_STRATEGY.NONE)

        expected_result = [[[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],
                           [[0.1145575292, 0.002366864524, 0.4941753504, 0.05291486486, 0.3537898399],
                            [0.07031890972, 0.2048122289, 0.1691654014, 0.1035978445, 0.08054911078]],
                           [[0, 0, 0, 0, 0],
                            [0.2102658664, 0.0283220131, 0.04511325643, 0.2887187793, 0.05874777231]]]

        self._do_linear_test(config, np.array(expected_result))

    def test_zb_all_bias(self):
        config = LRPConfiguration()
        config.set(LAYER.LINEAR, AlphaBetaConfiguration(bias_strategy=BIAS_STRATEGY.IGNORE))
        config.set_first_layer_zb(_l(-1), _l(1), BIAS_STRATEGY.ALL)

        expected_result = [[[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],

                           [[0.0698808139, -0.0423098508, 0.449498635, 0.008238149531, 0.3091131246],
                            [0.0628218352, 0.1973151543, 0.1616683269, 0.09610076993, 0.07305203625]],

                           [[0, 0, 0, 0, 0],
                            [0.193439198, 0.01149534468, 0.02828658801, 0.2718921108, 0.04192110389]]]

        self._do_linear_test(config, np.array(expected_result))

    def test_zb_active_bias(self):
        config = LRPConfiguration()
        config.set(LAYER.LINEAR, AlphaBetaConfiguration(bias_strategy=BIAS_STRATEGY.IGNORE))
        config.set_first_layer_zb(_l(-1), _l(1), BIAS_STRATEGY.ACTIVE)

        expected_result = [[[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],

                           [[0.0698808139, -0.0423098508, 0.449498635, 0.008238149531, 0.3091131246],
                            [0.0628218352, 0.1973151543, 0.1616683269, 0.09610076993, 0.07305203625]],

                           [[0, 0, 0, 0, 0],
                            [0.193439198, 0.01149534468, 0.02828658801, 0.2718921108, 0.04192110389]]]

        self._do_linear_test(config, np.array(expected_result))

    def _do_linear_test(self, config, expected_result):
        if not type(expected_result).__module__ == np.__name__:
            expected_result = np.array(expected_result)

        with tf.Graph().as_default():
            inp = tf.constant([[0.61447761, -0.47432536, -0.29292757, -0.78589278, -0.86108047],
                               [0.28479454, -0.60827365, 0.86519678, -0.65091976, -0.6819959],
                               [-0.4422958, 0.55866813, -0.88997564, -0.87868751, -0.0389981]]
                              , dtype=tf.float32)

            W1 = tf.constant([[-0.70950127, -0.15957509, -0.607047, 0.13172],
                              [-0.9520821, -0.79133917, -0.03131101, -0.00217408],
                              [-0.35051205, 0.84566609, 0.22297791, 0.39139763],
                              [-0.05067179, 0.07747386, -0.89703108, 0.22393099],
                              [-0.43415774, 0.44243544, -0.17682024, -0.31072929]], dtype=tf.float32)
            b1 = tf.constant([0.10282315, -0.07288911, -0.53922754, -0.3299993], dtype=tf.float32)

            out1 = tf.nn.relu(inp @ W1 + b1)

            W2 = tf.constant([[-0.3378281, -0.03719562, -0.05190714, 0.3983907],
                              [-0.92650528, -0.97646332, 0.08498075, 0.37901429],
                              [-0.36540267, -0.26421945, -0.79152602, 0.73636482],
                              [0.59652669, 0.89863044, 0.02424345, 0.09883726]], dtype=tf.float32)
            b2 = tf.constant([-0.26253957, 0.91930372, 0.11791677, -0.28088199], dtype=tf.float32)

            out2 = out1 @ W2 + b2

            out3 = tf.reshape(out2, (3, 2, 2))

            out = tf.nn.softmax(out3)

            expl = lrp.lrp(inp, out, config)

            with tf.Session() as s:
                explanation = s.run(expl)

                # Check if the explanation has the right shape
                self.assertEqual(explanation.shape, expected_result.shape, msg="Should be a wellformed explanation")

                # Check if the relevance scores are correct (the correct values are found by
                # calculating the example by hand)
                self.assertTrue(
                    np.allclose(explanation, expected_result, rtol=1e-03, atol=1e-03),
                    msg="Should be a good explanation")

    def test_conv_zb_ignore_bias(self):
        expected_result = [[[[0, 0],
                             [0, 0],
                             [-2.352069164e-05, -2.391419691e-05]],
                            [[0, 0],
                             [0, 0],
                             [-0.0002779048828, -0.0001325144132]],
                            [[0.04152410963, 0.01750268803],
                             [0.1589671162, 0.1768956439],
                             [-0.001919716057, -8.201815885e-05]]]]

        config = LRPConfiguration()
        config.set(LAYER.LINEAR, EpsilonConfiguration(epsilon=1, bias_strategy=BIAS_STRATEGY.IGNORE))
        config.set_first_layer_zb(_c(-0.5), _c(2.), bias_strategy=BIAS_STRATEGY.IGNORE)

        self._do_convolutional_test(config, expected_result)

    def test_conv_zb_no_bias(self):
        expected_result = [[[[0, 0],
                             [0, 0],
                             [-1.865625321e-05, -1.896837558e-05]],
                            [[0, 0],
                             [0, 0],
                             [-0.0002204299066, -0.0001051084077]],
                            [[0.03451383443, 0.01454781046],
                             [0.1321296177, 0.1470313757],
                             [-0.0010226063, -4.368994345e-05]]]]

        config = LRPConfiguration()
        config.set(LAYER.LINEAR, EpsilonConfiguration(epsilon=1, bias_strategy=BIAS_STRATEGY.IGNORE))
        config.set_first_layer_zb(_c(-0.5), _c(2.), bias_strategy=BIAS_STRATEGY.NONE)

        self._do_convolutional_test(config, expected_result)

    def test_conv_zb_all_bias(self):
        expected_result = [[[[0, 0],
                             [0, 0],
                             [-3.049265839e-05, -3.080478076e-05]],
                            [[0, 0],
                             [0, 0],
                             [-0.0002322663118, -0.0001169448129]],
                            [[0.04284719936, 0.0228811754],
                             [0.1404629826, 0.1553647406],
                             [-0.001139536047, -0.0001606196899]]]]

        config = LRPConfiguration()
        config.set(LAYER.LINEAR, EpsilonConfiguration(epsilon=1, bias_strategy=BIAS_STRATEGY.IGNORE))
        config.set_first_layer_zb(_c(-0.5), _c(2.), bias_strategy=BIAS_STRATEGY.ALL)

        self._do_convolutional_test(config, expected_result)

    def test_conv_zb_active_bias(self):
        expected_result = [[[[0, 0],
                             [0, 0],
                             [-4.232906357e-05, -4.264118594e-05]],
                            [[0, 0],
                             [0, 0],
                             [-0.0002441027169, -0.0001287812181]],
                            [[0.0511805643, 0.03121454034],
                             [0.1487963475, 0.1636981056],
                             [-0.001490325286, -0.0005114089294]]]]

        config = LRPConfiguration()
        config.set(LAYER.LINEAR, EpsilonConfiguration(epsilon=1, bias_strategy=BIAS_STRATEGY.IGNORE))
        config.set_first_layer_zb(_c(-0.5), _c(2.), bias_strategy=BIAS_STRATEGY.ACTIVE)

        self._do_convolutional_test(config, expected_result)

    def _do_convolutional_test(self, config, expected_result):
        if not type(expected_result).__module__ == np.__name__:
            expected_result = np.array(expected_result)

        with tf.Graph().as_default():
            # Shape (1, 3, 3, 2)
            inp = tf.constant([[[[0.26327863, 1.34089666],
                                 [1.10478271, 1.52725762],
                                 [-0.16542361, 0.9857486]],
                                [[0.59388839, 0.89302082],
                                 [-0.12325813, 1.37334976],
                                 [1.70439274, 1.37814247]],
                                [[0.37915105, 0.89512656],
                                 [-0.12834121, 0.72336895],
                                 [1.35641106, 1.76352144]]]], dtype=tf.float32)
            # Shape: (2, 2, 2, 1)
            kernel = tf.constant([[[[0.28530154],
                                    [-0.095688446]],
                                   [[-0.45116284],
                                    [0.87342771]]],
                                  [[[0.51163061],
                                    [-0.86481167]],
                                   [[-0.78022886],
                                    [-0.86212577]]]], dtype=tf.float32)

            bias = tf.constant([0.484489966914], dtype=tf.float32)

            # Shape: (1, 2, 2, 1)
            out1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(inp, kernel, [1, 2, 2, 1], 'SAME'), bias))

            # Shape: (1, 4)
            out2 = tf.reshape(out1, (1, 4))

            # Shape: (4, 2)
            weights = tf.constant([[-0.93597473, -0.26599479],
                                   [-0.58296088, -0.05599708],
                                   [-0.21809592, 0.93448646],
                                   [-0.36266413, -0.00806697]], dtype=tf.float32)

            # Shape: (2,)
            bias2 = tf.constant([0.0794559, -0.48629082], dtype=tf.float32)

            # Shape: (1, 2)
            out3 = out2 @ weights + bias2

            # Shape: (1, 2)
            out4 = tf.nn.softmax(out3)

            # Explanation shape: (1, 3, 3, 2)
            expl = lrp.lrp(inp, out4, config)

            with tf.Session() as s:
                explanation = s.run(expl)

                # Check if the explanation has the right shape
                self.assertEqual(explanation.shape, expected_result.shape, msg="Should be a wellformed explanation")

                print(expected_result)
                print()
                print(explanation)

                # Check if the relevance scores are correct (the correct values are found by
                # calculating the example by hand)
                self.assertTrue(
                    np.allclose(explanation, expected_result, rtol=1e-06, atol=1e-06),
                    msg="Should be a good explanation")
