import tensorflow as tf
import numpy as np
import unittest

from configuration import LRPConfiguration, AlphaBetaConfiguration, LAYER, BIAS_STRATEGY
from lrp import lrp


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
        self._do_test(config, np.array(expected_result))

    def _do_test(self, config, expected_result):
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

                print(explanation.shape)
                print(explanation)

                # Check if the explanation has the right shape
                self.assertEqual(explanation.shape, expected_result.shape, msg="Should be a wellformed explanation")

                # Check if the relevance scores are correct (the correct values are found by
                # calculating the example by hand)
                self.assertTrue(
                    np.allclose(explanation, expected_result, rtol=1e-03, atol=1e-03),
                    msg="Should be a good explanation")
