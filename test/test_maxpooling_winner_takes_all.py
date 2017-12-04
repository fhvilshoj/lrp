import tensorflow as tf
import numpy as np
import unittest

from configuration import LRPConfiguration, LAYER, RULE, BaseConfiguration
from lrp import lrp

class TestMaxPoolingDistribute(unittest.TestCase):
    def test_distribute(self):
        with tf.Graph().as_default():
            input = tf.constant([[1., 2., 2., 3., 3., 3., 3., 2., 1]],
                                dtype=tf.float32)

            i = tf.reshape(input, (1, 3, 3, 1))

            # Create max pooling layer
            # (1, 2, 2, 1)
            activation = tf.nn.max_pool(i, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

            # Reshape to get 1 p/s
            output = tf.reshape(activation, (2, 2))

            config = LRPConfiguration()
            config.set(LAYER.MAX_POOLING, BaseConfiguration(RULE.WINNER_TAKES_ALL))

            explanation = lrp.lrp(input, output, config)

            with tf.Session() as s:
                expl = s.run(explanation)

                # Check if the explanation has the right shape
                self.assertEqual((1, 9), expl.shape,
                                 msg="Should be a wellformed explanation")

                # Expected explanation
                expected = np.array([[0., 0., 0., 3., 0., 0., 3., 0., 0]])

                # Check if the relevance scores are correct (the correct values
                # are found by calculating the example by hand)
                self.assertTrue(
                    np.allclose(expected,
                                expl,
                                rtol=1e-03,
                                atol=1e-03),
                    msg="Should be a good explanation")

                self.assertTrue(True)
