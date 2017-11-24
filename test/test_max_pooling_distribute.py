import tensorflow as tf
import numpy as np
import unittest

from configuration import LRPConfiguration, LAYER, EpsilonConfiguration
from lrp import lrp


class TestMaxPoolingDistribute(unittest.TestCase):
    def test_distribute(self):
        with tf.Graph().as_default():
            input = tf.constant([[0.3315922947, 1.053559579, 0.7477053648, 1.22290369,
                                  0.3730588596, -1.034354431, 0.9187013371, 1.478589349,
                                  -0.7325915066, -0.3569675024, -1.136600512, 0.5516666285,
                                  0.4834049101, -1.613833301, 0.1520745652, 0.117390006]],
                                dtype=tf.float32)
            i = tf.reshape(input, (1, 4, 4, 1))

            # Create max pooling layer
            # (1, 2, 2, 1)
            activation = tf.nn.max_pool(i, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

            # Reshape to get 1 p/s
            output = tf.reshape(activation, (1, 4))

            config = LRPConfiguration()
            config.set(LAYER.MAX_POOOING, EpsilonConfiguration())

            explanation = lrp.lrp(input, output, config)

            with tf.Session() as s:
                expl = s.run(explanation)

                # Check if the explanation has the right shape
                self.assertEqual((1, 16), expl.shape,
                                 msg="Should be a wellformed explanation")

                # Expected explanation
                expected = np.array(
                    [[0, 0, 0.2531077301, 0.4139683781, 0, 0, 0.3109920311, 0.50052121, 0, 0, 0, 0, 0, 0, 0, 0]])

                # Check if the relevance scores are correct (the correct values
                # are found by calculating the example by hand)
                self.assertTrue(
                    np.allclose(expected,
                                expl,
                                rtol=1e-03,
                                atol=1e-03),
                    msg="Should be a good explanation")

                self.assertTrue(True)
