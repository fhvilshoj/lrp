import tensorflow as tf
import numpy as np
import unittest

from configuration import LRPConfiguration
from lrp import lrp


class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        with tf.Graph().as_default():
            inp = tf.constant([[[1, 2, 3],[1, 2, 3]],[[1, 2, 3],[1, 2, 3]]], dtype=tf.float32)

            out = tf.nn.softmax(inp)

            config = LRPConfiguration()
            expl = lrp.lrp(inp, out, config)

            with tf.Session() as s:
                explanation = s.run(expl)

                # expected = np.array([-0.05989202454, -0.162803402, 0.8879363823, -0.05989202454, -0.162803402, 0.8879363823]).reshape((2, 2, 2, 3))
                expected = np.array([1, 2, 2.])

                print("Expected shape: ", expected.shape)
                print("Explanation shape: ", explanation.shape)

                # Check if the relevance scores are correct (the correct values are found by
                # calculating the example by hand)
                self.assertTrue(
                    np.allclose(expected, explanation, rtol=1e-01, atol=1e-01),
                    msg="Should be a good explanation")
