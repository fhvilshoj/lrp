import tensorflow as tf
import numpy as np
import unittest
from lrp import lrp


class TestSlicing(unittest.TestCase):
    def runTest(self):
        with tf.Graph().as_default():
            inp = tf.constant([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3]],
                               [[11, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                               [[1, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]], dtype=tf.float32)

            out = tf.slice(inp, [2, 0, 0], [1, 3, 4])

            R = lrp.lrp(inp, out)

            with tf.Session() as s:
                expected_explanation = np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                                 [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                                 [[0, 2, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]]])
                explanation = s.run(R)

                self.assertTrue(np.allclose(expected_explanation, explanation, atol=1e-03, rtol=1e-03),
                                "The explanation does not match the expected explanation")
