import tensorflow as tf
import numpy as np
import unittest
from lrp import lrp


class TestReshape(unittest.TestCase):
    def runTest(self):
        with tf.Graph().as_default():
            # Input shape: (2,3,4)
            inp = tf.constant([[[1, 2, 3, 4],
                                [5, 6, 7, 8],
                                [9, 10, 11, 12]],
                               [[-1, -2, -3, -4],
                                [-5, -6, -7, -8],
                                [-9, -10, -11, -12]]], dtype=tf.float32)

            # Output shape: (2, 12)
            out = tf.reshape(inp, [2, 12])

            # Calculate relevances
            expl = lrp.lrp(inp, out)

            with tf.Session() as s:
                output = s.run(out)
                explanation = s.run(expl)

                # Shape: (2,3,4)
                expected_explanation = np.array([[[0., 0., 0., 0.],
                                                 [0., 0., 0., 0.],
                                                 [0., 0., 0., 12.]],
                                                [[-1., 0., 0., 0.],
                                                 [0., 0., 0., 0.],
                                                 [0., 0., 0., 0.]]]
                                               )

                self.assertTrue(np.allclose(expected_explanation, explanation),
                                "Explanation does not match expected explanation")
