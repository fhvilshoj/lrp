import tensorflow as tf
import numpy as np
import unittest
from lrp import lrp


class TestTile(unittest.TestCase):
    def runTest(self):
        with tf.Graph().as_default():
            # Create the input
            # Shape: (1,2,3)
            inp = tf.constant([[[1, 2, 3],
                                [4, 5, 6]]], dtype=tf.float32)

            # Shape: (1, 4, 6)
            pred = tf.tile(inp, (1, 2, 2))

            # Calculate the explanation
            expl = lrp.lrp(inp, pred)

            # Run a tensorflow session to evaluate the graph
            with tf.Session() as sess:
                # Initialize the variables
                sess.run(tf.global_variables_initializer())

                # Calculate the explanation
                explanation = sess.run(expl)

                # Create the expected explanation
                # Shape: (1,4,2,3)
                expected_explanation = np.array([[[[0, 0, 3],
                                                   [0, 0, 0]],

                                                  [[0, 0, 0],
                                                   [0, 0, 6]],

                                                  [[0, 0, 3],
                                                   [0, 0, 0]],

                                                  [[0, 0, 0],
                                                   [0, 0, 6]]]]
                                                )

                # Check if the calculated explanation matches the expected explanation
                self.assertTrue(
                    np.allclose(expected_explanation,
                                explanation,
                                rtol=1e-03,
                                atol=1e-03),
                    msg="The explanation does not match the expected explanation")
