import tensorflow as tf
import numpy as np
import unittest
from lrp import lrp

# Test that the linear layer can handle batches
# of more than one input.
class TestLinearBatch(unittest.TestCase):
    def runTest(self):
        with tf.Graph().as_default():
            inp = tf.constant([[1, 2, 3],
                               [4, 5, 6]],
                              dtype=tf.float32)

            w1 = tf.constant([[1, 1],
                              [1, 1],
                              [1, 1]],
                             dtype=tf.float32)

            b1 = tf.constant([2, 4], dtype=tf.float32)
            activation = tf.matmul(inp, w1) + b1

            # Calculate relevances using lrp
            explanation = lrp.lrp(inp, activation)

            with tf.Session() as s:
                act, expl = s.run([activation, explanation])

                self.assertTrue(np.allclose(
                    np.array([[1, 2, 3], [4, 5, 6]]),
                    expl))
