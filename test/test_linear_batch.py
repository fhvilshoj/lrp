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

            # The tf.expand_dims() is necessary because we call _lrp which means that
            # we bypass the part of the framework that takes care of adding and removing
            # an extra dimension for multiple predictions per sample
            explanation = lrp._lrp(inp, activation, tf.expand_dims(activation, 1))

            with tf.Session() as s:
                act, expl = s.run([activation, explanation])

                # We loose 2 * (2 + 4) relevance from bias
                self.assertEqual(np.sum(act) - 12, np.sum(expl))
                self.assertTrue(np.allclose(
                    np.array([[[2, 4, 6]], [[8, 10, 12]]]),
                    expl))
