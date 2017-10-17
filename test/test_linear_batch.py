import tensorflow as tf
import numpy as np
import unittest
from lrp import lrp


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
            explanation = lrp._lrp(inp, activation, activation)

            with tf.Session() as s:
                act, expl = s.run([activation, explanation])

                print("Activation: \n", act)
                print("Explanation: \n", expl)

                print(np.sum(act), np.sum(expl))

                self.assertEqual(np.sum(act)-12, np.sum(expl))
