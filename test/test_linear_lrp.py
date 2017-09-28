import tensorflow as tf
import numpy as np
from lrp import lrp
import unittest


class LinearLRPTest(unittest.TestCase):

    def runTest(self):
        g = tf.Graph()
        with g.as_default():
            inp = tf.placeholder(tf.float32, shape=(1, 4))

            with tf.name_scope('linear'):
                weights = tf.constant([
                    [1, 2, -2],
                    [3, -2, 0],
                    [3, -6, 1],
                    [-1, 1, 10]], dtype=tf.float32)

                biases = tf.constant([[-3, 2, 8]], dtype=tf.float32)
                mul = tf.matmul(inp, weights)
                activation = mul + biases

            with tf.name_scope('linear'):
                weights = tf.constant(
                    np.array([
                        [2, 0],
                        [0, 8],
                        [1, 1]]), dtype=tf.float32)
                biases = tf.constant([[-5, 8]], dtype=tf.float32)
                mul = tf.matmul(activation, weights)
                activation = mul + biases

            pred = activation
            expl = lrp.lrp(pred)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                prediction, explanation = sess.run([pred, expl], feed_dict={inp: [[1, -2, 3, 4]]})

                self.assertEqual(prediction.shape, (1, 2), msg="Should be able to do a linear forward pass")
                self.assertEqual(explanation.shape, inp.shape, msg="Should be a wellformed explanation")
                self.assertTrue(
                    np.allclose(explanation[0], [0., 0., 2.69, 35.87], rtol=1e-03, atol=1e-03),
                    msg="Should be a good linear explanation")
