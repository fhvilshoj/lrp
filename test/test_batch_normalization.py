import time


import tensorflow as tf
import unittest
from lrp import lrp
import numpy as np


class TestBatchNormalization(unittest.TestCase):
    def runTest(self):
        g = tf.Graph()
        with g.as_default():
            inp = tf.placeholder(tf.float32, (2, 5))

            x = tf.contrib.layers.batch_norm(inp, is_training=False, scale=True)
            vars = tf.global_variables()
            beta = next(i for i in vars if 'beta' in i.name)
            gamma = next(i for i in vars if 'gamma' in i.name)
            mean = next(i for i in vars if 'mean' in i.name)
            variance = next(i for i in vars if 'variance' in i.name)

            b = tf.constant([0, 1, 0, 1, 0], dtype=tf.float32)
            assign_beta = tf.assign(beta, b)
            g = tf.constant([0.1, 0.2, 0.3, 0.4, 0.5], dtype=tf.float32)
            assign_gamma = tf.assign(gamma, g)
            m = tf.constant([1, 2, 3, 4, 4.5], dtype=tf.float32)
            assign_mean = tf.assign(mean, m)
            v = tf.constant([0.2, 0.2, 0.2, 0.2, 0.2], dtype=tf.float32)
            assign_variance = tf.assign(variance, v)

            # The tf.expand_dims() is necessary because we call _lrp which means that
            # we bypass the part of the framework that takes care of adding and removing
            # an extra dimension for multiple predictions per sample
            explanation = lrp._lrp(inp, x, tf.expand_dims(x, 1))

            with tf.Session() as s:
                s.run(tf.global_variables_initializer())
                s.run([assign_beta, assign_gamma, assign_mean, assign_variance])

                expected_relevances = np.array(
                    [[[0, 0.8921994513, 0, 1, 0.5576246571],
                     [0, 0.8921994513, 0, 1, -0.5576246571]]])
                relevances = s.run(explanation, feed_dict={inp: [[1, 2, 3, 4, 5],
                                                                 [1, 2, 3, 4, 4]]})
                self.assertTrue(np.allclose(expected_relevances, relevances, rtol=1e-03, atol=1e-03),
                                msg="The relevances do not match the expected")
