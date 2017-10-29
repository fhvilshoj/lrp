import tensorflow as tf
import unittest
from lrp import lrp
import numpy as np


class TestBatchNormalization(unittest.TestCase):
    def test_one_prediction_per_sample(self):
        g = tf.Graph()
        with g.as_default():
            inp = tf.placeholder(tf.float32, (2, 1, 5))

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

            # Get the explanation
            explanation = lrp.lrp(inp, x)

            with tf.Session() as s:
                s.run(tf.global_variables_initializer())
                s.run([assign_beta, assign_gamma, assign_mean, assign_variance])

                # Shape: (2, 1, 1, 5)
                expected_relevances = np.array(
                    [[[[0, 0, 0, 1, 0]]],
                     [[[0, 0.8921994513, 0, 0, 0]]]])
                relevances = s.run(explanation, feed_dict={inp: [[[1, 0, 3, 4, 5]],
                                                                 [[1, 2, 3, 0, 4]]]})

                self.assertTrue(np.allclose(expected_relevances, relevances, rtol=1e-03, atol=1e-03),
                                msg="The relevances do not match the expected")

    def test_four_predictions_per_sample(self):
        g = tf.Graph()
        with g.as_default():
            inp = tf.placeholder(tf.float32, (2, 4, 5))

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

            # Get the explanation
            explanation = lrp.lrp(inp, x)

            with tf.Session() as s:
                s.run(tf.global_variables_initializer())
                s.run([assign_beta, assign_gamma, assign_mean, assign_variance])

                input = np.array([[[-0.3597203655, 2.416366089, -0.7543762749, 0.8718006654, -2.221761776],
                                   [1.099448308, 0.4108163696, 1.798067039, -0.6544576652, -0.6968107745],
                                   [-0.5612699962, -0.2597267932, 0.06325442832, -1.236885473, 0.9369620591],
                                   [-0.06784464057, -0.004403155247, -1.195337879, -0.528265092, -0.1020843691]],
                                  [[0.8766405077, 0.522839272, 0.4197016166, -1.497174712, 0.05348117451],
                                   [0.08739119149, -0.9997059536, -0.6212993685, 0.04027413639, 0.3979749684],
                                   [0.06180908495, -0.5322826252, -1.585670194, -0.5220654844, -1.096597863],
                                   [1.003261811, -1.865129316, -1.134796217, -0.1038509194, -0.8933464003]]])

                # Shape: (2, 4, 4, 5)
                expected_relevances = np.array(
                    [[[[0.0, 1.07794025, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0]],

                      [[0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.1832650698, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0]],

                      [[0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0]],

                      [[0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0]]],
                     [[[0.0, 0.2332384558, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0]],
                      [[0.0, 0.0, 0.0, 0.0, 0.0],
                       [-0.2035572695, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0]],
                      [[0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0]],
                      [[0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0007275465407, 0.0, 0.0, 0.0, 0.0]]]])
                relevances = s.run(explanation, feed_dict={inp: input})
                self.assertTrue(np.allclose(expected_relevances, relevances, rtol=1e-03, atol=1e-03),
                                msg="The relevances do not match the expected")
