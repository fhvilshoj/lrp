import tensorflow as tf
import unittest

from configuration import AlphaBetaConfiguration, LRPConfiguration, LAYER, FlatConfiguration, EpsilonConfiguration, \
    LOG_LEVEL, BIAS_STRATEGY, RULE, BaseConfiguration
from lrp import lrp
import numpy as np


class TestBatchNormalization(unittest.TestCase):
    def test_flat(self):
        g = tf.Graph()
        with g.as_default():
            inp = tf.placeholder(tf.float32, (3, 5))

            W = tf.constant([[1.493394546, 0.5987773779],
                             [0.7321155851, 1.23063763],
                             [2.488971816, 0.9885881838],
                             [0.9965223115, 0.8397688134],
                             [2.089138346, 0.8398492639]]
                            , dtype=tf.float32)
            b = [[2.665864718, 0.8793648172]]

            to_normalize = inp @ W + b

            x = tf.contrib.layers.batch_norm(to_normalize, is_training=False, scale=True)
            vars = tf.global_variables()
            beta = next(i for i in vars if 'beta' in i.name)
            gamma = next(i for i in vars if 'gamma' in i.name)
            mean = next(i for i in vars if 'mean' in i.name)
            variance = next(i for i in vars if 'variance' in i.name)

            b = tf.constant([0.8481817169, -1.118752611], dtype=tf.float32)
            assign_beta = tf.assign(beta, b)
            g = tf.constant([0.1005506696, 0.308355701], dtype=tf.float32)
            assign_gamma = tf.assign(gamma, g)
            m = tf.constant([-0.8224766215, 0.9257031289], dtype=tf.float32)
            assign_mean = tf.assign(mean, m)
            v = tf.constant([0.7134228722, 1.065337135], dtype=tf.float32)
            assign_variance = tf.assign(variance, v)

            # Get the explanation
            config = LRPConfiguration()
            config.set(LAYER.LINEAR, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.NONE))
            config.set(LAYER.ELEMENTWISE_LINEAR, BaseConfiguration(RULE.IDENTITY))
            explanation = lrp.lrp(inp, x, config)

            with tf.Session() as s:
                s.run(tf.global_variables_initializer())
                s.run([assign_beta, assign_gamma, assign_mean, assign_variance])

                expected_relevances = np.array([[0.3331443396, 0.5080297851, 0.2213932963, 0.1821642756, 0.832728544],
                                                 [0.3800239444, 0.1179592254, 0.5207348458, 0.5280974564, 0.3505242006],
                                                 [0.5231591662, 0.0895191051, 0.6016423127, 0.423324388, -0.1957729137]]
)

                out, relevances = s.run([x, explanation],
                                   feed_dict={inp: [[1.187187323, 3.692928471, 0.4733755909, 0.9728313491, 2.121278175],
                                                    [1.302276662, 0.8245540266, 1.070689437, 2.712026357, 0.8586529453],
                                                    [1.57555465, 0.5499336234, 1.087157802, 1.910559011,
                                                     -0.4214631393]]})

                self.assertTrue(np.allclose(expected_relevances, relevances, rtol=1e-03, atol=1e-03),
                                msg="The relevances do not match the expected")

    def test_alpha_beta(self):
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

            # Get the explanation
            config = LRPConfiguration()
            config.set(LAYER.ELEMENTWISE_LINEAR, AlphaBetaConfiguration())
            explanation = lrp.lrp(inp, x, config)

            with tf.Session() as s:
                s.run(tf.global_variables_initializer())
                s.run([assign_beta, assign_gamma, assign_mean, assign_variance])

                expected_relevances = np.array(
                    [[0, 0, 0, 1, 0],
                     [0, 0.8921994513, 0, 0, 0]])
                relevances = s.run(explanation, feed_dict={inp: [[1, 0, 3, 4, 5],
                                                                 [1, 2, 3, 0, 4]]})

                self.assertTrue(np.allclose(expected_relevances, relevances, rtol=1e-03, atol=1e-03),
                                msg="The relevances do not match the expected")
