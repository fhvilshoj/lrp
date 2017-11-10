import tensorflow as tf
import numpy as np
import unittest

from configuration import *
from lrp import lrp


class TestSparseMatmul(unittest.TestCase):
    def test_without_bias(self):
        with tf.Graph().as_default() as g:
            values = tf.placeholder(tf.float32, (10,), 'sparse_values')
            indices = tf.placeholder(tf.int64, (10, 2), 'sparse_indices')
            shape = tf.placeholder(tf.int64, (2,), 'sparse_shape')

            weights = tf.constant(
                [[0.44168384, -0.14901647, 0.18877962, 0.77717885, 0.28132105],
                 [0.29898775, 0.78258813, 0.02738412, -0.39713207, 0.7206406],
                 [0.0916185, -0.48300626, 0.61379333, -0.164125, 0.69977015],
                 [-0.27808439, 0.89466766, 0.41370821, 0.59659951, -0.3470153]]
            )

            sparse_tensor = tf.SparseTensor(indices, values, shape)

            # Copy of shape (40, 6)
            copy = tf.sparse_tensor_dense_matmul(sparse_tensor, weights)

            out = copy

            R = lrp.lrp(sparse_tensor, out)

            with tf.Session() as s:
                _indices = [[0, 0],
                            [0, 3],
                            [2, 2],
                            [2, 3],
                            [3, 3],
                            [4, 1],
                            [5, 1],
                            [5, 2],
                            [6, 0],
                            [6, 2]]
                _values = [1.411525599, -0.4125681457, 0.007600154656, 0.1866370931, 1.271072685,
                           0.3672089764, 0.7266022409, 0.237290756, 1.210895782, 0.8650810186]
                _shape = (7, 4)

                explanation = s.run(R, feed_dict={values: _values,
                                                  indices: _indices,
                                                  shape: _shape})

                expected_indices = [[0, 0],
                                    [0, 3],
                                    [2, 2],
                                    [2, 3],
                                    [3, 3],
                                    [4, 1],
                                    [5, 1],
                                    [5, 2],
                                    [6, 0],
                                    [6, 2]]
                expected_values = [0.850869893, 0.0, 0.0, 0.1633072495, 1.137187628,
                                   0.2873733863, 0.5236190745, 0.1660489874, 0.3406504681, 0.6053578722]

                self.assertTrue(np.all(np.equal(expected_indices, explanation.indices)),
                                "expected indices did not equal actual indices")
                self.assertTrue(np.allclose(expected_values, explanation.values, rtol=1.e-3, atol=1.e-3),
                                "expected indices did not equal actual indices")

    def test_with_bias(self):
        with tf.Graph().as_default() as g:
            values = tf.placeholder(tf.float32, (10,), 'sparse_values')
            indices = tf.placeholder(tf.int64, (10, 2), 'sparse_indices')
            shape = tf.placeholder(tf.int64, (2,), 'sparse_shape')

            weights = tf.constant(
                [[0.44168384, -0.14901647, 0.18877962, 0.77717885, 0.28132105],
                 [0.29898775, 0.78258813, 0.02738412, -0.39713207, 0.7206406],
                 [0.0916185, -0.48300626, 0.61379333, -0.164125, 0.69977015],
                 [-0.27808439, 0.89466766, 0.41370821, 0.59659951, -0.3470153]]
            )
            bias = tf.constant([-0.16923287, 1.10315915, 0.78755005, -0.03560105, 0.26801184])

            sparse_tensor = tf.SparseTensor(indices, values, shape)

            copy = tf.sparse_tensor_dense_matmul(sparse_tensor, weights) + bias

            out = copy

            R = lrp.lrp(sparse_tensor, out)

            with tf.Session() as s:
                _indices = [[0, 0],
                            [0, 3],
                            [2, 2],
                            [2, 3],
                            [3, 3],
                            [4, 1],
                            [5, 1],
                            [5, 2],
                            [6, 0],
                            [6, 2]]
                _values = [1.411525599, -0.4125681457, 0.007600154656, 0.1866370931, 1.271072685,
                           0.3672089764, 0.7266022409, 0.237290756, 1.210895782, 0.8650810186]
                _shape = (7, 4)

                explanation = s.run(R, feed_dict={values: _values,
                                                  indices: _indices,
                                                  shape: _shape})

                expected_indices = [[0, 0],
                                    [0, 3],
                                    [2, 2],
                                    [2, 3],
                                    [3, 3],
                                    [4, 1],
                                    [5, 1],
                                    [5, 2],
                                    [6, 0],
                                    [6, 2]]
                expected_values = [0.2233167596, 0.0, 0.0, 0.1664955752, 1.137187628, 0.2873733863, 0.529646685, 0.0,
                                   0.228592451, 0.5309809592]

                self.assertTrue(np.all(np.equal(expected_indices, explanation.indices)),
                                "expected indices did not equal actual indices")
                self.assertTrue(np.allclose(expected_values, explanation.values, rtol=1.e-3, atol=1.e-3),
                                "expected indices did not equal actual indices")

    def test_with_bias_and_reshape(self):
        with tf.Graph().as_default() as g:
            values = tf.placeholder(tf.float32, (10,), 'sparse_values')
            indices = tf.placeholder(tf.int64, (10, 2), 'sparse_indices')
            shape = tf.placeholder(tf.int64, (2,), 'sparse_shape')

            weights = tf.constant(
                [[0.44168384, -0.14901647, 0.18877962, 0.77717885, 0.28132105],
                 [0.29898775, 0.78258813, 0.02738412, -0.39713207, 0.7206406],
                 [0.0916185, -0.48300626, 0.61379333, -0.164125, 0.69977015],
                 [-0.27808439, 0.89466766, 0.41370821, 0.59659951, -0.3470153]]
            )
            bias = tf.constant([-0.16923287, 1.10315915, 0.78755005, -0.03560105, 0.26801184])

            sparse_tensor = tf.SparseTensor(indices, values, shape)
            sparse_tensor_reordered = tf.sparse_reorder(sparse_tensor)
            sparse_tensor_reshaped = tf.sparse_reshape(sparse_tensor_reordered, (7, 4))

            copy = tf.sparse_tensor_dense_matmul(sparse_tensor_reshaped, weights) + bias

            out = copy

            R = lrp.lrp(sparse_tensor, out)

            with tf.Session() as s:
                _indices = [[0, 0],
                            [1, 1],
                            [5, 0],
                            [5, 1],
                            [7, 1],
                            [8, 1],
                            [10, 1],
                            [11, 0],
                            [12, 0],
                            [13, 0]]

                _values = [1.411525599, -0.4125681457, 0.007600154656, 0.1866370931, 1.271072685,
                           0.3672089764, 0.7266022409, 0.237290756, 1.210895782, 0.8650810186]
                _shape = (14, 2)

                explanation = s.run(R, feed_dict={values: _values,
                                                  indices: _indices,
                                                  shape: _shape})

                expected_indices = [[0, 0],
                                    [1, 1],
                                    [5, 0],
                                    [5, 1],
                                    [7, 1],
                                    [8, 1],
                                    [10, 1],
                                    [11, 0],
                                    [12, 0],
                                    [13, 0]]
                expected_values = [0.2233167596, 0.0, 0.0, 0.1664955752, 1.137187628, 0.2873733863, 0.529646685, 0.0,
                                   0.228592451, 0.5309809592]
                self.assertTrue(np.all(np.equal(expected_indices, explanation.indices)),
                                "expected indices did not equal actual indices")
                self.assertTrue(np.allclose(expected_values, explanation.values, rtol=1.e-3, atol=1.e-3),
                                "expected indices did not equal actual indices")

    def test_epsilon_no_bias(self):
        # Prepare configuration
        config = LRPConfiguration()
        config.set(LAYER.SPARSE_LINEAR, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.NONE))

        expected_values = [[0, 0.03214352877, 0.2618736551, -0.6062815296, 0.384333896,
                            1.01422683, 0.7098777496, 0.2303268615, 0.105638811, -0.1920372947]]

        self._do_test_with_configuration(expected_values, config)

    def test_alpha_and_beta_no_bias(self):
        # Prepare configuration
        config = LRPConfiguration()
        config.set(LAYER.SPARSE_LINEAR, AlphaBetaConfiguration(alpha=2, beta=-1, bias_strategy=BIAS_STRATEGY.NONE))

        expected_values = [[0, 0.03244332333, 0.2643160845, -0.6176972702, 0.387918481,
                            1.882732545, 1.317762361, 0.4275610398, 0.1960997496, -2.481142585]]

        self._do_test_with_configuration(expected_values, config)


    def test_epsilon_active_bias(self):
        config = LRPConfiguration()
        config.set(LAYER.SPARSE_LINEAR, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.ACTIVE))

        expected_values = [[0, 0.1685504588, 0.3982805851, -0.4698745996, 0.5207408259,
                            1.136848756, 0.8324996751, 0.352948787, 0.2282607365, -0.06941536923]]

        self._do_test_with_configuration(expected_values, config)


    def test_alpha_and_beta_active_bias(self):
        config = LRPConfiguration()
        config.set(LAYER.SPARSE_LINEAR, AlphaBetaConfiguration(alpha=2, beta=-1, bias_strategy=BIAS_STRATEGY.ACTIVE))

        expected_values = [[0,           0.2160155405, 0.4478883017, -0.6176972702,  0.5714906982,
                            2.167264914, 1.60229473,   0.7120934085,  0.4806321183, -2.481142585]]

        self._do_test_with_configuration(expected_values, config)


    def _do_test_with_configuration(self, expected_values, config):
        with tf.Graph().as_default():
            values = tf.placeholder(tf.float32, (10,), 'sparse_values')
            indices = tf.placeholder(tf.int64, (10, 2), 'sparse_indices')
            shape = tf.placeholder(tf.int64, (2,), 'sparse_shape')

            weights = tf.constant(
                [[-1.229296047, 0.608532863, -0.05627508999],
                 [-0.5737899975, -1.706384751, 0.2477672936],
                 [0.7408662702, 0.4972367935, 0.1137446909],
                 [0.4407868665, 0.2622986811, 0.7576768343],
                 [0.3393422057, -0.5966289301, -0.9215300495]]
            )
            bias = tf.constant([[0.54562772, 0.6131096274, -0.3660361247]])

            sparse_tensor = tf.SparseTensor(indices, values, shape)
            sparse_tensor_reordered = tf.sparse_reorder(sparse_tensor)
            sparse_tensor_reshaped = tf.sparse_reshape(sparse_tensor_reordered, (2, 5))

            copy = tf.sparse_tensor_dense_matmul(sparse_tensor_reshaped, weights) + bias

            out = copy

            R = lrp.lrp(sparse_tensor, out, config)

            with tf.Session() as s:
                _indices = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4]]

                _values = [0, -0.05601967429, 0.3534695338, -1.3754528, 1.132585012, 1.666675527, -0.4160127129,
                           0.4632136328, 0.4027424405, 0.3218705715]
                _shape = (2, 5)

                explanation = s.run(R, feed_dict={values: _values,
                                                  indices: _indices,
                                                  shape: _shape})
                self.assertTrue(np.all(np.equal(_indices, explanation.indices)),
                                "expected indices did not equal actual indices")
                self.assertTrue(np.allclose(expected_values, explanation.values, rtol=1.e-3, atol=1.e-3),
                                "expected indices did not equal actual indices")
