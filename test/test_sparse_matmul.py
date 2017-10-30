import tensorflow as tf
import numpy as np
import unittest
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

                self.assertTrue(np.all(np.equal(expected_indices, explanation.indices)), "expected indices did not equal actual indices")
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
            bias = tf.constant([-0.16923287,  1.10315915,  0.78755005, -0.03560105,  0.26801184])

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
                expected_values = [0.2233167596, 0.0, 0.0, 0.1664955752, 1.137187628, 0.2873733863, 0.529646685, 0.0, 0.228592451, 0.5309809592]

                self.assertTrue(np.all(np.equal(expected_indices, explanation.indices)), "expected indices did not equal actual indices")
                self.assertTrue(np.allclose(expected_values, explanation.values, rtol=1.e-3, atol=1.e-3),
                                "expected indices did not equal actual indices")
