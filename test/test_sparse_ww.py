import tensorflow as tf
import numpy as np

from configuration import LRPConfiguration, WWConfiguration, LAYER
from lrp import lrp

import unittest


class TestSparseWW(unittest.TestCase):
    def test_ww_rule_sparse(self):
        with tf.Graph().as_default():
            indices = [[0, 2],
                       [0, 5],
                       [1, 0],
                       [1, 6]]
            values = tf.constant([2, 1, 1, 2.], dtype=tf.float32)
            inp = tf.SparseTensor(indices, values, (3, 7))
            sparse_tensor_reordered = tf.sparse_reorder(inp)
            sparse_tensor_reshaped = tf.sparse_reshape(sparse_tensor_reordered, (3, 7))

            W = tf.constant([[0.48237237, -2.17243375, -0.97473115],
                             [2.35669847, 3.11619017, 0.84322384],
                             [5.01346225, 1.69588809, -1.47801861],
                             [2.28595446, 0.24175502, -0.23067427],
                             [0.41892012, -1.44306815, 2.21516808],
                             [1.88990215, 1.46110879, 2.89949934],
                             [2.01381318, 2.12360494, 2.34057405]], dtype=tf.float32)

            b = tf.constant([1.47436833, -0.27795767, 4.28945125], dtype=tf.float32)
            out = tf.sparse_tensor_dense_matmul(sparse_tensor_reshaped, W) + b

            config = LRPConfiguration()
            config.set(LAYER.SPARSE_LINEAR, WWConfiguration())
            expl = lrp.lrp(inp, out, config)

            with tf.Session() as s:
                output, explanation = s.run([out, expl])

                expected_indices = indices
                expected_values = np.array([11.72503303, 1.666161958, 1.181770802, 6.814097394])

                self.assertTrue(np.allclose(expected_indices, explanation.indices))
                self.assertTrue(np.allclose(expected_values, explanation.values))
