import tensorflow as tf
import numpy as np
import unittest

from config_selection.pertubation import Pertuber


class PertubationTest(unittest.TestCase):
    def runTest(self):
        with tf.Graph().as_default():
            X_indices = tf.constant([
                [0, 1, 1],
                [0, 1, 3],
                [0, 1, 4],
                [0, 2, 2],

                [1, 1, 1],
                [1, 1, 3],
                [1, 1, 4],
                [1, 2, 2]
            ], dtype=tf.int64)
            X_values = [1, 1, 0, 2., 1, 1, 0, 2.]
            X = tf.SparseTensor(X_indices, X_values, (2, 4, 5))

            R_indices = tf.constant([
                [0, 1, 1],
                [0, 1, 3],
                [1, 1, 1],
                [1, 1, 3]
            ], dtype=tf.int64)
            R_values = [0.2, 0.3, 0.2, 0.3]
            R = tf.SparseTensor(R_indices, R_values, (2, 4, 5))

            def _scale_sum(x):
                x = tf.sparse_reshape(x, (8, 5))
                scalar = tf.get_variable('scalar',
                                         dtype=tf.float32,
                                         shape=(5, 2),
                                         initializer=tf.constant_initializer(
                                             [1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
                                             dtype=tf.float32))
                # shape = (5, 2)
                mult = tf.sparse_tensor_dense_matmul(x, scalar)
                mult = tf.reshape(mult, (2, 4, 2))

                # mult = tf.Print(mult, [mult], summarize=100)
                # reduce sum
                # shape = (1, 2)
                res = tf.reduce_sum(mult, 1)
                # res = tf.Print(res, [res], "RES: ", summarize=10)
                return {'y_hat': res}

            scale_sum = tf.make_template('scale_sum', _scale_sum)

            def _some_other(x):
                # Test to see that we can do decorators around templates
                return scale_sum(x)

            pertuber = Pertuber(X, R, 2, **{'pertubations': 2})

            res = pertuber.build_pertubation_graph(_some_other)

            with tf.Session() as s:
                s.run(tf.global_variables_initializer())
                results = s.run(res)
                expected = np.array([[[2, 2], [2, 1], [2, 0]], [[2, 2], [2, 1], [2, 0]]])

                print(results)

                # Check for shape and actual result
                self.assertEqual(expected.shape, results.shape,
                                 "Shapes of expected relevance and relevance should be equal")
                self.assertTrue(np.allclose(expected, results, rtol=1e-03, atol=1e-03),
                                "The relevances do not match")
