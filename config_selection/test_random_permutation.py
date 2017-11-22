import tensorflow as tf
import unittest

from config_selection.random_permutation import get_random_relevance

class TestRandomPermutation(unittest.TestCase):
    def runTest(self):
        with tf.Graph().as_default():
            X_indices = tf.constant([
                [0, 1, 1],
                [0, 1, 3],
                [0, 1, 4],
                [0, 2, 2]
            ], dtype=tf.int64)
            X_values = [1, 1, 0, 2.]
            X = tf.SparseTensor(X_indices, X_values, (1, 4, 5))

            R = get_random_relevance(X)

            with tf.Session() as s:
                sum = s.run(tf.reduce_sum(R.values))
                self.assertTrue(sum == 1.)
