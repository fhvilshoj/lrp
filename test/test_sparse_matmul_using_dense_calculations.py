import tensorflow as tf
import numpy as np
import unittest

from configuration import LRPConfiguration, LAYER, AlphaBetaConfiguration, BIAS_STRATEGY, EpsilonConfiguration
from lrp import lrp
from lrp.nice_to_have import dense_to_sparse


# noinspection PyTypeChecker
class TestSparseMatMulUsingDenseCalculations(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self._set_weights_and_input()

    def setUp(self):
        super().setUp()
        self._set_weights_and_input()

    def _set_weights_and_input(self):
        self.input = np.array([[1., -2, 3, 4]])
        self.W1 = np.array([[1., 2, -2],
                            [3, -2, 0],
                            [3, -6, 1],
                            [-1, 1, 10]])

        self.b1 = np.array([[-3., 2, 8]])

        self.W2 = np.array([[2., 0],
                            [0, 8],
                            [1, 1]])
        self.b2 = np.array([[-5., 8]])

    def test_linear_alpha(self):
        expected_result = [[0., 0., 2.2352, 29.8039]]
        self._do_test(expected_result)

    def test_linear_with_beta_no_bias(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.SPARSE_LINEAR, AlphaBetaConfiguration(alpha=2, beta=-1))
        config.set(LAYER.LINEAR, AlphaBetaConfiguration(alpha=2, beta=-1))

        expected_result = [[-80.14545455, 9.566433566, -28.36791444, 125.5933087]]

        self._do_test(expected_result, config)

    def test_linear_with_beta_ignore_bias(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.SPARSE_LINEAR, AlphaBetaConfiguration(alpha=2, beta=-1, bias_strategy=BIAS_STRATEGY.IGNORE))
        config.set(LAYER.LINEAR, AlphaBetaConfiguration(alpha=2, beta=-1, bias_strategy=BIAS_STRATEGY.IGNORE))

        expected_result = [[-83.6, 22.8, -57.79534884, 156.5953488]]

        self._do_test(expected_result, config)

    def test_linear_lrp_alpha_beta_active_bias(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.SPARSE_LINEAR, AlphaBetaConfiguration(alpha=2, beta=-1, bias_strategy=BIAS_STRATEGY.ACTIVE))
        config.set(LAYER.LINEAR, AlphaBetaConfiguration(alpha=2, beta=-1, bias_strategy=BIAS_STRATEGY.ACTIVE))

        expected_result = [[-83.6, 21.92307692, -47.5372549, 147.214178]]

        self._do_test(expected_result, config)

    def test_linear_with_epsilon_without_bias(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.SPARSE_LINEAR, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.NONE))
        config.set(LAYER.LINEAR, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.NONE))

        expected_result = [[0., -12, 21, 32]]

        self._do_test(expected_result, config)

    def test_linear_with_epsilon_ignore_bias(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.SPARSE_LINEAR, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.IGNORE, epsilon=1e-12))
        config.set(LAYER.LINEAR, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.IGNORE, epsilon=1e-12))

        expected_result = [[-5302325581397, 31813953488372, -47720930232555, 21209302325624]]

        self._do_test(expected_result, config)

    def test_linear_lrp_epsilon_active_bias(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.SPARSE_LINEAR, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.ACTIVE))
        config.set(LAYER.LINEAR, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.ACTIVE))

        expected_result = [[1.340986395, -19.125, 28.75255102, 27.03146259]]

        self._do_test(expected_result, config)

    def _do_test(self, expected_result, config=None):
        # Make sure that expected_result is an np array
        if not type(expected_result).__module__ == np.__name__:
            expected_result = np.array(expected_result)

        with tf.Graph().as_default():
            inputs, indices, _ = dense_to_sparse(self.input)
            sparse_tensor_reordered = tf.sparse_reorder(inputs)
            sparse_tensor_reshaped = tf.sparse_reshape(sparse_tensor_reordered, self.input.shape)

            W = tf.constant(self.W1, dtype=tf.float32)
            b = tf.constant(self.b1, dtype=tf.float32)

            # Sparse layer
            logits = tf.sparse_tensor_dense_matmul(sparse_tensor_reshaped, W) + b

            # Dense layer
            logits = logits @ tf.constant(self.W2, tf.float32) + tf.constant(self.b2, tf.float32)

            explanation = lrp.lrp(inputs, logits, config)

            with tf.Session() as s:
                expl = s.run(explanation)
                self.assertTrue(np.all(np.equal(indices, expl.indices)),
                                "expected indices did not equal actual indices")
                self.assertTrue(np.allclose(expl.values, expected_result.reshape((-1)), rtol=1.e-3, atol=1.e-3),
                                "expected indices did not equal actual indices")
