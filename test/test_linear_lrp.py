import tensorflow as tf
import numpy as np

from configuration import LRPConfiguration, AlphaBetaConfiguration, FlatConfiguration, WWConfiguration, LAYER, \
    EpsilonConfiguration, BIAS_STRATEGY
from lrp import lrp
import unittest


# Class for testing that inherits from the unittest.TestCase class
class LinearLRPTest(unittest.TestCase):
    # Test case that builds a simple two layer linear network, finds the relevance
    # and compares them to the results obtained by calculating the same example by hand
    def test_linear_alpha(self):
        expected_result = [[0., 0., 2.2352, 29.8039]]
        self.do_test_with_config_and_result(expected_result)

    def test_linear_with_beta(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.LINEAR, AlphaBetaConfiguration(alpha=2, beta=-1))

        expected_result = [[-80.14545455, 9.566433566, -28.36791444, 125.5933087]]

        self.do_test_with_config_and_result(expected_result, config)

    def test_linear_lrp_alpha_beta_equal_bias(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.LINEAR, AlphaBetaConfiguration(alpha=2, beta=-1, bias_strategy=BIAS_STRATEGY.ALL))

        expected_result = [[-70.90120207, 14.94277618, -27.09395311, 121.052379]]

        self.do_test_with_config_and_result(expected_result, config)

    def test_linear_lrp_alpha_beta_active_bias(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.LINEAR, AlphaBetaConfiguration(alpha=2, beta=-1, bias_strategy=BIAS_STRATEGY.ACTIVE))

        expected_result = [[-83.6, 21.92307692, -47.5372549, 147.214178]]

        self.do_test_with_config_and_result(expected_result, config)


    def test_linear_lrp_alpha_beta_ignore_bias(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.LINEAR, AlphaBetaConfiguration(alpha=2, beta=-1, bias_strategy=BIAS_STRATEGY.IGNORE))

        expected_result = [[-83.6, 22.8, -57.79534884, 156.5953488]]

        self.do_test_with_config_and_result(expected_result, config)

    def test_linear_with_flat(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.LINEAR, FlatConfiguration())

        expected_result = [[9.5, 9.5, 9.5, 9.5]]

        self.do_test_with_config_and_result(expected_result, config)

    def test_linear_with_ww(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.LINEAR, WWConfiguration())

        expected_result = [[1.80952381, 13.68, 13.75238095, 8.758095238]]

        self.do_test_with_config_and_result(expected_result, config)

    def test_linear_with_epsilon_and_bias(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.LINEAR, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.ALL))

        expected_result = [[1.333333333, -14.06802721, 21.0521542, 29.68253968]]

        self.do_test_with_config_and_result(expected_result, config)

    def test_linear_with_epsilon_without_bias(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.LINEAR, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.NONE))

        expected_result = [[0., -12, 21, 32]]

        self.do_test_with_config_and_result(expected_result, config)


    def test_linear_with_epsilon_ignore_bias(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.LINEAR, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.IGNORE, epsilon=1e-12))

        expected_result = [[-5302325581397, 31813953488372, -47720930232555, 21209302325624]]

        self.do_test_with_config_and_result(expected_result, config)

    def test_linear_lrp_epsilon_active_bias(self):
        # Prepare configuration of linear layer
        config = LRPConfiguration()
        config.set(LAYER.LINEAR, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.ACTIVE))

        expected_result = [[1.340986395, -19.125, 28.75255102, 27.03146259]]

        self.do_test_with_config_and_result(expected_result, config)

    def do_test_with_config_and_result(self, expectex_result, config=LRPConfiguration()):
        # Get a tensorflow graph
        g = tf.Graph()
        # Set the graph as default
        with g.as_default():
            # Create a placeholder for the input
            inp = tf.placeholder(tf.float32, shape=(1, 4))

            # Create the first linear layer
            with tf.name_scope('linear'):
                # Set the weights
                weights = tf.constant([
                    [1, 2, -2],
                    [3, -2, 0],
                    [3, -6, 1],
                    [-1, 1, 10]], dtype=tf.float32)

                # Set the bia
                biases = tf.constant([[-3, 2, 8]], dtype=tf.float32)

                # Multiply the input and the weights
                mul = tf.matmul(inp, weights)

                # Add the bias
                activation = mul + biases

            # Create the second linear layer
            with tf.name_scope('linear'):
                # Set the weights
                weights = tf.constant(
                    np.array([
                        [2, 0],
                        [0, 8],
                        [1, 1]]), dtype=tf.float32)

                # Set the bias
                biases = tf.constant([[-5, 8]], dtype=tf.float32)

                # Multiply the input and the weights
                mul = tf.matmul(activation, weights)

                # Add the bias
                activation = mul + biases

            # Set the prediction to be equal to the activations of the last
            # layer (there is no softmax in this network)
            pred = activation

            # Calculate the relevance scores using lrp
            expl = lrp.lrp(inp, pred, config)

            # Run a tensorflow session to evaluate the graph
            with tf.Session() as sess:
                # Initialize the variables
                sess.run(tf.global_variables_initializer())

                # Run the operations of interest and feed an input to the network
                prediction, explanation = sess.run([pred, expl], feed_dict={inp: [[1, -2, 3, 4]]})

                # Check if the predictions has the right shape
                self.assertEqual(prediction.shape, (1, 2), msg="Should be able to do a linear forward pass")
                # Check if the relevance scores are correct (the correct values
                # are found by calculating the example by hand)
                self.assertTrue(
                    np.allclose(expectex_result, explanation, rtol=1e-03, atol=1e-03),
                    msg="Should be a good linear explanation")
