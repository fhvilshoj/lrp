import tensorflow as tf
import numpy as np

from configuration import LRPConfiguration, AlphaBetaConfiguration, LINEAR_LAYER
from lrp import lrp
import unittest


# Class for testing that inherits from the unittest.TestCase class
class LinearLRPTest(unittest.TestCase):
    # Test case that builds a simple two layer linear network, finds the relevance
    # and compares them to the results obtained by calculating the same example by hand
    def runTest(self):
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
            expl = lrp.lrp(inp, pred)

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
                    np.allclose([[0., 0., 2.2352, 29.8039]], explanation, rtol=1e-03, atol=1e-03),
                    msg="Should be a good linear explanation")

    def test_linear_with_beta(self):
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

            # Prepare configuration of linear layer
            config = LRPConfiguration()
            config.set(LINEAR_LAYER, AlphaBetaConfiguration(alpha=2, beta=-1))

            # Calculate the relevance scores using lrp
            expl = lrp.lrp(inp, pred, configuration=config)

            # Run a tensorflow session to evaluate the graph
            with tf.Session() as sess:
                # Initialize the variables
                sess.run(tf.global_variables_initializer())

                # Run the operations of interest and feed an input to the network
                prediction, explanation = sess.run([pred, expl], feed_dict={inp: [[1, -2, 3, 4]]})

                # Check if the predictions has the right shape
                self.assertEqual(prediction.shape, (1, 2), msg="Should be able to do a linear forward pass")

                print(explanation)

                # Check if the relevance scores are correct (the correct values
                # are found by calculating the example by hand)
                self.assertTrue(
                    np.allclose([[-80.14545455, 9.566433566, -28.36791444, 125.5933087]],
                                explanation,
                                rtol=1e-03, atol=1e-03),
                    msg="Should be a good linear explanation")
