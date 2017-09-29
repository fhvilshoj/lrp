import tensorflow as tf
import numpy as np
from lrp import lrp
import unittest

# Class for testing that inherits from the unittest.TestCase class
class LinearLRPTest(unittest.TestCase):

    # Test case that builds a simple two layer linear network, finds the relevance and compares them to the results obtained by calculating the same example by hand
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

                # Set the bia
                biases = tf.constant([[-5, 8]], dtype=tf.float32)

                # Multiply the input and the weights
                mul = tf.matmul(activation, weights)

                # Add the bias
                activation = mul + biases

            # Set the prediction to be equal to the activations of the last layer (there is no softmax in this network)
            pred = activation

            # Calculate the relevance scores using lrp
            expl = lrp.lrp(pred)

            # Run a tensorflow session to evaluate the graph
            with tf.Session() as sess:
                # Initialize the variables
                sess.run(tf.global_variables_initializer())

                # Run the operations of interest and feed an input to the network
                prediction, explanation = sess.run([pred, expl], feed_dict={inp: [[1, -2, 3, 4]]})

                # Check if the predictions has the right shape
                self.assertEqual(prediction.shape, (1, 2), msg="Should be able to do a linear forward pass")

                # Check if the explanation has the right shape
                self.assertEqual(explanation.shape, inp.shape, msg="Should be a wellformed explanation")

                # Check if the relevance scores are correct (the correct values are found by calculating the example by hand)
                self.assertTrue(
                    np.allclose(explanation[0], [0., 0., 2.69, 35.87], rtol=1e-03, atol=1e-03),
                    msg="Should be a good linear explanation")
