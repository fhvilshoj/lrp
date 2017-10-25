import tensorflow as tf
import numpy as np
from lrp import lrp
import unittest


# Class for testing that inherits from the unittest.TestCase class
class ConvolutionBiasLRPTest(unittest.TestCase):
    # Test case that builds a simple one layer convolution network with bias,
    # finds the relevance and compares them to the results obtained by
    # calculating the same example by hand
    def runTest(self):
        # Get a tensorflow graph
        g = tf.Graph()
        # Set the graph as default
        with g.as_default():
            # Create a placeholder for the input
            inp = tf.placeholder(tf.float32, shape=(1, 2, 2, 2))

            # Create the convolutional layer
            # Set the filters and biases
            filters = tf.constant([[[[2., 1, 1],
                                     [1., 1, 0]],
                                    [[2., 2, 1],
                                     [1., 1, 1]]],
                                   [[[1., -1, 1],
                                     [0., 1, 0]],
                                    [[2., 0, 0],
                                     [0., -1, 1]]]], dtype=tf.float32)
            bias = tf.constant([1, 2, -2], dtype=tf.float32)

            # Perform the convolution
            activation = tf.nn.conv2d(inp, filters, [1, 1, 1, 1], "SAME")

            # Add bias
            activation = tf.nn.bias_add(activation, bias)

            # Reshape predictions to (batch_size, pred. pr. sample, classes) to fit the required input shape of
            # the framework
            pred = tf.reshape(activation, (1, 2, 6))

            # Calculate the relevance scores using lrp
            expl = lrp.lrp(inp, pred)

            # Run a tensorflow session to evaluate the graph
            with tf.Session() as sess:
                # Initialize the variables
                sess.run(tf.global_variables_initializer())

                # Run the operations of interest and feed an input to the network
                prediction, explanation = sess.run([pred, expl],
                                                   feed_dict={inp: [[[[1., 0.],
                                                                      [-1., 2.]],
                                                                     [[2., -1.],
                                                                      [3., 0.]]]]
                                                              })

                # Check if the relevance scores are correct
                self.assertTrue(
                    np.allclose([[[[[1.692307692, 0],
                                    [0, 1.692307692]],

                                   [[1.692307692, 0],
                                    [5.076923077, 0]]],


                                  [[[0, 0],
                                    [0, 0]],

                                   [[3.636363636, 0],
                                    [5.454545455, 0]]]]], explanation,
                                rtol=1e-03, atol=1e-03),
                    msg="The calculated relevances do not match the expected relevances")
