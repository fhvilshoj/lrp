import tensorflow as tf
import numpy as np
from lrp import lrp
import unittest


# Class for testing that inherits from the unittest.TestCase class
class ConvolutionLRPTest(unittest.TestCase):
    # Test case that builds a simple one layer convolution network,
    # finds the relevance and compares them to the results obtained
    # by calculating the same example by hand
    def test_simple_convolution_default_config(self):
        # Get a tensorflow graph
        g = tf.Graph()
        # Set the graph as default
        with g.as_default():
            # Create a placeholder for the input
            inp = tf.placeholder(tf.float32, shape=(1, 2, 2, 2))

            # Create the convolutional layer
            # Set the filters
            filters = tf.constant([[[[2., 1, 1],
                                     [1., 1, 0]],
                                    [[2., 2, 1],
                                     [1., 1, 1]]],
                                   [[[1., -1, 1],
                                     [0., 1, 0]],
                                    [[2., 0, 0],
                                     [0., -1, 1]]]], dtype=tf.float32)

            # Perform the convolution
            activation = tf.nn.conv2d(inp, filters, [1, 1, 1, 1], "SAME")

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
                    np.allclose([[[[[1.666666667, 0],
                                    [0, 1.666666667]],

                                   [[1.666666667, 0],
                                    [5, 0]]],


                                  [[[0, 0],
                                    [0, 0]],

                                   [[3.6, 0],
                                    [5.4, 0]]]]], explanation, rtol=1e-03, atol=1e-03),
                    msg="Should be a good convolutional explanation")
