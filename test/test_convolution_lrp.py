import tensorflow as tf
import numpy as np
from lrp import lrp
import unittest


# Class for testing that inherits from the unittest.TestCase class
class ConvolutionLRPTest(unittest.TestCase):
    # Test case that builds a simple one layer convolution network, finds the relevance and compares them to the results obtained by calculating the same example by hand
    def runTest(self):
        # Get a tensorflow graph
        g = tf.Graph()
        # Set the graph as default
        with g.as_default():
            # Create a placeholder for the input
            inp = tf.placeholder(tf.float32, shape=(1, 2, 2, 2))

            # Create the convolutional layer
            # Set the weights
            weights = tf.constant([[[[1, 2], [1, -1]],
                                    [[2, 2], [2, 0]]],
                                   [[[1, 0], [0, 1]],
                                    [[1, 0], [1, -1]]]
                                   ], dtype=tf.float32)

            # Perform the convolution
            activation = tf.nn.conv2d(inp, weights, [1,1,1,1], "SAME")

            # Set the prediction to be equal to the activations of the last layer (there is no softmax in this network)
            pred = activation

            # Calculate the relevance scores using lrp
            R_mock = tf.constant([[[[3, 4], [1, 1]], [[2, 3], [1, 0]]]], dtype=tf.float32)
            expl = lrp._lrp(pred, R_mock)

            # Run a tensorflow session to evaluate the graph
            with tf.Session() as sess:
                # Initialize the variables
                sess.run(tf.global_variables_initializer())

                # Run the operations of interest and feed an input to the network
                prediction, explanation = sess.run([pred, expl], feed_dict={inp: [[[[1, -1], [2, 3]],
                                                                                   [[0, 2], [-1, 0]]]
                                                                                  ]})

                # Check if the predictions has the right shape
                self.assertEqual(prediction.shape, (1,2,2,2), msg="Should be able to do a convolutional forward pass")

                # Check if the explanation has the right shape
                self.assertEqual(explanation.shape, inp.shape, msg="Should be a wellformed explanation")

                # Check if the relevance scores are correct (the correct values are found by calculating the example by hand)
                print(explanation[0])
                self.assertTrue(
                    np.allclose(explanation[0], [[[1.25, 0], [6 + 1/28, 4 + 9/14]],
                                                 [[0, 3 + 1/14], [0, 0]]], rtol=1e-03, atol=1e-01),
                    msg="Should be a good convolutional explanation")
