import tensorflow as tf
import numpy as np
import unittest
from lrp import lrp


class Convolution2LayersLRPTest(unittest.TestCase):
    def runTest(self):
        # Construct tensorflow graph
        g = tf.Graph()
        # Use tensorflows default graph
        with g.as_default():
            # Create a placeholder for the input
            inp = tf.placeholder(tf.float32, shape=(1, 2, 2, 2))

            # Create first convolutional layer (simple layer that copies the
            # input to the next layer and copies relevances backwards in the
            # same manner (so we were able to reuse calculations from earlier
            # test case, `test_convolution_with_bias.py`)
            with tf.name_scope('conv1'):
                weights = tf.constant([[[[1, 0],
                                         [0, 1]],
                                        [[0, 0],
                                         [0, 0]]],
                                       [[[0, 0],
                                         [0, 0]],
                                        [[0, 0],
                                         [0, 0]]]],
                                      dtype=tf.float32)
                activation = tf.nn.conv2d(inp, weights, [1, 1, 1, 1], "SAME")

            # Create the second convolutional layer equal to `test_convolution_with_bias.py`
            with tf.name_scope('conv2'):
                # Set the weights and biases
                weights = tf.constant([[[[2., 1, 1],
                                         [1., 1, 0]],
                                        [[2., 2, 1],
                                         [1., 1, 1]]],
                                       [[[1., -1, 1],
                                         [0., 1, 0]],
                                        [[2., 0, 0],
                                         [0., -1, 1]]]], dtype=tf.float32)
                bias = tf.constant([1, 2, -2], dtype=tf.float32)

                # Perform the convolution
                activation = tf.nn.conv2d(activation, weights, [1, 1, 1, 1], "SAME")

                # Add bias
                activation = tf.nn.bias_add(activation, bias)

            # Set the prediction to be equal to the activations of the last layer
            # (there is no softmax in this network)
            pred = activation

            # Calculate the relevance scores using lrp
            R_mock = tf.constant([[[[3., 2., 1],
                                    [4., 3., 2]],
                                   [[1., 1., 3],
                                    [1., 0., 4]]]], dtype=tf.float32)
            expl = lrp._lrp(inp, pred, R_mock)

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

                # Check if the predictions has the right shape
                self.assertEqual(prediction.shape, (1, 2, 2, 3),
                                 msg="Should be able to do a convolutional forward pass")

                # Check if the explanation has the right shape
                self.assertEqual(explanation.shape, inp.shape, msg="Should be a wellformed explanation")

                # Check if the relevance scores are correct (the correct values are
                # found by calculating the example by hand)
                self.assertTrue(
                    np.allclose(explanation[0], [[[[1.06, 0],
                                                   [0, 4.49]],
                                                  [[2.62, 0],
                                                   [13.19, 0]]]], rtol=1e-01, atol=1e-01),
                    msg="Should be a good convolutional explanation")
