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
                weights = tf.constant([[[[9, 5],
                                         [0, -2]],
                                        [[2, 0],
                                         [4, 0]]],
                                       [[[4, 6],
                                         [1, 0]],
                                        [[-8, 7],
                                         [2, 3]]]],
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
                print(activation)

            # Flatten the activations to get something of the shape (batch_size, predictions_per_sample, classes)
            # as required by the lrp framework
            final_output = tf.reshape(activation, [1, 2, 6])

            # Calculate the relevance scores using lrp
            expl = lrp.lrp(inp, final_output)

            # Run a tensorflow session to evaluate the graph
            with tf.Session() as sess:
                # Initialize the variables
                sess.run(tf.global_variables_initializer())

                # Run the operations of interest and feed an input to the network
                prediction, explanation = sess.run([final_output, expl],
                                                   feed_dict={inp: [[[[-1., 0.],
                                                                      [1., 2.]],
                                                                     [[12., 9.],
                                                                      [13., 6.]]]]
                                                              })

                # The expected relevances (calculated in Google sheet)
                expected_explanation = np.array([[[[[0, 0],
                                                    [20.77998952, 0]],

                                                   [[162.531828, 0],
                                                    [480.9247907, 48.85034831]]],

                                                  [[[0, 0],
                                                    [0, 0]],

                                                   [[258, 0],
                                                    [339, 48]]]]])

                # Check if the relevance scores are correct
                self.assertTrue(
                    np.allclose(expected_explanation, explanation, rtol=1e-03, atol=1e-03),
                    msg="Should be a good convolutional explanation")
