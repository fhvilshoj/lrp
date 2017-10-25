import tensorflow as tf
import numpy as np
from lrp import lrp
import unittest


# Class for testing that inherits from the unittest.TestCase class
class MaxPoolLRPTest(unittest.TestCase):
    # Test case that builds a very simple network with a max pooling layer
    def runTest(self):
        # Get a tensorflow graph
        g = tf.Graph()
        # Set the graph as default
        with g.as_default():
            # Create a placeholder for the input
            inp = tf.placeholder(tf.float32, shape=(1, 6, 2, 1))

            # Create max pooling layer
            activation = tf.nn.max_pool(inp, [1, 2, 2, 1], [1, 2, 1, 1], "SAME")

            # Reshape predictions to (batch_size, pred. pr. sample, classes)
            pred = tf.reshape(activation, (1, 6, 1))

            # Get the explanation tensor
            expl = lrp.lrp(inp, pred)

            # Run a tensorflow session to evaluate the graph
            with tf.Session() as sess:
                # Initialize the variables
                sess.run(tf.global_variables_initializer())

                # Run the operations of interest and feed an input to the network
                prediction, explanation = sess.run([pred, expl],
                                                   feed_dict={inp: [[[[1],
                                                                      [2]],

                                                                     [[3],
                                                                      [4]],

                                                                     [[5],
                                                                      [6]],

                                                                     [[-1],
                                                                      [-2]],

                                                                     [[-3],
                                                                      [-4]],

                                                                     [[10],
                                                                      [0]]]]})

                # Check if the predictions has the right shape
                self.assertEqual(prediction.shape, (1, 6, 1),
                                 msg="Should be able to do a linear forward pass")

                # Check if the explanation has the right shape
                self.assertEqual((1, 6, 6, 2, 1), explanation.shape,
                                 msg="Should be a wellformed explanation")

                # Shape: (1, 6, 6, 2, 1)
                # i.e, (batch_size, pred. pr. sample, input_sizes ...)
                expected = np.array([[[[[0], [0]],
                                       [[0], [4]],
                                       [[0], [0]],
                                       [[0], [0]],
                                       [[0], [0]],
                                       [[0], [0]]],

                                      [[[0], [0]],
                                       [[0], [4]],
                                       [[0], [0]],
                                       [[0], [0]],
                                       [[0], [0]],
                                       [[0], [0]]],

                                      [[[0], [0]],
                                       [[0], [0]],
                                       [[0], [6]],
                                       [[0], [0]],
                                       [[0], [0]],
                                       [[0], [0]]],

                                      [[[0], [0]],
                                       [[0], [0]],
                                       [[0], [6]],
                                       [[0], [0]],
                                       [[0], [0]],
                                       [[0], [0]]],

                                      [[[0], [0]],
                                       [[0], [0]],
                                       [[0], [0]],
                                       [[0], [0]],
                                       [[0], [0]],
                                       [[10], [0]]],

                                      [[[0], [0]],
                                       [[0], [0]],
                                       [[0], [0]],
                                       [[0], [0]],
                                       [[0], [0]],
                                       [[0], [0]]]]])

                # Check if the relevance scores are correct (the correct values
                # are found by calculating the example by hand)
                self.assertTrue(
                    np.allclose(expected,
                                explanation,
                                rtol=1e-03,
                                atol=1e-03),
                    msg="Should be a good explanation")
