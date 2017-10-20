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
            inp = tf.placeholder(tf.float32, shape=(1, 1, 4, 6))

            # Create max pooling layer
            activation = tf.nn.max_pool(inp, [1, 1, 2, 1], [1, 1, 2, 1], "SAME")

            # Set the prediction to be equal to the activations of the last layer
            pred = activation

            # Calculate the relevance scores using lrp
            # The tf.expand_dims() is necessary because we call _lrp which means that
            # we bypass the part of the framework that takes care of adding and removing
            # an extra dimension for multiple predictions per sample
            expl = lrp._lrp(inp, pred, tf.expand_dims(pred, 1))

            # Run a tensorflow session to evaluate the graph
            with tf.Session() as sess:
                # Initialize the variables
                sess.run(tf.global_variables_initializer())

                # Run the operations of interest and feed an input to the network
                prediction, explanation = sess.run([pred, expl],
                                                   feed_dict={inp: [[[[1, 2, 3, 4, 5, 6],
                                                                      [3, 4, 3, 5, 6, 7],
                                                                      [5, 6, -1, -5, 0, -1],
                                                                      [-1, -2, 1, 1, 1, 1]]]]})

                # Check if the predictions has the right shape
                self.assertEqual(prediction.shape, (1, 1, 2, 6),
                                 msg="Should be able to do a linear forward pass")

                # Check if the explanation has the right shape
                self.assertEqual(list(explanation[0].shape), inp.get_shape().as_list(),
                                 msg="Should be a wellformed explanation")

                # Check if the relevance scores are correct (the correct values
                # are found by calculating the example by hand)
                self.assertTrue(
                    np.allclose([[[[[0, 0, 3, 0, 0, 0],
                                    [3, 4, 3, 5, 6, 7],
                                    [5, 6, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 1, 1]]]]],
                                explanation,
                                rtol=1e-03,
                                atol=1e-03),
                    msg="Should be a good explanation")
