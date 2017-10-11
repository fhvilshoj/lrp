import tensorflow as tf
import numpy as np
from lrp import lrp
import unittest


# Class for testing that inherits from the unittest.TestCase class
class LinearConvolutionLinearRPTest(unittest.TestCase):
    # Test case that builds a FC-Conv-FC network,
    # finds the relevance and compares them to the results
    # obtained by calculating the same example by hand
    def runTest(self):
        # Get a tensorflow graph
        g = tf.Graph()
        # Set the graph as default
        with g.as_default():
            # Create a placeholder for the input
            inp = tf.placeholder(tf.float32, shape=(1, 4))

            # --------------------- Linear layer -----------------------------
            # Setup the linear layer
            weights_1 = tf.constant([[10, 1, 0],
                                     [-10, 1, 1],
                                     [5, 1, 0],
                                     [17, 1, -1]], dtype=tf.float32)
            bias_1 = tf.constant([-20, 13, 2], dtype=tf.float32)

            # Calculate the activations
            activation_1 = tf.matmul(inp, weights_1, name="MUL1") + bias_1

            # --------------------- Conv layer -----------------------------
            # Create the convolutional layer
            # Set the filters and biases
            filters = tf.constant([[[-3, 3]], [[3, 10]]], dtype=tf.float32)
            bias_2 = tf.constant([-9, 19], dtype=tf.float32)

            # Expand the dimensions of the output from the first layer,
            # since the 1dconv takes rank 3 tensors as input
            activation_1 = tf.expand_dims(activation_1, -1)

            # Perform the convolution and add bias
            activation_2 = tf.nn.conv1d(activation_1, filters, 1, "SAME") + bias_2

            # --------------------- Linear layer -----------------------------

            # Flatten the output from the conv layer so it can serve as input
            # to a linear layer
            activation_2_flattened = tf.reshape(activation_2, (1, 6))

            # Setup the linear layer
            weights_3 = tf.constant([[14],
                                     [14],
                                     [1],
                                     [3],
                                     [5],
                                     [-1]], dtype=tf.float32)
            bias_3 = tf.constant([0], dtype=tf.float32)

            # Calculate the activations
            activation_3 = tf.matmul(activation_2_flattened, weights_3, name="MUL3") + bias_3

            # Set the prediction to be equal to the activations of the last layer
            # (there is no softmax in this network)
            pred = activation_3

            # --------------------- Calculate relevances -----------------------------
            # Calculate the relevance scores using lrp

            expl = lrp.lrp(inp, pred)

            # Run a tensorflow session to evaluate the graph
            with tf.Session() as sess:
                # Initialize the variables
                sess.run(tf.global_variables_initializer())

                # Run the operations of interest and feed an input to the network
                prediction, explanation = sess.run([pred, expl], feed_dict={inp: [[-1, 3, 55, 0]]})


                # Check if the predictions has the right shape
                self.assertEqual(prediction.shape, (1, 1),
                                 msg="Should be able to do a forward pass")

                # Check if the explanation has the right shape
                self.assertEqual(explanation.shape, inp.shape, msg="Should be a wellformed explanation")

                # Check if the relevance scores are correct (the correct values are found by
                # calculating the example by hand)
                self.assertTrue(
                    np.allclose(explanation[0], [[0, 353.58, 11466.76, 0]], rtol=1e-01, atol=1e-01),
                    msg="Should be a good explanation")
