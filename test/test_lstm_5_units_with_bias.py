from lrp import lrp
from test.test_constants import LSTM_WEIGHTS, LSTM_BIAS
import tensorflow as tf
import numpy as np
import unittest


class LSTM5UnitsWithBiasLRPTest(unittest.TestCase):
    def runTest(self):
        with tf.Graph().as_default() as g:
            lstm_units = 5

            # Make static input
            np_input = np.reshape(np.arange(-10, 14), (1, 8, 3))
            inp = tf.constant(np_input, dtype=tf.float32)

            # Create lstm layer
            lstm = tf.contrib.rnn.LSTMCell(lstm_units,
                                                   # initializer=tf.constant_initializer(1., dtype=tf.float32),
                                                   forget_bias=0.)

            # Put it into Multi RNN Cell
            lstm = tf.contrib.rnn.MultiRNNCell([lstm])
            # Let dynamic rnn setup the control flow (making while loops and stuff)
            lstm_output, _ = tf.nn.dynamic_rnn(lstm, inp, dtype=tf.float32)

            # Construct operation for assigning mock weights
            kernel = next(i for i in tf.global_variables() if i.shape == (8, 20))
            assign_kernel = kernel.assign(LSTM_WEIGHTS)

            # Construct operation for assigning mock bias
            bias = next(i for i in tf.global_variables() if i.shape == (20,))
            assign_bias = bias.assign(LSTM_BIAS)

            # Fake the relevance
            R = tf.ones_like(tf.slice(lstm_output, [0, -1, 0], [1, 1, lstm_units]))

            # Get the explanation from the LRP framework.
            # TODO The output of the lstm is not quite right.
            R = lrp._lrp(inp, lstm_output, R)

            with tf.Session() as s:
                # Initialize variables
                s.run(tf.global_variables_initializer())

                # Assign mock kernel and bias
                s.run([assign_kernel, assign_bias])

                # Calculate relevance
                relevances = s.run(R)

                # Expected result calculated in
                # https://docs.google.com/spreadsheets/d/1_bmSEBSWVOkpdlZYEUckgrnUtxhEfnR84LZy1cU5fIw/edit?usp=sharing
                expected_result = np.array([[[-3.863966, -4.055491, 0.059204],
                                             [-3.547705, -4.202078, -1.405731],
                                             [-2.133164, -1.981917, -0.768870],
                                             [-0.377051, -0.018967, -0.201945],
                                             [0.780807, -0.580487, -1.465316],
                                             [2.728909, -2.918414, -2.193734],
                                             [24.634636, -78.773084, 33.587597],
                                             [-73.446891, 445.763106, -320.116405]]])

                # Check for shape and actual result
                self.assertEqual(expected_result.shape, relevances.shape,
                                 "Shapes of expected relevance and relevance should be equal")
                self.assertTrue(np.allclose(relevances, expected_result, rtol=1e-03, atol=1e-03),
                                "The relevances do not match")
