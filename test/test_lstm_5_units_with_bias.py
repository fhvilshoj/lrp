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
            output = tf.squeeze(tf.slice(lstm_output, [0, 7, 0], [1, 1, lstm_units]), 1)

            # Get the explanation from the LRP framework.
            # TODO The output of the lstm is not quite right.
            R = lrp.lrp(inp, output)

            with tf.Session() as s:
                # Initialize variables
                s.run(tf.global_variables_initializer())

                # Assign mock kernel and bias
                s.run([assign_kernel, assign_bias])

                # Calculate relevance
                relevances = s.run(R)

                # Expected result calculated in
                # https://docs.google.com/spreadsheets/d/1_bmSEBSWVOkpdlZYEUckgrnUtxhEfnR84LZy1cU5fIw/edit?usp=sharing
                expected_result = np.array([[[0.163075, 0.036412, -0.107258],
                                             [0.158804, 0.021996, -0.090554],
                                             [0.102691, 0.013007, -0.037685],
                                             [0.030479, 0.005401, 0.029605],
                                             [-0.057824, 0.011063, 0.130999],
                                             [-0.200446, 0.02317, 0.312361],
                                             [-0.327078, -0.200649, 0.655012],
                                             [-0.624182, 0.185695, 0.660877]]])

                # Check for shape and actual result
                self.assertEqual(expected_result.shape, relevances.shape,
                                 "Shapes of expected relevance and relevance should be equal")
                self.assertTrue(np.allclose(relevances, expected_result, rtol=1e-03, atol=1e-03),
                                "The relevances do not match")
