from lrp import lrp
from test.test_constants import LSTM_WEIGHTS
import tensorflow as tf
import numpy as np
import unittest


class LSTM5UnitsLRPTest(unittest.TestCase):
    def runTest(self):
        with tf.Graph().as_default():
            lstm_units = 5

            # Make static input
            np_input = np.reshape(np.arange(-10, 14), (1, 8, 3))
            inp = tf.constant(np_input, dtype=tf.float32)

            # Create lstm layer
            lstm = tf.contrib.rnn.LSTMCell(lstm_units, forget_bias=0.)
            # Put it into Multi RNN Cell
            lstm = tf.contrib.rnn.MultiRNNCell([lstm])
            # Let dynamic rnn setup the control flow (making while loops and stuff)
            lstm_output, _ = tf.nn.dynamic_rnn(lstm, inp, dtype=tf.float32)

            # Construct operation for assigning mock weights
            kernel = next(i for i in tf.global_variables() if i.shape == (8, 20))
            assign_kernel = kernel.assign(LSTM_WEIGHTS)

            # Slice the output to get shape (batch_size, 1, lstm_units) and then squueze the 2nd dimension to
            # get shape (batch_size, lstm_units) so we test if the framework if capable of handling starting point
            # predictions without the predictions_per_sample dimension
            output = tf.squeeze(tf.slice(lstm_output, [0, 7, 0], [1, 1, lstm_units]), 1)

            # Get the explanation from the LRP framework.
            # TODO The output of the lstm is not quite right.
            explanation = lrp.lrp(inp, output)

            with tf.Session() as s:
                # Initialize variables
                s.run(tf.global_variables_initializer())

                # Assign mock kernel
                s.run(assign_kernel)

                # Calculate relevance
                relevances = s.run(explanation)

                # Expected result calculated in
                # https://docs.google.com/spreadsheets/d/1_bmSEBSWVOkpdlZYEUckgrnUtxhEfnR84LZy1cU5fIw/edit?usp=sharing
                expected_result = np.array([[[-0.001298088358, 0.01145384055, 0.0006777067672],
                                            [0.02191306626, 0.008657823488, -0.01480988454],
                                            [0.05461008726, 0.0005180473113, -0.02527708783],
                                            [0.02971554238, 0.0, 0.02727937854],
                                            [-0.08685950242, 0.02556374875, 0.1572219068],
                                            [-0.2561188815, 0.06659819392, 0.3236008156],
                                            [-0.4212428569, 0.1028748125, 0.4771196328],
                                            [-0.569597743, 0.1599460364, 0.5879939284]]]
                                           )

                # Check for shape and actual result
                self.assertEqual(expected_result.shape, relevances.shape,
                                 "Shapes of expected relevance and actual relevance should be the same")
                self.assertTrue(np.allclose(relevances, expected_result, rtol=1e-03, atol=1e-03),
                                "The relevances do not match")
