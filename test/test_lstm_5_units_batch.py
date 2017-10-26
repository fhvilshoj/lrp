from lrp import lrp
from test.test_constants import LSTM_WEIGHTS
import tensorflow as tf
import numpy as np
import unittest


class LSTM5UnitsLRPTest(unittest.TestCase):
    def runTest(self):
        with tf.Graph().as_default():
            lstm_units = 5

            # Make static input of shape (2, 8, 3)
            inp = tf.constant([[[-10, -9, -8],
                                [-7, -6, -5],
                                [-4, -3, -2],
                                [-1, 0, 1],
                                [2, 3, 4],
                                [5, 6, 7],
                                [8, 9, 10],
                                [11, 12, 13]],
                               [[-1, 1, 0],
                                [1, 1, 1],
                                [3, 2, 2],
                                [5, 4, 5],
                                [-9, 7, 7],
                                [0, 0, -1],
                                [2, 1, -1],
                                [0, 0, 1]]],
                              dtype=tf.float32)

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
            final_output = tf.slice(lstm_output, [0, 7, 0], [2, 1, lstm_units])

            # Get the explanation from the LRP framework
            # TODO The output of the lstm is not quite right.
            R = lrp.lrp(inp, final_output)

            with tf.Session() as s:
                # Initialize variables
                s.run(tf.global_variables_initializer())

                # Assign mock kernel
                s.run(assign_kernel)

                # Calculate relevance
                relevances = s.run(R)

                # Expected result calculated in
                # https://docs.google.com/spreadsheets/d/1_bmSEBSWVOkpdlZYEUckgrnUtxhEfnR84LZy1cU5fIw/edit?usp=sharing
                expected_result = np.array([[[[-1.29808836e-03, 1.14538406e-02, 6.77706767e-04],
                                              [2.19130663e-02, 8.65782349e-03, -1.48098845e-02],
                                              [5.46100873e-02, 5.18047311e-04, -2.52770878e-02],
                                              [2.97155424e-02, 0.00000000e+00, 2.72793785e-02],
                                              [-8.68595024e-02, 2.55637488e-02, 1.57221907e-01],
                                              [-2.56118882e-01, 6.65981939e-02, 3.23600816e-01],
                                              [-4.21242857e-01, 1.02874812e-01, 4.77119633e-01],
                                              [-5.69597743e-01, 1.59946036e-01, 5.87993928e-01]]],
                                            [[[-0.0008051434312, 0.0002057962522, 0],
                                              [0.001197399192, 0.0002610343082, -0.0007029896272],
                                              [0.003650561121, 0.001195686468, -0.001623972171],
                                              [0.01503129936, -0.01258498743, 0.002731678024],
                                              [-0.03269409114, 0.0578316263, 0.05384422134],
                                              [0, 0, -0.03243832997],
                                              [0.02649537586, 0.01091601577, -0.02432502036],
                                              [0, 0, 0.06533259294]]]]

                                           )

                # Check for shape and actual result
                self.assertEqual(expected_result.shape, relevances.shape,
                                 "Shapes of expected relevance and actual relevance should be the same")
                self.assertTrue(np.allclose(relevances, expected_result, rtol=1e-03, atol=1e-03),
                                "The relevances do not match")
