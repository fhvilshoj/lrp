from lrp import lrp
import tensorflow as tf
import numpy as np
import unittest

class LSTM5Units1StackedLRPTest(unittest.TestCase):
    def runTest(self):
        with tf.Graph().as_default() as g:
            lstm_units = 5
            np_input = np.reshape(np.arange(-10, 14), (1, 8, 3))
            inp = tf.constant(np_input, dtype=tf.float32)

            tf.set_random_seed(1337)
            with tf.variable_scope('the_lstm'):
                stacked_lstm = tf.contrib.rnn.LSTMCell(lstm_units,
                                                       forget_bias=0.)  # initializer=tf.constant_initializer(1.)
                stacked_lstm = tf.contrib.rnn.MultiRNNCell([stacked_lstm] * 1)
                lstm_output, _ = tf.nn.dynamic_rnn(stacked_lstm, inp, dtype=tf.float32)
            R = tf.ones_like(tf.slice(lstm_output, [0, -1, 0], [1, 1, lstm_units]))

            # TODO The output of the lstm is not quite right.
            R = lrp._lrp(inp, lstm_output, R)

            with tf.Session() as s:
                s.run(tf.global_variables_initializer())
                relevances = s.run(R)

                expected_result = np.array([[[0.04114789, 0.07254411, 0.13169597],
                                             [0.17604621, 0.1747233, 0.19547687],
                                             [0.37701777, 0.42817616, 0.29883289],
                                             [0.18942904, 0., -0.38005733],
                                             [-0.13430591, -0.19828944, -0.08908054],
                                             [-0.30317006, 0.22794639, 0.16859262],
                                             [-0.12844698, -1.49989772, 1.67834044],
                                             [-3.57829499, 13.87882996, -6.72725677]]])

                self.assertTrue(relevances.shape == expected_result.shape, "Shapes not equal")
                self.assertTrue(np.allclose(relevances, expected_result, rtol=1e-03, atol=1e-03), "NO LONGER EQUAL RELEVANCES")
