import tensorflow as tf
import numpy as np
import unittest

from configuration import LRPConfiguration, LAYER, BIAS_STRATEGY, EpsilonConfiguration, AlphaBetaConfiguration, \
    WWConfiguration, FlatConfiguration
from lrp import lrp


class TestLSTMConfigs(unittest.TestCase):
    def test_lstm_epsilon_no_bias(self):
        config = LRPConfiguration()
        config.set(LAYER.LSTM, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.NONE))

        expected_result = np.array([[[[0.19220784, -0.04284667, 0.32415234],
                                      [0., 0., 0.]],
                                     [[0.23325387, -0.13045811, 1.0724654],
                                      [0.16538274, 0.24215863, 0.04471665]]]])
        self._do_test_with_config_and_expected_result(config, expected_result)

    def test_lstm_epsilon_all_bias(self):
        config = LRPConfiguration()
        config.set(LAYER.LSTM, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.ALL))

        expected_result = np.array([[[[0.23963733, 0.00458281, 0.37158183],
                                      [0., 0., 0.]],
                                     [[0.10851287, -0.22616417, 0.89225635],
                                      [0.13618791, 0.2129638, 0.01552182]]]])
        self._do_test_with_config_and_expected_result(config, expected_result)

    def test_lstm_epsilon_active_bias(self):
        config = LRPConfiguration()
        config.set(LAYER.LSTM, EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.ACTIVE))

        expected_result = np.array([[[[0.27125698, 0.03620247, 0.40320148],
                                      [0., 0., 0.]],
                                     [[0.03854221, -0.29613484, 0.82228568],
                                      [0.13618791, 0.2129638, 0.01552182]]]])
        self._do_test_with_config_and_expected_result(config, expected_result)

    def test_lstm_alpha_beta_no_bias(self):
        config = LRPConfiguration()
        config.set(LAYER.LSTM, AlphaBetaConfiguration(alpha=2, beta=-1, bias_strategy=BIAS_STRATEGY.NONE))

        expected_result = np.array([[[[0.72511334, -1.42132187, 1.22288032],
                                      [0., 0., 0.]],
                                     [[1.17308824, -1.30891597, 4.40328549],
                                      [0.52473047, 0.76832693, 0.14187811]]]])
        self._do_test_with_config_and_expected_result(config, expected_result)

    def test_lstm_alpha_beta_all_bias(self):
        config = LRPConfiguration()
        config.set(LAYER.LSTM, AlphaBetaConfiguration(alpha=2, beta=-1, bias_strategy=BIAS_STRATEGY.ALL))

        expected_result = np.array([[[[0.90404336, -1.24239185, 1.40181034],
                                      [0., 0., 0.]],
                                     [[0.58241955, -1.00368592, 3.04841668],
                                      [0.30076316, 0.54435962, -0.08208919]]]])
        self._do_test_with_config_and_expected_result(config, expected_result)

    def test_lstm_alpha_beta_active_bias(self):
        config = LRPConfiguration()
        config.set(LAYER.LSTM, AlphaBetaConfiguration(alpha=2, beta=-1, bias_strategy=BIAS_STRATEGY.ACTIVE))

        expected_result = np.array([[[[1.17243838, -1.42132187, 1.67020536],
                                      [0., 0., 0.]],
                                     [[1.47216972, -3.08726835, 4.70236698],
                                      [0.52473047, 0.76832693, 0.14187811]]]])
        self._do_test_with_config_and_expected_result(config, expected_result)

    def test_ww(self):
        config = LRPConfiguration()
        config.set(LAYER.LSTM, WWConfiguration())

        expected_result = np.array([[[[0.26074915, 0.0391348, 0.01872167],
                                      [0., 0., 0.]],
                                     [[0.09823269, 0.04826224, 0.02665878],
                                      [0.04887697, 0.05858786, 0.03348749]]]])
        self._do_test_with_config_and_expected_result(config, expected_result)

    def test_flat(self):
        config = LRPConfiguration()
        config.set(LAYER.LSTM, FlatConfiguration())

        expected_result = np.array([[[[0.07106609, 0.07106609, 0.07106609],
                                      [0., 0., 0.]],
                                     [[0.06505566, 0.06505566, 0.06505566],
                                      [0.09293666, 0.09293666, 0.09293666]]]])
        self._do_test_with_config_and_expected_result(config, expected_result)

    def _do_test_with_config_and_expected_result(self, config, expected_result):
        with tf.Graph().as_default():
            lstm_units = 2

            # Make static input shape: (1, 2, 3)
            input = [[[0.42624769, -0.24526631, 2.6827445],
                      [1.50864387, 2.01764531, 0.49280675]]]

            inp = tf.constant(input, dtype=tf.float32)

            # Create lstm layer
            lstm = tf.contrib.rnn.LSTMCell(lstm_units,
                                           # initializer=tf.constant_initializer(1., dtype=tf.float32),
                                           forget_bias=0.)

            # Put it into Multi RNN Cell
            lstm = tf.contrib.rnn.MultiRNNCell([lstm])
            # Let dynamic rnn setup the control flow (making while loops and stuff)
            lstm_output, _ = tf.nn.dynamic_rnn(lstm, inp, dtype=tf.float32)

            # Construct operation for assigning mock weights
            kernel = next(i for i in tf.global_variables() if i.shape == (5, 8))
            assign_kernel = kernel.assign(
                [[1.48840206, -0.948746, 0.77125305, 2.55016428, 1.18691298, -0.24476569, 2.56425766, 0.13880421],
                 [2.08840452, -0.46632275, 0.84440069, 0.98795753, 0.61527844, 2.7123294, 0.33261274, 1.86915179],
                 [1.42373006, 1.23778513, 0.63839003, 0.68332758, 0.82368828, -0.11620465, 0.11787995, 1.58372134],
                 [2.35450518, -0.41308389, 1.31977204, 0.91312955, -0.13488139, 0.93544023, 0.0894083, 0.12383227],
                 [0.0330369, 2.63521215, 1.48256475, -0.28661456, 1.70166103, 1.80855782, 1.35295711, 0.58774797]])

            # Construct operation for assigning mock bias
            bias = next(i for i in tf.global_variables() if i.shape == (8,))
            assign_bias = bias.assign([1.39120539, -0.91791735, -1.02699767, 1.34115046, 0.19859183,
                                       0.73738726, 1.6626231, 2.31063315])

            output = lstm_output

            # Get the explanation from the LRP framework.
            R = lrp.lrp(inp, output, config)

            with tf.Session() as s:
                # Initialize variables
                s.run(tf.global_variables_initializer())

                # Assign mock kernel and bias
                s.run([assign_kernel, assign_bias])

                # Calculate relevance
                relevances = s.run(R)

                # Check for shape and actual result
                self.assertEqual(expected_result.shape, relevances.shape,
                                 "Shapes of expected relevance and relevance should be equal")
                self.assertTrue(np.allclose(relevances, expected_result, rtol=1e-03, atol=1e-03),
                                "The relevances do not match")
