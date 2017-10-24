import tensorflow as tf
import numpy as np
import unittest
from lrp import lrp


class TESTLSTMWithDropoutWrapper(unittest.TestCase):
    def runTest(self):
        with tf.Graph().as_default():
            lstm_units = 2
            max_time_step = 3
            input_depth = 4
            TYPES_OF_GATES_IN_A_LSTM = 4

            # Create input
            inp = tf.constant(
                [[[-5, -4, -3, -2],
                  [-1, 0, 1, 2],
                  [3, 4, 5, 6]],
                 [[7, 8, 9, 10],
                  [11, 12, 13, 14],
                  [15, 16, 17, 18]]]
                , dtype=tf.float32)

            # Create mock weights
            LSTM_WEIGHTS = tf.constant([[0.19517923, 0.24883463, 0.65906681, 0.43171532, 0.30894309,
                                         0.18143875, 0.9064917, 0.34376469],
                                        [0.36688612, 0.41893102, 0.68622539, 0.92279857, 0.18437027,
                                         0.7582207, 0.70674838, 0.8861974],
                                        [0.60149935, 0.84269909, 0.3129998, 0.75019745, 0.75946505,
                                         0.76014145, 0.012957, 0.06685569],
                                        [0.09053277, 0.91693017, 0.11203575, 0.85798137, 0.14988363,
                                         0.96619787, 0.63018615, 0.77663712],
                                        [0.18449367, 0.61801985, 0.07125719, 0.02529254, 0.42940272,
                                         0.96136843, 0.95259111, 0.33910939],
                                        [0.88669326, 0.58888385, 0.11549774, 0.63704878, 0.85553019,
                                         0.39069136, 0.56481662, 0.27301619]]
                                       , dtype=tf.float32)

            # Create bias weights
            LSTM_BIAS = tf.constant(
                [0.37282269, 0.16629956, 0.92285417, 0.86485604, -0.13370907, 0.75214074, 0.72669859, -0.261183],
                dtype=tf.float32)

            # Create lstm layer
            lstm = tf.contrib.rnn.LSTMCell(lstm_units, forget_bias=0.)

            # Place the lstm cell in a dropout wrapper
            lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=1)

            # Put it into Multi RNN Cell
            lstm = tf.contrib.rnn.MultiRNNCell([lstm])

            # Let dynamic rnn setup the control flow (making while loops and stuff)
            lstm_output, _ = tf.nn.dynamic_rnn(lstm, inp, dtype=tf.float32)

            # Construct operation for assigning mock weights
            kernel = next(i for i in tf.global_variables() if
                          i.shape == (input_depth + lstm_units, lstm_units * TYPES_OF_GATES_IN_A_LSTM))
            assign_kernel = kernel.assign(LSTM_WEIGHTS)

            # Construct operation for assigning mock bias
            bias = next(i for i in tf.global_variables() if i.shape == (lstm_units * TYPES_OF_GATES_IN_A_LSTM,))
            assign_bias = bias.assign(LSTM_BIAS)

            # Get the explanation from the LRP framework
            R = lrp.lrp(inp, lstm_output)
            R_sum = tf.reduce_sum(R, [2, 3])

            with tf.Session() as s:
                # Initialize variables
                s.run(tf.global_variables_initializer())

                # Assign mock weights and mock bias
                s.run([assign_kernel, assign_bias])

                # Calculate the relevances
                explanation = s.run(R)

                # Create array with expected relevances
                expected_explanation = np.array(
                    [[[[-0.00000000329187, -0.00000000579639, -0.00000000344224, -0.00000000256858],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]],
                      [[0.01917747822, 0.02132167498, 0.009472542897, 0.004885821672],
                       [-0.05303456479, 0.02658300098, 0.1649355033, 0.3430433395],
                       [0, 0, 0, 0]],
                      [[0.01775032921, 0.01975440857, 0.008784149807, 0.004537977098],
                       [-0.05051601016, 0.02468158083, 0.1500054205, 0.3081801763],
                       [0.04747183272, 0.1265003211, 0.1284724633, 0.1745462303]]],
                     [[[0.09064830212495, 0.21548798884060, 0.19743329647079, 0.24977081145921],
                       [0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000],
                       [0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000]],
                      [[0.05894237044, 0.1393483447, 0.127375767, 0.1608060994],
                       [0.06054255911, 0.1387999706, 0.1224546803, 0.1504080344],
                       [0, 0, 0, 0]],
                      [[0.04103863682, 0.09681307209, 0.08841408434, 0.1115271072],
                       [0.04215389263, 0.09642979223, 0.0849954929, 0.1043121762],
                       [0.04310312299, 0.09707452916, 0.08397782613, 0.1014948822]]]]
                )

                self.assertTrue(np.allclose(expected_explanation, explanation, rtol=1e-3, atol=1e-3),
                                "The eplanation should match the expected explanation")



