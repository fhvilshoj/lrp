from lrp import lrp
import tensorflow as tf
import numpy as np
import unittest


class FCConvMaxLSTMFCTest(unittest.TestCase):
    def runTest(self):
        # Build the computational graph
        with tf.Graph().as_default():
            # Make static input shape: (4, 6)
            inp = tf.constant([[0.49529851, -0.64648792, 0.18508197, -0.14201359, 0.29480708,
                                -0.23168202],
                               [0.03458613, -0.11823616, -0.67511888, -0.17805438, 0.7495242,
                                -0.29286811],
                               [-1.51681199, -1.05214575, -0.31338711, -0.14845488, 0.32269641,
                                2.08227179],
                               [0.18766482, 0.10273498, 0.93645419, -0.71804516, -0.92730127,
                                0.11013126]]

                              , dtype=tf.float32)

            # -------------------------------------------- FC 1--------------------------------------------
            # shape: (6, 10)
            weights_1 = tf.constant([[0.63023564, 0.72418477, -0.03827982, 0.48678625, 1.04275436,
                                      0.35443291, 0.92035296, -0.89705308, -0.90312034, 0.51454559],
                                     [-0.40643197, -0.19538514, 0.52029634, -0.43955133, -1.19634436,
                                      0.33259718, -0.21127183, 0.6793771, -0.72406187, 1.54054022],
                                     [-0.67484287, -1.14928279, 1.28560718, -0.02242465, 0.3377433,
                                      0.74823952, 2.18620002, 0.18857024, 0.6531554, 2.72489568],
                                     [-0.15728904, 0.28771674, -1.18420233, 2.17638949, -0.47370135,
                                      -0.02005775, 0.41663315, 0.60860928, 0.57529257, -0.104214],
                                     [-0.4055075, 0.8596076, -0.89655813, -1.39272219, 0.73927064,
                                      1.44179635, -1.01808758, 1.20547704, -1.30409662, 0.02295059],
                                     [0.80911711, 0.99273549, 0.31395664, 2.29630583, 0.58090097,
                                      -1.05635963, -0.90120138, 1.63487712, 2.27660196, 0.51776111]]

                                    ,
                                    dtype=tf.float32)

            # shape: (10,)
            bias_1 = tf.constant([-1.58876296, 1.44444094, 0.73600305, 0.99948251, -0.25653983,
                                  0.54555058, 0.80193129, -0.46552793, 0.30203156, -0.28628351]
                                 , dtype=tf.float32)

            # shape: (4, 10)
            output_1 = tf.matmul(inp, weights_1) + bias_1

            # Prepare the output for the convolutional layer
            # New shape: (4, 5, 2)
            output_1 = tf.reshape(output_1, (4, 5, 2))

            # -------------------------------------------- Convolution 1 --------------------------------------------
            # Create the filter which has shape [filter_width, in_channels, out_channels]
            # Shape: (2,2,1)
            filter_2 = tf.constant(
                [[[-0.41445586],
                  [1.26795033]],
                 [[-1.61688659],
                  [1.50628238]]]
                , dtype=tf.float32)

            # Shape: (1,)
            bias_2 = tf.constant([0.84705889], dtype=tf.float32)

            # Perform the convolution
            # Shape of output_2: (4, 5, 1)
            output_2 = tf.nn.conv1d(output_1, filter_2, 1, "SAME") + bias_2

            # Prepare output for the max pooling
            # Pooling is defined for 2d, so add dim of 1 (height)
            # New shape of output_2_reshaped: (4,1,5,1)
            output_2 = tf.expand_dims(output_2, 1)

            # -------------------------------------------- Max pooling --------------------------------------------

            # Kernel looks at 1 sample, 1 height, 2 width, and 1 depth
            ksize = [1, 1, 2, 1]

            # Move 1 sample, 1 height, 2 width, and 1 depth at a time
            strides = [1, 1, 2, 1]

            # Perform the max pooling
            # Shape of pool: (4, 1, 3, 1)
            pool = tf.nn.max_pool(output_2, ksize, strides, padding='SAME')

            # Remove the "height" dimension again
            # New shape: (4, 3, 1)
            output_3 = tf.squeeze(pool, 1)

            # -------------------------------------------- Convolution 2--------------------------------------------

            # Create the filter which has shape [filter_width, in_channels, out_channels]
            # Shape: (2,1,2)
            filter_4 = tf.constant(
                [[[0.19205464, -0.90562985]],
                 [[1.0016198, 0.89356491]]],
                dtype=tf.float32)

            # Shape: (2, )
            bias_4 = tf.constant([0.07450981, -0.14901961], dtype=tf.float32)

            # Perform the convolution
            # Shape: (4, 3, 2)
            output_4 = tf.nn.conv1d(output_3, filter_4, 1, "SAME") + bias_4

            # Prepare output for the max pooling
            # Pooling is defined for 2d, so add dim of 1 (height)
            # New shape of output_2_reshaped: (4,1,3,2)
            output_4 = tf.expand_dims(output_4, 1)

            # -------------------------------------------- Max pooling --------------------------------------------

            # Kernel looks at 1 sample, 1 height, 2 width, and 1 depth
            ksize = [1, 1, 2, 1]

            # Move 1 sample, 1 height, 2 width, and 1 depth at a time
            strides = [1, 1, 2, 1]

            # Perform the max pooling
            # Shape of pool: (4, 1, 2, 2)
            pool = tf.nn.max_pool(output_4, ksize, strides, padding='SAME')

            # Remove the "height" dimension again
            # New shape: (4, 2, 2)
            output_5 = tf.squeeze(pool, 1)

            # -------------------------------------------- LSTM --------------------------------------------
            lstm_units = 3

            # Shape of weights: (5, 12)
            LSTM_weights = [[-0.6774261, -1.77336803, 0.37789944, -1.47642675, -0.77326061,
                             -0.41624885, -0.80737161, -1.00830384, 0.80501084, -0.10506079,
                             -0.42341706, 1.61561637],
                            [0.65131449, -1.25813521, -1.01188983, 1.58355103, -0.55863594,
                             0.59259386, -1.15333092, 1.31657508, -0.3582473, -0.91620798,
                             1.30231276, 0.32319264],
                            [1.11120673, 0.60646556, -1.11294626, -0.26202266, -1.53741017,
                             -0.09405062, -0.82200596, -0.41727707, 0.69017403, -2.67866443,
                             1.08780302, -0.53820959],
                            [0.12222124, 0.17716194, -0.96654223, 0.64953949, 1.55478632,
                             -1.22787184, 0.67456202, 0.34234439, -2.42116309, 0.22752669,
                             -0.40613203, -0.42356035],
                            [0.9004432, 1.9286521, 1.04199918, 1.17486178, -1.30394625,
                             0.60571671, 1.30499515, 2.12358405, -1.82775648, 0.81019163,
                             0.20284197, 0.72304922]]
            # Shape: (12,)
            LSTM_bias = [-9.46940060e-01, 5.69888879e-02, -4.06483928e-05,
                         -9.60644436e-01, -1.18161660e+00, -2.04222054e+00,
                         -2.27343882e-02, 2.39842965e-01, -4.42784509e-01,
                         2.86647829e+00, 2.92904572e-02, -1.45679881e+00]

            # Create lstm layer
            lstm = tf.contrib.rnn.LSTMCell(lstm_units,
                                           forget_bias=0.)

            # Put it into Multi RNN Cell
            lstm = tf.contrib.rnn.MultiRNNCell([lstm])

            # Let dynamic rnn setup the control flow (making while loops and stuff)
            # Shape of output_6: (4, 2, 3)
            output_6, _ = tf.nn.dynamic_rnn(lstm, output_5, dtype=tf.float32)

            # Construct operation for assigning mock weights
            kernel = next(i for i in tf.global_variables() if i.shape == (5, 12))
            assign_kernel = kernel.assign(LSTM_weights)

            # Construct operation for assigning mock bias
            bias = next(i for i in tf.global_variables() if i.shape == (12,))
            assign_bias = bias.assign(LSTM_bias)

            # Prepare output for the linear layer by flattening the batch and timesteps
            # to be able to use linear layer on all predictions and all samples in batch at the same time
            # New shape of output_6: (8, 3)
            output_6 = tf.reshape(output_6, (-1, lstm_units))

            # -------------------------------------------- FC 2--------------------------------------------
            # Shape of weights_7: (3,2)
            weights_7 = tf.constant([[1.6295322, 1.54609607],
                                     [1.04818304, -0.98323105],
                                     [-1.35106161, -1.20737747]],
                                    dtype=tf.float32)

            bias_7 = tf.constant([0.85856544, 0.75856544], dtype=tf.float32)

            # Perform the matmul
            output_7 = tf.matmul(output_6, weights_7) + bias_7

            # Reshape to shape (4, 2, 2)
            output_7 = tf.reshape(output_7, (4, 2, 2))

            # -------------------------------------------- Softmax -------------------------------------------

            output_final = tf.nn.softmax(output_7)

            # -------------------------------------------- LRP -------------------------------------------

            # Get the explanation from the LRP framework.
            R = lrp.lrp(inp, output_final)

            # Run the computations
            with tf.Session() as s:
                # Initialize variables
                s.run(tf.global_variables_initializer())

                # Assign mock bias
                s.run([assign_kernel, assign_bias])

                output = s.run(output_final)
                print("out shape", output.shape)

                # # Calculate relevance
                relevances = s.run(R)

                print(relevances.shape)

                # Expected result calculated in
                # https://docs.google.com/spreadsheets/d/1_bmSEBSWVOkpdlZYEUckgrnUtxhEfnR84LZy1cU5fIw/edit?usp=sharing
                expected_result = np.array([[0, 0.00009744890346, 0.00005380548547, 0.00006157330073, 0.0002224629534,
                                             0.000001148301027, 0, 0, 0.00009112270278, 0.0005228184956],
                                            [0.0003926704111, 0.000690260643, 0.00002775464123, 0.00002765821888,
                                             0.0001484110742, 0.0001955332403, 0.000003241945558, 0.0008382231393, 0,
                                             0],
                                            [0.0001084081589, 0.0003690088929, 0.0001606364539, 0, 0.0001548509328,
                                             0.00001354378626, 0, 0, 0.00002533766043, 0.000005178358804],
                                            [0.0002870030767, 0.0003006917612, 0.0005409557879, 0.001673765604,
                                             0.0005882002186, 0.0009939816563, 0.001152595479, 0.002409395133,
                                             0.00007871882761, 0.0001197214162],
                                            [0.003720715575, 0, 0.0008488224481, 0.001008967078, 0.007691724936,
                                             0.007605366069, 0.001017835398, 0.00411259372, 0.0005971349689,
                                             0.000256612695],
                                            [0, 0.0004449329458, 0, 0.01210797823, 0.00004847106631, 0.003506019562,
                                             0.001415993078, 0, 0.002416949139, 0.00006304742861],
                                            [0.005359096108, 0.005972816778, 0.002764248481, 0.007948120325,
                                             0.0003087310192, 0.01114928751, 0.0001196646517, 0.003871852483,
                                             0.0004964681915, 0.0003417654898],
                                            [0.02020125409, 0.0009174208261, 0.000388005489, 0.01936216584, 0,
                                             0.0243733322, 0.002747249697, 0.05362725729, 0.01680197517,
                                             0.02779999447]])

                self.assertEqual(expected_result.shape, relevances.shape,
                                 "Shapes of expected relevance and relevance should be equal")
                self.assertTrue(np.allclose(relevances, expected_result, rtol=1e-03, atol=1e-03),
                                "The relevances do not match")
