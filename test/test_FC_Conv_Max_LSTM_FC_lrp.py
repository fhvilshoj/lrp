from lrp import lrp
import tensorflow as tf
import numpy as np
import unittest


class FCConvMaxLSTMFCTest(unittest.TestCase):
    def runTest(self):
        # Build the computational graph
        with tf.Graph().as_default() as g:
            # Make static input
            inp = tf.constant([[-2, -17, 17, 4, 8, -1, 0, -16, -16, 17],
                               [11, 18, -2, -10, -7, 5, -11, 18, 0, -10],
                               [2, -7, -17, -15, 3, 11, -5, -11, 18, 4],
                               [4, 5, 9, -19, 10, 8, 8, 13, 1, -7],
                               [9, 0, -6, -7, 19, 20, 4, 8, 2, -3],
                               [-2, -17, 0, 10, -17, 17, 19, -15, 2, -7],
                               [7, 14, 11, 16, -1, 10, -9, 3, -4, -7],
                               [11, 1, 1, 16, 0, 14, 2, 20, -20, 13]], dtype=tf.float32)

            # -------------------------------------------- FC 1--------------------------------------------
            weights_1 = tf.constant([[0.99215692, 0.27058825, 0.34509805, 0.99215692, -0.98823535, 0.322352958],
                                     [0.87843144, -0.17254902, 0.50196081, -0.98823535, 0.98823535, 0.241568646],
                                     [-0.54509807, 0.0627451, 0.64705884, -0.1647058, 0.94117653, 0.18823532],
                                     [0.20000002, -0.77647066, 0.99215692, 0.88627458, 0.77254909, 0.41490199],
                                     [0.98823535, 0.10196079, 0.84705889, 0.9450981, 0.22352943, -0.91372555],
                                     [0.9333334, 0.8705883, 0.10980393, -0.20000002, 0.22352943, 0.42745101],
                                     [0.14901961, 0.98823535, -0.0509804, 0.1137255, 0.56470591, 0.07450981],
                                     [0.77254909, 0.88235301, 0.40784317, 0.98823535, 0.80784321, 0.65882355],
                                     [0.98823535, -0.0627451, 0.99215692, -0.98823535, 0.99215692, 0.91764712],
                                     [0.88235301, 0.84705889, 0.99215692, -0.64705884, 0.98823535, 0.98823535]],
                                    dtype=tf.float32)

            bias_1 = tf.constant([-1.5, -1, -0.5, 0, 0.5, 1], dtype=tf.float32)

            output_1 = tf.matmul(inp, weights_1) + bias_1
            output_1 = tf.Print(output_1, [output_1], message="output_1: ", summarize=1000)


            # -------------------------------------------- Convolution 1 --------------------------------------------
            # Create the filter which has shape [filter_width, in_channels, out_channels]
            filter_2 = tf.constant(
                [[[0.14901961, 0.99215692, -0.01176471, -0.85490203],
                  [0.98823535, 0.32156864, 0.49803925, 0.08235294],
                  [-0.0509804, -0.09803922, -0.99215692, 0.66274512],
                  [0.1137255, 0.99215692, 0.99215692, -0.99215692],
                  [0.56470591, -0.99215692, 0.66274512, 0.99215692],
                  [0.07450981, 0.84705889, -0.30588236, -0.52156866]],
                 [[0.14901961, 0.09019608, 0.99215692, 0.1254902],
                  [0.98823535, -0.19215688, -0.99215692, 0.50980395],
                  [-0.0509804, 0.99215692, 0.65882355, -0.99607849],
                  [0.1137255, -0.99215692, 0.27843139, 0.99215692],
                  [0.56470591, 0.76862752, -0.99215692, -0.99215692],
                  [0.07450981, 0.04705883, 0.99215692, 0.44313729]]],
                dtype=tf.float32)

            bias_2 = tf.constant([0.84705889, 0.09019608, -0.19215688, 0.99215692], dtype=tf.float32)

            # Add an extra dimension to fit the expected input shape of conv1d
            output_1_reshaped = tf.expand_dims(output_1, 0)
            output_2 = tf.nn.conv1d(output_1_reshaped, filter_2, 1, "SAME") + bias_2
            output_2 = tf.Print(output_2, [output_2], message="output_2: ", summarize=1000)


            # -------------------------------------------- Max pooling --------------------------------------------

            # Pooling is defined for 2d, so add dim of 1 (height)
            output2_reshaped = tf.expand_dims(output_2, 1)
            # Kernel looks at 1 sample, 1 height, 2 width, and 1 depth
            ksize = [1, 1, 2, 1]

            # Move 1 sample, 1 height, 2 width, and 1 depth at a time
            strides = [1, 1, 2, 1]

            # Perform the max pooling
            pool = tf.nn.max_pool(output2_reshaped, ksize, strides, padding='SAME')

            # Remove the "height" dimension again
            output_3 = tf.squeeze(pool, 1)
            output_3 = tf.Print(output_3, [output_3], message="output_3: ", summarize=1000)


            # -------------------------------------------- Convolution 2--------------------------------------------

            # Create the filter which has shape [filter_width, in_channels, out_channels]
            filter_4 = tf.constant(
                [[[-0.99215692, 0.99215692, -0.52156866, 0.1254902],
                  [0.50980395, -0.99607849, 0.99215692, -0.99215692],
                  [0.44313729, 0.50588238, -0.99215692, 0.99215692],
                  [0.33725491, 0.84313732, -0.99215692, 0.99607849]],
                 [[0.71764708, 0.0627451, -0.19215688, 0.99215692],
                  [-0.19215688, 0.99215692, -0.99215692, 0.99215692],
                  [0.99215692, -0.98039222, 0.56862748, 0.05490196],
                  [-0.60392159, 0.99215692, 0.65882355, -0.27450982]]],
                dtype=tf.float32)

            bias_4 = tf.constant([0.07450981, 0.14901961, 0.98823535, -0.0509804], dtype=tf.float32)

            output_4 = tf.nn.conv1d(output_3, filter_4, 1, "SAME") + bias_4
            output_4 = tf.Print(output_4, [output_4], message="output_4: ", summarize=1000)


            # -------------------------------------------- Max pooling --------------------------------------------

            # Pooling is defined for 2d, so add dim of 1 (height)
            output2_reshaped = tf.expand_dims(output_4, 1)
            # Kernel looks at 1 sample, 1 height, 2 width, and 1 depth
            ksize = [1, 1, 2, 1]

            # Move 1 sample, 1 height, 2 width, and 1 depth at a time
            strides = [1, 1, 2, 1]

            # Perform the max pooling
            pool = tf.nn.max_pool(output2_reshaped, ksize, strides, padding='SAME')

            # Remove the "height" dimension again
            output_5 = tf.squeeze(pool, 1)
            output_5 = tf.Print(output_5, [output_5], message="output_5: ", summarize=1000)


            # -------------------------------------------- LSTM --------------------------------------------
            lstm_units = 4

            LSTM_weights = [
                [0.047461336, -0.2525125, 0.133023, 0.11263981, 0.34583206, -0.61531019, -0.8803405, 0.99488366,
                 0.85897743, -0.36209199, -0.63935106, -0.05743177, 0.00316062, -0.61984265, -0.65267929, 0.25477407],
                [0.057721273, 0.0218620435, -0.01197869, 0.052430139, -0.28346617, -0.00683066, -0.71810405, 0.09335204,
                 0.62675115, 0.90367544, 0.39301098, -0.6796605, 0.77573529, 0.72351356, 0.30118468, -0.16851472],
                [-0.857999441, 0.023618397, 0.02000072, 0.185856544, -0.8450011, -0.7853939, -0.00261568, -0.70895696,
                 -0.20551959, -0.2137256, 0.04685076, 0.78725204, 0.6767832, 0.93165379, 0.24688992, 0.54482694],
                [-0.15851247, 0.18707229, -0.03427629, -0.089051459, 0.54172115, 0.94506127, -0.1827192, 0.44763495,
                 0.69120797, 0.29112628, 0.85072496, -0.0153968, -0.39041728, 0.18252702, 0.15263933, 0.35599324],
                [-0.83214537, 0.11154398, -0.028220953, -0.065826703, -0.46254772, -0.04844626, 0.58888535, 0.71568758,
                 -0.39430272, 0.78568475, -0.17918778, -0.79950794, 0.77709696, -0.21645212, 0.2234143, -0.85459363],
                [0.08630577, -0.073923609, 0.12680474, -0.96924272, 0.72734454, 0.0777216, -0.60380081, -0.9895412,
                 0.37282269, 0.16629956, 0.92285417, 0.86485604, -0.13370907, -0.75214074, 0.72669859, -0.261183],
                [0.72360051, 0.82933379, 0.06519956, 0.25991941, 0.11860591, -0.99293746, -0.08927943, -0.56968878,
                 -0.33370412, 0.09363034, 0.13263364, -0.72481075, 0.88884886, 0.41754448, 0.5463333, -0.80689945],
                [0.75863926, -0.16137513, 0.21030268, 0.05861826, 0.16492918, -0.12813282, -0.8740667, -0.0847981,
                 -0.52497674, -0.29709172, -0.3040518, 0.31963997, 0.24175961, -0.91323495, -0.61515323, 0.32519525]]

            LSTM_bias = [0.77461336, -0.28620435, 0.75000072, -0.89051459, -0.46254772, -0.04844626, -0.60380081,
                         -0.56968878, -0.52497674, 0.09363034, 0.92285417, 0.86485604, -0.77709696, 0.21645212,
                         0.15263933, 0.54482694]

            # Create lstm layer
            lstm = tf.contrib.rnn.LSTMCell(lstm_units,
                                           forget_bias=0.)

            # Put it into Multi RNN Cell
            lstm = tf.contrib.rnn.MultiRNNCell([lstm] * 1)

            # Let dynamic rnn setup the control flow (making while loops and stuff)
            output_lstm, _ = tf.nn.dynamic_rnn(lstm, output_5, dtype=tf.float32)

            # Use only the last hidden state
            output_6 = tf.slice(output_lstm, [0, 1, 0], [1, 1, 4])
            output_6 = tf.Print(output_6, [output_6], message="output_6: ", summarize=1000)


            # Construct operation for assigning mock weights
            kernel = next(i for i in tf.global_variables() if i.shape == (8, 16))
            assign_kernel = kernel.assign(LSTM_weights)

            # Construct operation for assigning mock bias
            bias = next(i for i in tf.global_variables() if i.shape == (16,))
            assign_bias = bias.assign(LSTM_bias)

            # -------------------------------------------- FC 2--------------------------------------------
            weights_7 = tf.constant([[0.86485604, 0.76485604],
                                     [0.75214074, 0.65214074],
                                     [-0.13370907, -0.23370907],
                                     [0.72669859, 0.62669859]],
                                    dtype=tf.float32)

            bias_7 = tf.constant([0.85856544, 0.75856544], dtype=tf.float32)

            # Remove the "height" dimension to fit the required input shape of the matmul
            output_6_reshaped = tf.squeeze(output_6, 1)

            # Perform the matmul
            output_7 = tf.matmul(output_6_reshaped, weights_7) + bias_7
            output_7 = tf.Print(output_7, [output_7], message="output_7: ", summarize=1000)


            # -------------------------------------------- Softmax -------------------------------------------

            output_final = tf.nn.softmax(output_7)

            # -------------------------------------------- LRP -------------------------------------------

            # Get the explanation from the LRP framework.
          #  R = lrp.lrp(inp, output_final)

            # Run the computations
            with tf.Session() as s:
                # Initialize variables
                s.run(tf.global_variables_initializer())

                # Assign mock bias
                s.run([assign_kernel, assign_bias])

                out = s.run(output_final)
                print(out)

                # # Calculate relevance
                # relevances = s.run(R)
                #
                # # Expected result calculated in
                # # https://docs.google.com/spreadsheets/d/1_bmSEBSWVOkpdlZYEUckgrnUtxhEfnR84LZy1cU5fIw/edit?usp=sharing
                # expected_result = np.array([[[0, 0.00009805561105, 0.00005414047329, 0.00006195665023, 0.0002238479865,
                #                               0.00000115545024, 0, 0, 0.00009169002404, 0.0005260735137],
                #                              [0.0005442203312, 0.0009426821562, 0.00003942595867, 0.00003825591233,
                #                               0.0001996159965, 0.0002715702849, 0.00000357598169, 0.001143195536, 0, 0],
                #                              [0.0001969616413, 0.0006704347526, 0.0002918527529, 0, 0.000281341314,
                #                               0.00002460706276, 0, 0, 0.00004603479328, 0.000009408314461],
                #                              [0.000304159758, 0.000323865212, 0.0005835943529, 0.001749189186,
                #                               0.0006260772858, 0.001044452983, 0.001213588422, 0.002556191351,
                #                               0.00008511880281, 0.0001288592067],
                #                              [0.003942198954, 0, 0.0008948157171, 0.001034487118, 0.00821824816,
                #                               0.00795311754, 0.001054496115, 0.004341330156, 0.0006469008048,
                #                               0.0002727043103],
                #                              [0.00001670708886, 0.0004522287794, 0, 0.0125221455, 0.00008588814505,
                #                               0.003642134889, 0.001539205482, 0, 0.002505662528, 0.00006380696649],
                #                              [0.00461172526, 0.004842762115, 0.002068858186, 0.005723975178,
                #                               0.00000168020186, 0.009736343488, 0.0001198417775, 0.003214726948,
                #                               0.0004977156855, 0.0003426242566],
                #                              [0.03019218, 0.001282057794, 0.0004495161327, 0.02870312857, 0,
                #                               0.03205014973, 0.003254075662, 0.07376802562, 0.02590262033,
                #                               0.03669102436]]])
                #
                # # Check for shape and actual result
                # self.assertEqual(expected_result.shape, relevances.shape,
                #                  "Shapes of expected relevance and relevance should be equal")
                # self.assertTrue(np.allclose(relevances, expected_result, rtol=1e-03, atol=1e-03),
                #                 "The relevances do not match")
