import tensorflow as tf
import numpy as np
import unittest
from lrp import lrp


class TestChangingBatchDimension(unittest.TestCase):
    def runTest(self):
        with tf.Graph().as_default():
            batch_size = 4
            sequence_length = 3
            units = 2

            # Shape: (4, 3, 2)
            inp = tf.constant([[[0.25946831, 0.73117677],
                                [-0.86958342, 0.71637971],
                                [1.37765705, -0.94898418]],
                               [[-1.58566558, 0.81390618],
                                [-0.80477955, 0.93804462],
                                [-0.21492174, -0.84860344]],
                               [[1.14450824, 0.76930967],
                                [0.05394846, 0.26253664],
                                [1.47991874, 1.5336967]],
                               [[0.75599386, 0.04976039],
                                [1.37701431, 1.33116121],
                                [0.19442138, -0.2920729]]],
                              dtype=tf.float32)

            # Make a reshape that changes the 'batch_size'
            # New shape: (12, 2)
            inp_reshaped = tf.reshape(inp, (-1, 2))

            # Shape back to the original batch_size: (4, 3, 2)
            out = tf.reshape(inp_reshaped, (batch_size, sequence_length, units))

            # Perform lrp
            R = lrp.lrp(inp, out)

            with tf.Session() as s:
                explanation = s.run(R)

                expected_explanation = np.array([[[[0, 0.73117679],
                                                   [0, 0],
                                                   [0, 0]],

                                                  [[0, 0],
                                                   [0, 0.7163797],
                                                   [0, 0]],

                                                  [[0, 0],
                                                   [0, 0],
                                                   [1.37765706, 0]]],

                                                 [[[0, 0.81390619],
                                                   [0, 0],
                                                   [0, 0]],

                                                  [[0, 0],
                                                   [0, 0.93804461],
                                                   [0, 0]],

                                                  [[0, 0],
                                                   [0, 0],
                                                   [-0.21492174, 0]]],

                                                 [[[1.14450824, 0],
                                                   [0, 0],
                                                   [0, 0]],

                                                  [[0, 0],
                                                   [0, 0.26253664],
                                                   [0, 0]],

                                                  [[0, 0],
                                                   [0, 0],
                                                   [0, 1.53369665]]],

                                                 [[[0.75599384, 0],
                                                   [0, 0],
                                                   [0, 0]],

                                                  [[0, 0],
                                                   [1.37701428, 0],
                                                   [0, 0]],

                                                  [[0, 0],
                                                   [0, 0],
                                                   [0.19442138, 0]]]]
                                                )

                self.assertTrue(np.allclose(expected_explanation, explanation, atol=1e-03, rtol=1e-03),
                                "Explanation does not match expected explanation")
