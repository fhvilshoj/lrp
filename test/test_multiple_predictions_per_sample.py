# import tensorflow as tf
# import numpy as np
# import unittest
# from lrp import lrp
#
#
# class MultiplePredictionsPerSampleTest(unittest.TestCase):
#     def runTest(self):
#         with tf.Graph().as_default():
#             # Create input of shape (2, 1, 4)
#             inp = tf.constant([[1, 2, 3, 4],
#                                [1, -1, 0, -10]],
#                               dtype=tf.float32
#                               )
#
#             # --------------------- Linear layer -----------------------------
#             # Setup the linear layer
#
#             # Create weights of shape (4, 5)
#             weights_1 = tf.constant([[1, 2, 3, 4, 5],
#                                      [-3, 1, 0, 2, 1],
#                                      [1, -1, 1, -1, 1],
#                                      [0, 0, 0, 1, 1]], dtype=tf.float32)
#
#             # Create bias of shape (5, )
#             bias_1 = tf.constant([1, 1, 1, 1, 1], dtype=tf.float32)
#
#             # Calculate the activations
#             output_1 = tf.matmul(inp, weights_1, name="MUL1") + bias_1
#
#             # -------------------------------------------- Convolutional layer ----------------------------------------
#             # Setup the 1D convolutional layer
#
#             # Create a tensor of shape (2,1,2) that holds two filters of width 2. Notice that the 1D conv
#             # operation requires the following dimensions for the kernel
#             # tensor:  (filter_width, in_channels, out_channels)
#             filter_2 = tf.constant(
#                 [[[1, 3]], [[2, 4]]],
#                 dtype=tf.float32)
#
#             # Create bias of shape (2, )
#             bias_2 = tf.constant([2, 2], dtype=tf.float32)
#
#             # Add an extra dimension to fit the expected input shape of conv1d, which is:
#             # (batch, in_width, in_channels)
#             output_1_with_depth_dimension = tf.expand_dims(output_1, -1)
#
#             # Perform the convolution
#             output_2 = tf.nn.conv1d(output_1_with_depth_dimension, filter_2, 1, "SAME") + bias_2
#
#             # -------------------------------------------- Max pooling layer -----------------------------------------
#             # Setup the max pooling layer
#
#             # Pooling is defined for 2d, so add dim of 1 (height) to match the input shape required by max_pool:
#             # (batch, height, width, channels)
#             output_2_reshaped = tf.expand_dims(output_2, 1)
#
#             # Kernel looks at 1 sample, 1 height, 2 width, and 1 depth
#             ksize = [1, 1, 2, 1]
#
#             # Move 1 sample, 1 height, 2 width, and 1 depth at a time
#             strides = [1, 1, 2, 1]
#
#             # Perform the max pooling
#             output_3 = tf.nn.max_pool(output_2_reshaped, ksize, strides, padding='SAME')
#
#             # -------------------------------------------- Flatten output -----------------------------------------
#             final_out = tf.reshape(output_3, (2, 1, 6))
#
#             # -------------------------------------------- Calculate LRP -----------------------------------------
#
#             # Make mock relevances with two relevances per sample in the batch, i.e. simulate that there
#             # are multiple predictions per sample
#             R_mock = tf.constant([[[1, 1, 1, 1, 1, 1],
#                                    [2, 2, 2, 2, 2, 2]],
#                                   [[3, 3, 3, 3, 3, 3],
#                                    [4, 4, 4, 4, 4, 4]]],
#                                  dtype=tf.float32)
#
#             expl = lrp._lrp(inp, final_out, R_mock)
#
#             with tf.Session() as s:
#                 predictions, explanation = s.run([final_out, expl])
#
#
#                 # Expected result calculated in
#                 # https://docs.google.com/spreadsheets/d/1_bmSEBSWVOkpdlZYEUckgrnUtxhEfnR84LZy1cU5fIw/edit?usp=sharing
#                 expected_relevances = np.array(
#                     [[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]])
#
#                 # Compare the calculated relevances to the expected relevances
#
#                 self.assertTrue(np.allclose(expected_relevances, explanation, rtol=1e-03, atol=1e-03),
#                                 "The relevances do not match")
