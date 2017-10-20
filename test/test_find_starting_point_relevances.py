import tensorflow as tf
import unittest
import numpy as np
from lrp import lrp


# Test the function that finds the starting point relevances for lrp
class FindStartingPointRelevancesTest(unittest.TestCase):
    def runTest(self):
        with tf.Graph().as_default():
            # Create mock predictions with shape (batch_size, number_of_classes) = (3, 10)
            one_prediction_per_sample = tf.constant(
                [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                 [-5, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
                dtype=tf.float32)

            # Create mock predictions with shape (batch_size, predictions_per_sample, number_of_classes) = (2, 3, 4)
            five_predictions_per_sample = tf.constant([[[1, 2, 3, 4],
                                                        [5, 6, 7, 8],
                                                        [-1, -1, -1, -1]],
                                                       [[-1, 2, 3, 4],
                                                        [1, 1, 1, 1],
                                                        [1, 1, 1, 1]]], dtype=tf.float32)


            # Find the starting point relevances by calling lrp (a bit hacky but the function we want to test,
            # _find_starting_point_relevances, is not accessible directly)
            one_prediction_per_sample_starting_point_relevances = lrp.lrp(one_prediction_per_sample,
                                                                          one_prediction_per_sample)
            five_predictions_per_sample_starting_point_relevances = lrp.lrp(five_predictions_per_sample,
                                                                            five_predictions_per_sample)

            # Run the computations
            with tf.Session() as s:
                one_prediction_per_sample_starting_point_rel = s.run(
                    one_prediction_per_sample_starting_point_relevances)
                five_predictions_per_sample_starting_point_rel = s.run(
                    five_predictions_per_sample_starting_point_relevances)

                # Create expected starting point relevances
                expected_starting_point_one_prediction_per_sample = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 10]],
                                                                              [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                                                              [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]])

                expected_starting_point_five_predictions_per_sample = np.array([[[0, 0, 0, 4],
                                                                                 [0, 0, 0, 8],
                                                                                 [-1, 0, 0, 0]],
                                                                                [[0, 0, 0, 4],
                                                                                 [1, 0, 0, 0],
                                                                                 [1, 0, 0, 0]]])
                # Compare the calculated relevances with the expected relevances
                self.assertTrue(np.allclose(expected_starting_point_one_prediction_per_sample,
                                            one_prediction_per_sample_starting_point_rel, rtol=1e-3, atol=1e-3),
                                "Did not find the correct starting point relevances for one prediction per sample")

                self.assertTrue(np.allclose(expected_starting_point_five_predictions_per_sample,
                                            five_predictions_per_sample_starting_point_rel, rtol=1e-3, atol=1e-3),
                                "Did not find the correct starting point relevances for five predictions per sample")