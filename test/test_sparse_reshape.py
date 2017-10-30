import tensorflow as tf
import numpy as np
import unittest
from lrp import lrp


class TestSparseReshape(unittest.TestCase):
    def runTest(self):
        with tf.Graph().as_default() as g:
            # shape: (6,6)
            values = tf.constant([1], dtype=tf.float32, name="vals")
            inp = tf.SparseTensor([[5, 5]], values, [6, 6])
            inp_reordered = tf.sparse_reorder(inp)
            out = tf.sparse_reshape(inp_reordered, (9, 4))

            out_as_dense = tf.sparse_tensor_dense_matmul(out, tf.eye(4, dtype=tf.float32))

            writer = tf.summary.FileWriter("./tmp")
            writer.add_graph(g)

            # Calculate the explanation
            expl = lrp.lrp(inp, out_as_dense)

            # Run a tensorflow session to evaluate the graph
            with tf.Session() as s:
                s.run(inp)
                s.run(out)
                # Initialize the variables
                s.run(tf.global_variables_initializer())

                s.run(out)

                # Calculate the explanation
                explanation = s.run(expl)

                # Extract the indices of non-zero elements, the values of the non-zero elements and the dense shape of
                # the expected explanation
                calculated_indices = explanation[0]
                calculated_values = explanation[1]
                calculated_shape = explanation[2]

                # Create the expected explanation by creating an array that holds the indices of non-zero
                # elements, an array that holds the values of the non-zero elements and an array that holds
                # the dense shape of the expected explanation
                expected_indices = np.array([[5, 5]])
                expected_values = np.array([1])
                expected_shape = np.array([6, 6])

                # Check if the explanation matches the expected explanation
                self.assertTrue(np.array_equal(expected_indices, calculated_indices), msg="The indicies do not match")
                self.assertTrue(np.allclose(expected_values, calculated_values, atol=1e-03, rtol=1e-03),
                                msg="The values do not match")
                self.assertTrue(np.array_equal(expected_shape, calculated_shape), msg="The shapes do not match")
