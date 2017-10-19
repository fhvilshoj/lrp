import tensorflow as tf
import numpy as np
import unittest
from lrp import lrp
from lrp_util import get_operations_between_output_and_input

class TestSplitAndConcatenate(unittest.TestCase):
    def runTest(self):
        with tf.Graph().as_default():
            inp = tf.placeholder(tf.float32, shape=(3, 4))

            # Split input to construct path in lrp with two paths from
            # input to output.
            in1, in2 = tf.split(inp, 2, 1)

            # Do additional work on one of the paths
            w = tf.constant([[1, 2],
                             [3, 4]],
                            dtype=tf.float32)
            in1 = tf.matmul(in1, w)

            # Concatenate the paths again in order to end up with one output.
            out = tf.concat([in1, in2], 1)

            _, path = get_operations_between_output_and_input(inp, out)
            print(path)
            for p in path:
                print(p._id, p.type)

            expl = lrp.lrp(inp, out)

            # Do some testing
            # self.assertEqual(in1.shape, (3, 2))
            # self.assertEqual(inp.shape, out.shape)
            # self.assertEqual(inp.shape, expl.shape)

            # with tf.Session() as s:
            #     pass
            pass
