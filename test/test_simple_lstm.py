import tensorflow as tf
import numpy as np
import lrp as lrp
import unittest

class TestSimpleLSTM(unittest.TestCase):
    g = tf.Graph()

    with g.as_default() as g:

        inp = tf.constant([], dtype=tf.float32)

        cells = tf.contrib.rnn.LSTMCell(2)



