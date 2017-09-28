import tensorflow as tf
import numpy as np
import re
from lrp import lrp

g = tf.Graph()
with g.as_default():
    inp = tf.placeholder(tf.float32, shape=(1, 3))

    with tf.name_scope('linear'):
        weights = tf.constant(
            np.array([
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]]), dtype=tf.float32)

        biases = tf.constant([[1, 1, 1]], dtype=tf.float32)
        mul = tf.matmul(inp, weights)
        activation = tf.nn.relu(mul + biases)

    with tf.name_scope('linear'):
        weights = tf.constant(
            np.array([
                [1, 1],
                [1, 1],
                [1, 1]]), dtype=tf.float32)
        biases = tf.constant([[1, 1]], dtype=tf.float32)
        mul = tf.matmul(activation, weights)
        activation = tf.nn.relu(mul + biases)

    pred = tf.nn.softmax(activation)
    expl = lrp.lrp(pred)

    with tf.Session() as sess:
        print('\n########### START SESSION ###########')
        sess.run(tf.global_variables_initializer())
        prediction, explanation = sess.run([pred, expl], feed_dict={inp: [[-1, 2, 3]]})
        print(prediction)
        print(explanation)
