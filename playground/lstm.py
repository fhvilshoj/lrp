import tensorflow as tf
import numpy as np

from lrp import lrp

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

with tf.Graph().as_default() as g:
    inp = tf.constant(1., shape=(1, 3, 3), dtype=tf.float32)

    cell = tf.contrib.rnn.LSTMCell(3, initializer=tf.constant_initializer(1.), forget_bias=0.)
    cell.add_variable('k', (6,12), initializer=tf.constant_initializer(1.))
    cell.add_variable('b', (12,), initializer=tf.constant_initializer(0.))

    output = tf.nn.dynamic_rnn(cell, inp, dtype=tf.float32)
    print(output)


    known_operations = ['MatMul', 'Conv2D', 'ExpandDims', 'Squeeze', 'Reshape', 'MaxPool', 'Relu', 'Sigmoid', 'Tanh']

    def p(t, indent = "|", idx = 0):

        print(indent, t.op.type)
        depth = 20
        if t.op.inputs and idx < depth:

            for i in t.op.inputs:
                p(i, indent + " |", idx + 1)
        elif idx == depth:
            print(indent + "...")

    p(tf.nn.softmax(output[0]))

    writer = tf.summary.FileWriter('/tmp/tf/lstm')
    writer.add_graph(g)

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        print("Weights", s.run(cell.weights[0]))
        print("Biases", s.run(cell.weights[1]))
        print(s.run(output))

