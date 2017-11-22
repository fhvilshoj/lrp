import tensorflow as tf

def get_sensitivity_analysis(X, output):
    output = tf.Print(output, [tf.shape(output)])
    out_max = tf.argmax(output, axis=-1)
    selection = tf.one_hot(out_max, tf.squeeze(tf.shape(output[-1])), axis=1)

    out_selected = output * selection

    grad = tf.gradients(out_selected, X.values)
    sens = tf.squeeze(tf.pow(grad, 2))
    return tf.SparseTensor(X.indices, sens, X.dense_shape)
