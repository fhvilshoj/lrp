import tensorflow as tf

def get_random_relevance(X):
    # X shape: (1, sequence_length, features)
    random_order_values = tf.random_shuffle(tf.range(0, tf.size(X.values)))
    sum = tf.reduce_sum(random_order_values)

    # Scale new values to a "probability distribution"
    random_order_values = random_order_values / sum

    return tf.SparseTensor(X.indices, random_order_values, X.dense_shape)
