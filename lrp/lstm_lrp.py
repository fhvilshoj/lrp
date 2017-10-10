from lrp import lrp
import tensorflow as tf


def linear_lrp_helper(relevance_to_distribute, input, weights, output):
    activations = tf.multiply(input, weights)
    fractions = tf.divide(activations, output)
    relevances = tf.matmul(relevance_to_distribute, fractions)
    return relevances


def get_LSTM_weights_and_biases_for_g(tensor):
    return (tf.constant([[1, 2, 3],
                         [1, 2, 3],
                         [1, 2, 3]]), tf.constant([5, 5, 5]))


def get_LSTM_input(tensor):
    return tf.constant([[1, 2, 3],
                        [4, 5, 6]])


def get_LSTM_outputs(tensor):
    return tf.constant([[1, 1, 1],
                        [2, 2,2]])


def get_LSTM_g_outputs():
    return tf.constant([[1, 1, 1],
                        [2, 2, 2]])

def get_LSTM_i_outputs():
    return tf.constant([[1, 1, 1],
                        [2, 2, 2]])

def get_LSTM_f_outputs():
    return tf.constant([[1, 1, 1],
                        [2, 2, 2]])

def get_LSTM_o_outputs():
    return tf.constant([[1, 1, 1],
                        [2, 2, 2]])

def get_LSTM_cell_states():
    return tf.constant([[1, 1, 1],
                        [2, 2, 2]])


def t_larger_than_zero(t):
    return tf.less(0, t)



def lstm(tensor, R):
    """
    Convolutional lstm
    :param tensor: the tensor of the upper activation of the LSTM
    :param R: The upper layer relevance
    :return: lower layer relevance
    """

    # Get the weights and the biases
    (W_g, b_g) = get_LSTM_weights_and_biases_for_g(tensor)

    # Get the input (i.e. X) to the LSTM and find the number of time steps
    X = get_LSTM_input(tensor)
    max_timestep = X.size()

    # Get the outputs of the LSTM and find the number of units in the LSTM
    H = get_LSTM_outputs(tensor)
    (_, units) = H.read(0).get_shape().as_list()

    # Initialize relevances for X[t]
    relevance_xs = tf.TensorArray(tf.float32, max_timestep)

    # Initialize relevances for the hidden states
    relevance_hs = tf.TensorArray(tf.float32, max_timestep).split(tf.zeros_like(H.stack()), [1]*max_timestep)

    # Initialize relevances for the cell states
    relevance_cs = tf.TensorArray(tf.float32, max_timestep, clear_after_read=False).split(tf.zeros_like(H.stack()), [1]*max_timestep)

    # Find the outputs of the gate gate, the input gate, the forget gate, the output gate, and the cell state at all time steps
    gate_gate_outputs = get_LSTM_g_outputs()
    input_gate_outputs = get_LSTM_i_outputs()
    forget_gate_outputs = get_LSTM_f_outputs()
    cell_states = get_LSTM_cell_states()

    # Use the relevance from the upper layer as a starting point
    relevance_hs = relevance_hs.write(max_timestep - 1, R)

    # The counting variable for the while loop
    t = tf.Variable(max_timestep - 1, dtype=tf.int32)

    # The body of the while loop
    def calculate_relevances(t, rel_xs, rel_cs, rel_hs):
        # The relevance from the hidden state time t is added to the cell state time t (which might have some relevance from the previous time step)
        new_rel_cs = tf.add(rel_hs.read(t), rel_cs.read(t))
        rel_cs = rel_cs.write(t, new_rel_cs)

        # The relevance of the cell state in time t-1 is found using lrp for linear layers with relevance to distribute = relevance for cell state time t,
        # input = (forget gate time t * cell state time t-1), weights = identity (i.e. the input is not changed), output = cell state time t.
        rel_cs = rel_cs.write(t - 1, linear_lrp_helper(relevance_cs.read(t), tf.multiply(forget_gate_outputs.read(t), cell_states.read(t - 1)), tf.eye(units), cell_states.read(t)))

        # The relevance of the gate gate in time t is found using lrp for linear layers with relevance to distribute = relevance for cell state time t,
        # input = (gate gate time t * input gate time t), weights = identity (i.e. the input is not changed), output = cell state time t. Notice that we don't have to care about the
        # input gate, since relevance is just forwarded through to the actual input as described in the paper
        relevance_g = linear_lrp_helper(relevance_cs.read(t), tf.multiply(gate_gate_outputs.read(t), input_gate_outputs.read(t)), tf.eye(units), cell_states.read(t))

        # The relevance of x in time t and h in time t-1 are found using lrp for linear layers with relevance to distribute = relevance for gate gate in time t,
        # input = (x[x] h[t-1]), weights = W, output = (x[x] h[t-1]) * W.
        x_h_concat = tf.concat(X.read(t), H.read(t))
        (rel_xs_t, rel_hs_t_minus_one) = tf.split(linear_lrp_helper(relevance_g, x_h_concat, W_g, tf.multiply(x_h_concat, W_g)), [units], 1)
        rel_xs = rel_xs.write(t, rel_xs_t)
        rel_hs = rel_hs.write(t-1, rel_hs_t_minus_one)
        t = tf.add(t, -1)
        return t, rel_xs, rel_cs, rel_hs

    # The while loop
    (t, relevance_xs, relevance_cs, relevance_hs) = tf.while_loop(t_larger_than_zero(t), calculate_relevances, [t, relevance_xs, relevance_cs, relevance_hs])

    return relevance_xs
