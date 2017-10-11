from lrp import lrp
import tensorflow as tf

from lrp.lrp_util import print_value

def linear_lrp_helper(relevance_to_distribute, input, weights, output):
    # input = tf.Print(input, [relevance_to_distribute, input, weights, output],
    #                  message="\n\n HERE IS THE LINEAR HELPER \n\n", summarize=1000)
    activations = tf.multiply(tf.transpose(input), weights)
    fractions = tf.divide(activations, output)
    relevances = tf.matmul(relevance_to_distribute, tf.transpose(fractions))
    return relevances


def t_geq_one(t, *_):
    return tf.greater_equal(t, 1)


def lstm(R, W_g, b_g, X, H, cell_states, input_gate_outputs, gate_gate_outputs, forget_gate_outputs, max_timestep):
    """
    Convolutional lstm
    :param tensor: the tensor of the upper activation of the LSTM
    :param R: The upper layer relevance
    :return: lower layer relevance
    """

    (tmp, units) = W_g.get_shape().as_list()
    x_depth = tmp - units

    # Initialize relevances for X[t]
    relevance_xs = tf.TensorArray(tf.float32, max_timestep + 1, clear_after_read=False)
    relevance_xs = relevance_xs.write(0, tf.zeros((1, x_depth)))

    # Initialize relevances for the hidden states
    R_shape = R.get_shape().as_list()
    relevance_hs = tf.TensorArray(tf.float32, (max_timestep + 1), clear_after_read=False)

    # Use the relevance from the upper layer as a starting point
    relevance_hs = relevance_hs.write(max_timestep, R)

    # Initialize relevances for the cell states
    relevance_cs = tf.zeros(R_shape)

    # The counting variable for the while loop
    t = tf.Variable(max_timestep, dtype=tf.int32)

    # The body of the while loop
    def calculate_relevances(t, rel_xs, rel_cs, rel_hs):
        # The relevance from the hidden state time t is added to the cell
        # state time t (which might have some relevance from the previous time step)
        rel_hs_t = rel_hs.read(t)
        rel_cs_t = tf.add(rel_hs_t, rel_cs)

        # The relevance of the cell state in time t-1 is found using lrp for linear
        # layers with relevance to distribute = relevance for cell state time t,
        # input = (forget gate time t * cell state time t-1), weights = identity
        # (i.e. the input is not changed), output = cell state time t.
        cell_states_t = cell_states.read(t)
        forget_gate_t = forget_gate_outputs.read(t)
        input_gate_t = input_gate_outputs.read(t)
        gate_gate_t = gate_gate_outputs.read(t)

        from_old_cell_state = tf.multiply(forget_gate_t, cell_states.read(t - 1))
        rel_cs_t_minus_one = linear_lrp_helper(rel_cs_t, from_old_cell_state, tf.eye(units), cell_states_t)

        # The relevance of the gate gate in time t is found using lrp for linear
        # layers with relevance to distribute = relevance for cell state time t,
        # input = (gate gate time t * input gate time t), weights = identity
        # (i.e. the input is not changed), output = cell state time t. Notice
        # that we don't have to care about the input gate, since relevance is
        # just forwarded through to the actual input as described in the paper

        gg_and_ig_product = tf.multiply(gate_gate_t, input_gate_t)
        relevance_g = linear_lrp_helper(rel_cs_t, gg_and_ig_product, tf.eye(units), cell_states_t)

        # The relevance of x in time t and h in time t-1 are found using lrp for
        # linear layers with relevance to distribute = relevance for gate gate in time t,
        # input = (x[t] h[t-1]), weights = W, output = (x[t] h[t-1]) * W.
        h_t_minus_one = H.read(t - 1)
        x_t = X.read(t - 1)             # Note that X is indexed from 0 where H is from 1
        x_h_concat = tf.concat([x_t, h_t_minus_one], axis=1)
        gate_gate_before_tanh = tf.matmul(x_h_concat, W_g)

        rel_x_t_and_h_t_minus_one = linear_lrp_helper(relevance_g, x_h_concat, W_g, gate_gate_before_tanh)
        (rel_xs_t, rel_hs_t_minus_one) = tf.split(rel_x_t_and_h_t_minus_one, [x_depth, units], 1)

        rel_xs = rel_xs.write(t, rel_xs_t)
        rel_hs = rel_hs.write(t-1, rel_hs_t_minus_one)

        t = tf.add(t, -1)

        return t, rel_xs, rel_cs_t_minus_one, rel_hs

    # The while loop
    t, rel_xs, rel_cs, rel_hs = tf.while_loop(t_geq_one, calculate_relevances, [t, relevance_xs, relevance_cs,
                                                                                relevance_hs])

    R_new = rel_xs.stack()

    # Remove the second dimension (came from each element in the tensor array
    # being shape (1, input size)
    R_new = tf.squeeze(R_new, axis=[1])

    # Remove the first zero row from the tensor
    R_new = tf.slice(R_new, [1, 0], [-1, -1])

    return R_new
