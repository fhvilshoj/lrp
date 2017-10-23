import tensorflow as tf
from lrp.linear_lrp import linear_epsilon
from lrp import lrp_util
from constants import *


def _t_geq_one(t, *_):
    return tf.greater_equal(t, 1)


def _find_operation_from_path(path, operation):
    """
    This procedure takes a sub path of the output
    to input path and searches through it until it
    finds the operation type specified.
    :param path: a sub path from output to input
    :param operation: operation type (string)
    :return: returns first occurence in path of operation
    specified
    """
    for op in path:
        if op.type == operation:
            return op
    return None


def _walk_full_graph_by_path(start, path):
    """
    This procedure starts in operation start and
    looks for input operations of the type given
    in the path argument iteratively.
    :param start: the operation in the graph to start from
    :param path: an array of operation types (strings) to follow one at a time
    :return: the operation at the end of the path
    """
    o = start
    for p in path:
        for i in o.inputs:
            if i.op.type == p:
                o = i.op
                break
    return o

# Find the BiasAdd in the LSTM, since this is a good starting point for getting all the information
# about the LSTM we need
def _find_bias_add_operation_from_path(path):
    # Looking for an operation which is a BiasAdd where the input is a MatMul
    # and one of the consumers of the BiasAdd is a Split
    for op in path:
        # Check if the current operation in the path is a BiasAdd
        # If not skip to next operation
        if 'BiasAdd' in op.type:
            # Check if the operation has a Split operation as consumer
            has_split_consumer = False
            for consumer in op.outputs[0].consumers():
                if 'Split' in consumer.type:
                    has_split_consumer = True
                    break

            # Check if the operation has a MatMul as input
            has_matmul_input = False
            for input in op.inputs:
                if 'MatMul' in input.op.type:
                    has_matmul_input = True
                    break

            if has_split_consumer and has_matmul_input:
                # We have found the operation we were looking for
                return op

    # If none of the operations complied to the constraints we are in trouble
    raise ValueError("Cannot find LSTM in LSTM Context")

def lstm(path, R, LSTM_input):
    """
    Finds weights and bias and does forward pass inclusive recordings
    of activations
    :param path: path with all the operations belonging to the LSTM.
    :param R: Relevance for upper layer
    :param LSTM_input: the tensor which is input to the LSTM
    """

    # Find kernel, bias and additional forget bias
    bias_add_op = _find_bias_add_operation_from_path(path)
    kernel_ref = _walk_full_graph_by_path(bias_add_op, ['MatMul', 'Enter', 'Identity', 'VariableV2'])
    bias_ref = _walk_full_graph_by_path(bias_add_op, ['Enter', 'Identity', 'VariableV2'])

    # Both kernel and bias are references to actual objects
    kernel_ref = kernel_ref.outputs[0]
    bias_ref = bias_ref.outputs[0]

    # TODO Should be removed if we dont expand kernel no more
    # Expand the dimension of the kernel to be able to multiply it with the input || hidden state vector
    # below. The shape of the kernel is (input_depth + hidden_state_depth, units) is before the expand_dims
    # and (1, input_depth + hidden_state_depth, units) after the expand_dims

    # Finding extra bias associated with the forget gate
    # TODO We do not handle forget bias
    # before_output_gate = _find_operation_from_path(while_context, 'Tanh')
    # add_forget_bias_op = _walk_full_graph_by_path(before_output_gate, ['Add', 'Mul', 'Sigmoid', 'Add'])
    # forget_bias = tf.identity(add_forget_bias_op.inputs[1])

    # Find the number of LSTM units by dividing length of bias by 4
    # (LSTM_input, gate, forget and output)
    lstm_units = bias_ref.get_shape().as_list()[0] // 4

    # Find the length of the LSTM_input sequence
    batch_size, sequence_length, input_depth = LSTM_input.get_shape().as_list()

    # Slicing the weights and biases used for calculating gate gate
    # output (Ug and Wg). The kernel matrix has the following regions:
    # ---------------------
    # | Ui | Ug | Uf | Uo |
    # ---------------------
    # | Wi | Wg | Wf | Wo |
    # ---------------------
    # For LSTM implementation details see
    # https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/rnn_cell_impl.py#L565
    # where Us are used to weight LSTM_input and Ws are used for h^(t-1).
    # The same goes for bias.
    weights_gate_gate = tf.slice(kernel_ref, [0, lstm_units], [kernel_ref.get_shape().as_list()[0], lstm_units])
    bias_gate_gate = tf.slice(bias_ref, [lstm_units], [lstm_units])

    # Prepare for forward pass by constructing Tensor Arrays for LSTM_input,
    # hidden states, state cells, LSTM_input gate, gate gate, and forget_gate
    # We do not record output gate since it is not needed for the relevance calculations.
    # Just empty TAs with 0's in the first entry.
    input_a = tf.TensorArray(tf.float32, size=sequence_length, clear_after_read=False)
    # LSTM_input: ( batch_size, time, depth )
    LSTM_input = tf.transpose(LSTM_input, [1, 0, 2])

    # LSTM_input: ( time, batch_size, depth )
    # input_a: TensorArray containing sizes: (1, batch_size, depth)
    input_a = input_a.split(LSTM_input, [1] * sequence_length)

    def create_ta_with_0_at_idx_zero(clear_after_read=True):
        ta = tf.TensorArray(tf.float32, size=sequence_length + 1, clear_after_read=clear_after_read)
        return ta.write(0, tf.constant(0., shape=(batch_size, lstm_units)))

    hidden_states = create_ta_with_0_at_idx_zero(False)
    state_cells = create_ta_with_0_at_idx_zero(False)
    input_gate = create_ta_with_0_at_idx_zero()
    gate_gate = create_ta_with_0_at_idx_zero()
    forget_gate = create_ta_with_0_at_idx_zero()

    # While body for recalculating forward pass of lstm
    # in order to record the outputs of the different
    # states and gates.
    def _body(t, hs, sc, ig, gg, fg):
        # Read LSTM_input for time t (NOT time t-1!!) and hidden state, cell state for time t - 1
        i = tf.squeeze(input_a.read(t - 1), 0)
        h = hs.read(t - 1)
        s = sc.read(t - 1)

        # Concatenate LSTM_input and previous hidden state to
        # be able to multiply with kernel.
        tm = tf.concat([i, h], 1)

        # Multiply tm with kernel and add bias.
        before_activation_functions = tf.nn.bias_add(tf.matmul(tm, kernel_ref), bias_ref)

        # Split result into the four regions mentioned above.
        new_ig, new_gg, new_fg, new_og = tf.split(before_activation_functions, 4, axis=1)

        # Apply appropriate activation functions
        new_ig = tf.sigmoid(new_ig)
        new_fg = tf.sigmoid(new_fg)  # + forget_bias  TODO: what to do with forget bias
        new_gg = tf.tanh(new_gg)     # TODO: may need dynamic activation function
        new_og = tf.sigmoid(new_og)

        # Calculate new state cell
        new_state_cell = tf.add(tf.multiply(new_gg, new_ig),
                                tf.multiply(new_fg, s))

        # Calculate new hidden state from new cell state and output gate
        new_hidden_state = tf.multiply(tf.tanh(new_state_cell), new_og)

        # Store all the information that we need
        hs = hs.write(t, new_hidden_state)
        sc = sc.write(t, new_state_cell)
        ig = ig.write(t, new_ig)
        gg = gg.write(t, new_gg)
        fg = fg.write(t, new_fg)

        # Carry on
        return tf.add(t, 1), hs, sc, ig, gg, fg

    # Do the while loop
    i = tf.constant(1)
    time, hs, sc, ig, gg, fg = tf.while_loop(
        cond=lambda i, *_: tf.less_equal(i, sequence_length),
        body=_body,
        loop_vars=[i, hidden_states, state_cells, input_gate, gate_gate, forget_gate])

    # Call lstm lrp with the recorded information
    # R has shape (batch_size, predictions_pr_sample, units)
    # Transpose it to have shape (predictions_pr_sample, batch_size, units)
    R = tf.transpose(R, [1, 0, 2])

    # TODO Update to for loop FAKE!
    # R_tmp has shape (1, batch_size, units)
    R_tmp = tf.expand_dims(R[0], 0)

    R_new = _calculate_relevance_from_lstm(R_tmp, weights_gate_gate, bias_gate_gate, input_a, hs, sc, ig, gg, fg,
                                           sequence_length)

    R_new = tf.transpose(R_new, [1, 0, 2])

    return R_new


def _calculate_relevance_from_lstm(R, W_g, b_g, X, H, cell_states, input_gate_outputs, gate_gate_outputs,
                                   forget_gate_outputs, max_timestep):
    """
    Calculated the relevance for an LSTM
    :param tensor: the tensor of the upper activation of the LSTM
    :param R: The upper layer relevance
    :return: lower layer relevance
    """
    # Find batch size (X holds tensors of size (1, batch_size, input depth)
    _, batch_size, _ = X.read(0).get_shape().as_list()

    # Find the shape of the weights used for gate gate
    tmp, units = W_g.get_shape().as_list()

    # We know that the height of the weights (tmp) is input_depth + number of lstm units
    # since the number of units is equal to the depth of h
    x_depth = tmp - units

    # Initialize relevances for X[t]
    relevance_xs = tf.TensorArray(tf.float32, max_timestep + 1, clear_after_read=False)
    relevance_xs = relevance_xs.write(0, tf.zeros((batch_size, x_depth)))

    # Initialize relevances for the hidden states
    # R shape is (1, batch_size, units)
    # Reshape to (batch_size, units)
    R = tf.squeeze(R, 0)
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
        print("rel_cs_t shape:", rel_cs_t.shape)
        print("from_old_cell_state shape:", from_old_cell_state.shape)
        print("Weights shape: ", tf.eye(units).shape)
        print("cell_states_t shape:", cell_states_t.shape)

        rel_cs_t_minus_one = linear_epsilon(rel_cs_t, from_old_cell_state, tf.eye(units), output=cell_states_t)

        # The relevance of the gate gate in time t is found using lrp for linear
        # layers with relevance to distribute = relevance for cell state time t,
        # input = (gate gate time t * input gate time t), weights = identity
        # (i.e. the input is not changed), output = cell state time t. Notice
        # that we don't have to care about the input gate, since relevance is
        # just forwarded through to the actual input as described in the paper

        gg_and_ig_product = tf.multiply(gate_gate_t, input_gate_t)
        relevance_g = linear_epsilon(rel_cs_t, gg_and_ig_product, tf.eye(units), output=cell_states_t)

        # The relevance of x in time t and h in time t-1 are found using lrp for
        # linear layers with relevance to distribute = relevance for gate gate in time t,
        # input = (x[t] h[t-1]), weights = W, output = (x[t] h[t-1]) * W.
        h_t_minus_one = H.read(t - 1)
        x_t = tf.squeeze(X.read(t - 1), 0)  # Note that X is indexed from 0 where H is from 1
        x_h_concat = tf.concat([x_t, h_t_minus_one], axis=1)

        rel_x_t_and_h_t_minus_one = linear_epsilon(relevance_g, x_h_concat, W_g, bias=b_g)
        (rel_xs_t, rel_hs_t_minus_one) = tf.split(rel_x_t_and_h_t_minus_one, [x_depth, units], 1)

        rel_xs = rel_xs.write(t, rel_xs_t)
        rel_hs = rel_hs.write(t - 1, rel_hs_t_minus_one)

        t = tf.add(t, -1)

        return t, rel_xs, rel_cs_t_minus_one, rel_hs

    # The while loop
    t, rel_xs, rel_cs, rel_hs = tf.while_loop(_t_geq_one, calculate_relevances, [t, relevance_xs, relevance_cs,
                                                                                 relevance_hs])

    R_new = rel_xs.stack()

    # Remove the first zero row from the tensor
    R_new = tf.slice(R_new, [1, 0, 0], [-1, -1, -1])

    return R_new
