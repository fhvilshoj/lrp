import tensorflow as tf


def _linear_lrp_helper(relevance_to_distribute, input, weights, output):
    activations = tf.multiply(tf.transpose(input), weights)
    fractions = tf.divide(activations, output)
    relevances = tf.matmul(relevance_to_distribute, tf.transpose(fractions))
    return relevances


def _t_geq_one(t, *_):
    return tf.greater_equal(t, 1)


def _find_operation_from_ts(ts, operation):
    for op in ts:
        if op.type == operation:
            return op
    return None


def _search_operation_path(operation, path):
    o = operation
    for p in path:
        for i in o.inputs:
            if i.op.type == p:
                o = i.op
                break
    return o


def _handle_while_context(while_context, R, input):
    # Find kernel, bias and additional forget bias (potentially h0 and s0)
    bias_add_op = _find_operation_from_ts(while_context, 'BiasAdd')
    kernel_ref = _search_operation_path(bias_add_op, ['MatMul', 'Enter', 'Identity', 'VariableV2'])
    bias_ref = _search_operation_path(bias_add_op, ['Enter', 'Identity', 'VariableV2'])

    kernel_ref = kernel_ref.outputs[0]
    bias_ref = bias_ref.outputs[0]

    before_output_gate = _find_operation_from_ts(while_context, 'Tanh')
    add_forget_bias_op = _search_operation_path(before_output_gate, ['Add', 'Mul', 'Sigmoid', 'Add'])
    forget_bias = tf.identity(add_forget_bias_op.inputs[1])

    state_size = bias_ref.get_shape().as_list()[0] // 4
    sequence_length = input.get_shape().as_list()[1]

    weights_gate_gate = tf.slice(kernel_ref, [0, state_size], [kernel_ref.get_shape().as_list()[0], state_size])
    bias_gate_gate = tf.slice(bias_ref, [state_size], [state_size])

    # Do forward pass to collect gates, cell states and hidden states


    input_a = tf.TensorArray(tf.float32, size=sequence_length, clear_after_read=False)
    input_a = input_a.split(input[0], [1] * sequence_length)

    def get_ta_with_0_at_idx_zero(clear_after_read=True):
        ta = tf.TensorArray(tf.float32, size=sequence_length + 1, clear_after_read=clear_after_read)
        return ta.write(0, tf.constant(0., shape=(1, state_size)))

    hidden_states = get_ta_with_0_at_idx_zero(False)
    state_cells = get_ta_with_0_at_idx_zero(False)
    input_gate = get_ta_with_0_at_idx_zero()
    gate_gate = get_ta_with_0_at_idx_zero()
    forget_gate = get_ta_with_0_at_idx_zero()

    def _body(t, hs, sc, ig, gg, fg):
        i = input_a.read(t - 1)
        h = hs.read(t - 1)
        s = sc.read(t - 1)

        tm = tf.concat([i, h], 1)

        before_activation_functions = tf.nn.bias_add(tf.matmul(tm, kernel_ref), bias_ref)
        new_ig, new_gg, new_fg, new_og = tf.split(before_activation_functions, 4, axis=1)

        new_ig = tf.sigmoid(new_ig)
        new_fg = tf.sigmoid(new_fg)  # + forget_bias  TODO: what to do with forget bias
        new_gg = tf.tanh(new_gg)
        new_og = tf.sigmoid(new_og)

        new_state_cell = tf.add(tf.multiply(new_gg, new_ig),
                                tf.multiply(new_fg, s))

        new_hidden_state = tf.multiply(tf.tanh(new_state_cell), new_og)
        hs = hs.write(t, new_hidden_state)
        sc = sc.write(t, new_state_cell)
        ig = ig.write(t, new_ig)
        gg = gg.write(t, new_gg)
        fg = fg.write(t, new_fg)

        return tf.add(t, 1), hs, sc, ig, gg, fg

    i = tf.constant(1)
    time, hs, sc, ig, gg, fg = tf.while_loop(
        cond=lambda i, *_: tf.less_equal(i, sequence_length),
        body=_body,
        loop_vars=[i, hidden_states, state_cells, input_gate, gate_gate, forget_gate])

    # Call lstm lrp with the found information
    R_new = _calculate_relevance_form_lstm(R[0], weights_gate_gate, bias_gate_gate, input_a, hs, sc, ig, gg, fg, sequence_length)
    return tf.expand_dims(R_new, 0)


def lstm(ts, R):
    while_context = []

    next_idx = 1
    next = ts[next_idx]
    while next.type != 'TensorArrayScatterV3':
        next = ts.pop(0)
        while_context.append(next)

    # Find the transpose operation
    find_id = next.inputs[2].op._id
    while next._id != find_id:
        next = ts.pop(0)


    # Find input to transpose from path but keep operation in pather
    # for further handeling.
    find_id = next.inputs[0].op._id
    while ts[0]._id != find_id:
        ts.pop(0)

    while_context_input = ts[0].outputs[0]
    R = _handle_while_context(while_context, R, while_context_input)

    return ts, R


def _calculate_relevance_form_lstm(R, W_g, b_g, X, H, cell_states, input_gate_outputs, gate_gate_outputs, forget_gate_outputs, max_timestep):
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
        rel_cs_t_minus_one = _linear_lrp_helper(rel_cs_t, from_old_cell_state, tf.eye(units), cell_states_t)

        # The relevance of the gate gate in time t is found using lrp for linear
        # layers with relevance to distribute = relevance for cell state time t,
        # input = (gate gate time t * input gate time t), weights = identity
        # (i.e. the input is not changed), output = cell state time t. Notice
        # that we don't have to care about the input gate, since relevance is
        # just forwarded through to the actual input as described in the paper

        gg_and_ig_product = tf.multiply(gate_gate_t, input_gate_t)
        relevance_g = _linear_lrp_helper(rel_cs_t, gg_and_ig_product, tf.eye(units), cell_states_t)

        # The relevance of x in time t and h in time t-1 are found using lrp for
        # linear layers with relevance to distribute = relevance for gate gate in time t,
        # input = (x[t] h[t-1]), weights = W, output = (x[t] h[t-1]) * W.
        h_t_minus_one = H.read(t - 1)
        x_t = X.read(t - 1)             # Note that X is indexed from 0 where H is from 1
        x_h_concat = tf.concat([x_t, h_t_minus_one], axis=1)
        gate_gate_before_tanh = tf.matmul(x_h_concat, W_g)

        rel_x_t_and_h_t_minus_one = _linear_lrp_helper(relevance_g, x_h_concat, W_g, gate_gate_before_tanh)
        (rel_xs_t, rel_hs_t_minus_one) = tf.split(rel_x_t_and_h_t_minus_one, [x_depth, units], 1)

        rel_xs = rel_xs.write(t, rel_xs_t)
        rel_hs = rel_hs.write(t-1, rel_hs_t_minus_one)

        t = tf.add(t, -1)

        return t, rel_xs, rel_cs_t_minus_one, rel_hs

    # The while loop
    t, rel_xs, rel_cs, rel_hs = tf.while_loop(_t_geq_one, calculate_relevances, [t, relevance_xs, relevance_cs,
                                                                                 relevance_hs])

    R_new = rel_xs.stack()

    # Remove the second dimension (came from each element in the tensor array
    # being shape (1, input size)
    R_new = tf.squeeze(R_new, axis=[1])

    # Remove the first zero row from the tensor
    R_new = tf.slice(R_new, [1, 0], [-1, -1])

    return R_new
