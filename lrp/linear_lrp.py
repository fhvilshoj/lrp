import tensorflow as tf
from lrp import lrp_util
from configuration import LAYER, RULE, BIAS_STRATEGY
from constants import BIAS_DELTA, EPSILON


def _linear_flat(R, input, weights, config, bias=None):
    # Get width of the input
    input_width = tf.cast(tf.shape(weights)[0], tf.float32)

    # Fractions of size (input_width, output_width)
    # fractions = tf.constant(1 / input_width, shape=tf.shape(weights), dtype=tf.float32)
    fractions = tf.ones_like(weights) / input_width

    # Transpose fractions to be able to multiply with relevances
    # fraction shape after transpose: (output_width, input_width)
    fractions = tf.transpose(fractions)

    # Remember shape of R
    R_shape = tf.shape(R)

    # Reshape R to have only two dimensions to be able to use matrix multiplication
    # R shape becomes: (batch_size * predictions_per_sample, output_width)
    R = tf.reshape(R, (-1, R_shape[-1]))

    # Multiply relevances with fractions to find relevance per feature in input
    # Shape of R: (batch_size, predictions_per_sample, output_width)
    # Shape of fractions: (batch_size, output_width, input_width)
    # Shape of R_new: (batch_size, predictions_per_sample, input_width)
    R_new = tf.matmul(R, fractions)

    # Restore batch_size and predictions_per_sample
    # R_new shape after reshape: (batch_size, predictions_per_sample, input_width)
    R_new = tf.reshape(R_new, (R_shape[0], R_shape[1], -1))

    return R_new


def _linear_ww(R, input, weights, config, bias=None):
    # Square the weights
    # zs Shape: (input_width, output_width
    zs = tf.square(weights)

    # Sum the weights
    # zs_sum shape: (output_width,)
    zs_sum = tf.reduce_sum(zs, 0)

    # Get the fractions by dividing the two
    # fractions Shape: (input_width, output_width)
    fractions = zs / zs_sum

    # Transpose fractions to be able to multiply with relevances
    # fraction shape after transpose: (output_width, input_width)
    fractions = tf.transpose(fractions)

    # Remember shape of R
    R_shape = tf.shape(R)

    # Reshape R to have only two dimensions to be able to use matrix multiplication
    # R shape becomes: (batch_size * predictions_per_sample, output_width)
    R = tf.reshape(R, (-1, R_shape[-1]))

    # Multiply relevances with fractions to find relevance per feature in input
    # Shape of R: (batch_size, predictions_per_sample, output_width)
    # Shape of fractions: (batch_size, output_width, input_width)
    # Shape of R_new: (batch_size, predictions_per_sample, input_width)
    R_new = tf.matmul(R, fractions)

    # Restore batch_size and predictions_per_sample
    # R_new shape after reshape: (batch_size, predictions_per_sample, input_width)
    R_new = tf.reshape(R_new, (R_shape[0], R_shape[1], -1))

    return R_new


def _divide_bias_among_zs(config, zs, bias_to_divide):
    if config.bias_strategy == BIAS_STRATEGY.ALL:
        # Number of input features to divide relevance among (cast to float32 from int to perform the division below)
        zs_shape = tf.shape(zs)
        input_features = tf.cast(zs_shape[1], tf.float32)

        # Divide the bias (and stabilizer) equally between the `input_features` (rows of zs)
        # Shape: (output_width) or (batch, output_width)
        bias_per_feature = bias_to_divide / input_features
        # Expand the second to last dimension to be able to add the bias through the rows of zs
        # Shape: (1, output_width) or (batch, 1, output_width)
        bias_per_feature = tf.expand_dims(bias_per_feature, -2)

    elif config.bias_strategy == BIAS_STRATEGY.ACTIVE:
        # Find all the zijs that are not 0
        # active_zs shape: (batch_size, input_width, output_width)
        active_zs = tf.where(tf.equal(zs, 0), tf.zeros_like(zs), tf.ones_like(zs))

        # For each sample and each neuron count how many active activations
        # to divide the bias equally among
        # counts shape: (batch_size, 1, output_width
        counts = tf.reduce_sum(active_zs, 1, keep_dims=True)

        # Scale all the indicator ones with the bias
        # nonscaled shape: (batch_size, input_width, output_width)
        nonadjusted_biases = (active_zs * tf.expand_dims(bias_to_divide, -2))

        # Replace counts of 0 with one to avoid dividing with zero in next line.
        # We can do this because all the adjusted biases in the column where we
        # replaced the zeroes will be 0 so we will have the fraction 0/1 = 0 as those columns.
        counts = tf.where(tf.equal(counts, 0), tf.ones_like(counts), counts)

        # Adjust the the biases by the counts for each neuron
        bias_per_feature = nonadjusted_biases / counts
    else:
        # If no bias should be divided, return zs as is
        return zs

    # Add bias to zs
    # Shape of zs: (batch, input_width, output_width)
    zs = zs + bias_per_feature
    return zs


def _linear_epsilon(R, input, weights, config, bias=None):
    """
    Simple linear layer used for partial computations of LSTM
    :param R: tensor of relevance to distribute. Shape: (batch_size, output_width)
    :param input: The input tensor. Shape: (batch_size, input_width)
    :param weights: The kernel weights. Shape: (input_width, output_width)
    :param output: Optional output tensor. Shape: (batch_size, output_width)
           If none output is calculated as input times weights plus bias.
    :param bias: Optional tensor with bias. Shape: (output_width) or (batch_size, output_width)
    :return: Redistributed relevance. Shape: (batch_size, input_width)
    """
    # Calculate the output of the layer
    output = tf.matmul(input, weights)
    # Only add bias when bias is not none
    if bias is not None:
        output += bias

    # Prepare batch for elementwise multiplication
    # Shape of input: (batch_size, input_width)
    # Shape of input after expand_dims: (batch_size, input_width, 1)
    input = tf.expand_dims(input, -1)

    # Perform elementwise multiplication of input, weights to get z_kij which is the contribution from
    # feature i to neuron j for input k
    # Shape of zs: (batch_size, input_width, output_width)
    zs = tf.multiply(input, weights)

    # When bias is given divide it equally among the i's to avoid relevance loss
    if bias is not None:
        # Find the bias to divide among the rows (This includes the stabilizer: epsilon)
        # Shape: (output_width) or (batch, output_width)
        bias_to_divide = (BIAS_DELTA * bias + config.epsilon * tf.sign(output))

        zs = _divide_bias_among_zs(config, zs, bias_to_divide)

    # Add stabilizer to denominator to avoid dividing with 0
    # Shape of denominator: (batch, output_width)
    denominator = output + config.epsilon * tf.sign(output)

    # Expand the second to last dimension to be able to divide the denominator through the rows of zs
    # Shape after expand_dims: (batch, 1, output_width)
    denominator = tf.expand_dims(denominator, -2)

    # Find the relative contribution from feature i to neuron j for input k
    # Shape of fractions: (batch_size, input_width, output_width)
    fractions = tf.divide(zs, denominator)

    # Prepare the fractions for the matmul below
    # Shape of fractions after transpose: (batch_size, output_width, input_width)
    fractions = tf.transpose(fractions, [0, 2, 1])

    # Multiply relevances with fractions to find relevance per feature in input
    # Shape of R: (batch_size, predictions_per_sample, output_width)
    # Shape of fractions: (batch_size, output_width, input_width)
    # Shape of R_new: (batch_size, predictions_per_sample, input_width)
    R_new = tf.matmul(R, fractions)

    return R_new


def _linear_alpha(R, input, weights, config, bias=None):
    # Prepare batch for elementwise multiplication
    # Shape of input: (batch_size, input_width)
    # Shape of input after expand_dims: (batch_size, input_width, 1)
    input = tf.expand_dims(input, -1)

    # Perform elementwise multiplication of input, weights to get z_kij which is the contribution from
    # feature i to neuron j for input k
    # Shape of zs: (batch_size, input_width, output_width)
    zs = tf.multiply(input, weights)

    # Function to calculate both positive and negative fractions.
    def _find_fractions(selection, stabilizer_operation):
        # Replace the negative elements with zeroes to only have the positive z's left (i.e. z_kij^+)
        # Shape of zijs: (batch_size, input_width, output_width)
        zijs = selection(zs)

        # Take the sum of each column of z_kij^+'s
        # Shape of zj_sum: (batch_size, 1, output_width)
        zj_sum = tf.reduce_sum(zijs, axis=1, keep_dims=True)

        # Find and add the positive parts of an eventual bias (i.e. find and add the b^+s).
        if bias is not None:
            # Filter elements in bias to either positives of negatives according to selection callable
            bias_filtered = selection(bias)
            # Divide the bias according to the current configuration
            zijs = _divide_bias_among_zs(config, zijs, bias_filtered)

            # Add stabilizer to bias to be able to split that as well
            zj_sum = stabilizer_operation(zj_sum, EPSILON)

            # Add the sum of the z_ij^+'s and the positive bias (i.e. find the z_j^+'s)
            zj_sum = tf.add(zj_sum, bias_filtered)
        else:
            # Add stabilizer to the denominator
            zj_sum = stabilizer_operation(zj_sum, EPSILON)

        # Find the relative contribution from feature i to neuron j for input k
        # Shape of fractions: (batch_size, input_width, output_width)
        return tf.divide(zijs, zj_sum)

    fractions_alpha = config.alpha * _find_fractions(lrp_util.replace_negatives_with_zeros, tf.add)
    fractions_beta = config.beta * _find_fractions(lrp_util.replace_positives_with_zeros, tf.subtract)

    total_fractions = fractions_alpha + fractions_beta

    # Prepare the fractions for the matmul below
    # Shape of fractions after transpose: (batch_size, output_width, input_width)
    total_fractions = tf.transpose(total_fractions, perm=[0, 2, 1])

    # Multiply relevances with fractions to find relevance per feature in input
    # In other words: Calculate the lower layer relevances (a combination of equation 60 and 62 in Bach 2015)
    # Shape of R: (batch_size, predictions_per_sample, output_width)
    # Shape of fractions: (batch_size, output_width, input_width)
    # Shape of R_new: (batch_size, predictions_per_sample, input_width)
    R_new = tf.matmul(R, total_fractions)

    return R_new


def elementwise_linear(router, R):
    """
    Used to handle element wise multiplications (by transforming them into matrices)
    :param router: the router object to report changes to
    :param R: the list of tensors containing the relevances from the upper layers
    """
    # Tensor is the output of the current operation (i.e. Add, or Mul)
    current_operation = router.get_current_operation()
    current_tensor = current_operation.outputs[0]

    # Sum the potentially multiple relevances from the upper layers
    R = lrp_util.sum_relevances(R)

    multensor = current_tensor
    bias = None
    if current_tensor.op.type == 'Add':
        bias = lrp_util.get_input_bias_from_add(current_tensor)
        multensor = lrp_util.find_first_tensor_from_type(current_tensor, 'Mul')

    # Find the inputs to the matrix multiplication
    (input, weights) = multensor.op.inputs
    # Make the weights to a diagonal matrix
    weights = tf.diag(weights)

    # Get the shape of R to find 'batch_size' and 'predictions_per_sample'
    R_shape = tf.shape(R)
    batch_size = R_shape[0]
    predictions_per_sample = R_shape[1]

    # Reshape input and R to be able to "broadcast" weights if the input is rank > 2
    def _rank2():
        return input, R

    def _higher_rank():
        new_input = tf.reshape(input, (-1, tf.shape(input)[-1]))
        d = tf.shape(input)[-1]
        new_R = tf.reshape(R, (-1, predictions_per_sample, d))
        return new_input, new_R

    # Test if the rank of the input is > 2
    new_input, R = tf.cond(tf.equal(tf.rank(input), 2),
                           true_fn=_rank2,
                           false_fn=_higher_rank)

    layer_config = router.get_configuration(LAYER.LINEAR)
    R_new = linear_with_config(R, new_input, weights, layer_config, bias)

    # if layer_config.type == ALPHA_BETA_RULE:
    #     # Calculate new relevances with the alpha rule
    #     R_new = linear_alpha(R, new_input, weights, bias=bias, alpha=layer_config.alpha, beta=layer_config.beta)
    # elif layer_config.type == EPSILON_RULE:
    #     # Calculate new relevances with the epsilon rule
    #     R_new = linear_epsilon(R, new_input, weights, bias=bias, epsilon=layer_config.epsilon)

    # Turn the calculated relevances into the correct form if the rank of the input was > 2
    def _revert_rank2():
        return R_new

    def _revert_higher_rank():
        return tf.reshape(R_new, tf.concat(([batch_size], [predictions_per_sample], tf.shape(input)[1:]), 0))

    # Test if the rank of the input is > 2
    R_new = tf.cond(tf.equal(tf.rank(input), 2),
                    true_fn=_revert_rank2,
                    false_fn=_revert_higher_rank)

    # Mark handled operations
    router.mark_operation_handled(current_operation)
    router.mark_operation_handled(multensor.op)

    # Forward relevance
    router.forward_relevance_to_operation(R_new, multensor.op, input.op)


def linear_with_config(R, input, weights, configuration, bias=None):
    config_rule = configuration.type

    handler = None
    if config_rule == RULE.EPSILON:
        handler = _linear_epsilon
    elif config_rule == RULE.ALPHA_BETA:
        handler = _linear_alpha
    elif config_rule == RULE.FLAT:
        handler = _linear_flat
    elif config_rule == RULE.WW:
        handler = _linear_ww

    return handler(R, input, weights, configuration, bias)


def linear(router, R):
    """
    linear lrp
    :param router: the router object to report changes to
    :param R: the list of tensors containing the relevances from the upper layers
    """
    # Tensor is the output of the current operation (i.e. Add, or MatMul)
    tensor = router.get_current_operation().outputs[0]
    # Sum the potentially multiple relevances from the upper layers
    R = lrp_util.sum_relevances(R)

    # Start by assuming the activation tensor is the output of a matrix multiplication
    # (i.e. not an addition with a bias)
    # Tensor shape: (1, upper layer size)
    matmultensor = tensor
    bias = None

    # If the activation tensor is the output of an addition (i.e. the above assumption does not hold),
    # move through the graph to find the output of the nearest matrix multiplication.
    if tensor.op.type == 'Add':
        bias = lrp_util.get_input_bias_from_add(tensor)
        matmultensor = lrp_util.find_first_tensor_from_type(tensor, 'MatMul')

    # Find the inputs to the matrix multiplication
    (input, weights) = matmultensor.op.inputs

    layer_config = router.get_configuration(LAYER.LINEAR)
    R_new = linear_with_config(R, input, weights, layer_config, bias)
    # if layer_config.type == ALPHA_BETA_RULE:
    #     # Calculate new relevances with the alpha rule
    #     R_new = linear_alpha(R, input, weights, bias=bias, alpha=layer_config.alpha, beta=layer_config.beta)
    # elif layer_config.type == EPSILON_RULE:
    #     # Calculate new relevances with the epsilon rule
    #     R_new = linear_epsilon(R, input, weights, bias=bias, epsilon=layer_config.epsilon)


    # Mark handled operations
    router.mark_operation_handled(tensor.op)
    router.mark_operation_handled(matmultensor.op)

    # Forward relevance
    router.forward_relevance_to_operation(R_new, matmultensor.op, input.op)


def _sparse_flat(config, zijs, bias):
    # TODO should we implement these? They will end up being huge, since all parts of the input (also zeros) will
    # get relevance
    raise NotImplementedError("Flat relevance distribution is not implemented for sparse matrix multiplications")


def _sparse_ww(config, zijs, bias):
    # TODO should we implement these? They will end up being huge, since all parts of the input (also zeros) will
    # get relevance
    raise NotImplementedError("WW relevance distribution is not implemented for sparse matrix multiplications")


def _sparse_distribute_bias(config, zijs, bias):

    # Return if bias or the bias strategy is none or throw error if it is all
    # since all will kill the memory (remember we are in sparse land here ;) )
    if bias is None:
        return zijs
    if config.bias_strategy == BIAS_STRATEGY.NONE:
        return zijs
    elif config.bias_strategy == BIAS_STRATEGY.ALL:
        raise NotImplementedError("BIAS_STRATEGY.ALL is not implemented for sparse matmul")

    # From here on we assume that the BIAS_STRATEGY is BIAS_STRATEGY.ACTIVE
    # Strategy for the rest of this function is to
    # 1) Count how many active zijs (zij != 0) there are for each column (output_width dimension)
    # 2) Use the counts to distribute bias equaly among the active zijs

    # Squeeze bias to only have one dimension and shape: (output_width,)
    bias = tf.squeeze(bias)

    # zijs dense_shape: (batch_size, input_width, output_width)
    # Column_cnt is the the output_width
    column_cnt = tf.cast(zijs.dense_shape[2], tf.int32)

    # Prepare tensor array to hold the counts. It will hold both count
    # and number of elements considered in current column. The reason
    # for this is that there might be values in the values array that
    # are 0 meaning that they are not active even though they are represented
    # in the sparse matrix.
    active_counts = tf.TensorArray(tf.int32, column_cnt,
                                   clear_after_read=False,
                                   dynamic_size=True)

    # Number of values and associated indices to consider
    number_of_values = tf.size(zijs.values)

    # Transpose zijs in order to be able to scan through columns
    # zijs shape before: (batch_size, input_width, output_width)
    # zijs shape after: (batch_size, output_width, input_width)
    zijs = tf.sparse_transpose(zijs, [0, 2, 1])

    # Each element in indices_ta is shape (3,) (indicating batch_idx, output_width_idx, and input_width_idx)
    indices_ta = tf.TensorArray(tf.int64, number_of_values, clear_after_read=False).unstack(zijs.indices)

    # Each element in values_ta holds the value associated with the index in indices_ta
    values_ta = tf.TensorArray(tf.float32, number_of_values, clear_after_read=False).unstack(zijs.values)

    # Small helper function used to extact information about a single value and its associated index
    # at index `t` in indices_ta and values_ta
    def _get_value(t):
        # Get the position for the current value
        value_position = tf.cast(indices_ta.read(t), tf.int32)
        # Return a tuple containing (batch_index, column_index and value) for index `t`
        return value_position[0], value_position[1], values_ta.read(t)

    # Extract first position for feeding into the count while_loop
    first_batch, first_col, _ = _get_value(0)

    # Loop over every value in sparse tensor zijs and count how many values
    # is in each column of zijs
    def _count_loop(value_idx, actives, actives_idx, batch_idx, col_idx, cnt, considered):

        # Get the position and value for the current value_idx
        value_batch_idx, value_col_idx, value_value = _get_value(value_idx)

        # Helper function that adds one to `prev_cnt` only if `value_value` is not 0
        def _get_new_cnt(prev_cnt):
            return tf.cond(tf.not_equal(value_value, 0),
                              true_fn=lambda: prev_cnt + 1,
                              false_fn=lambda: prev_cnt)

        # Helper function that adds one to considered and counts active if `value_value` is not 0
        # This function is called whenever we see an element from the same column and batch of zij
        # as the previous element we considered in the while loop.
        def _increase_cnt():
            return actives, actives_idx, batch_idx, col_idx, _get_new_cnt(cnt), considered + 1

        # Helper function that records current count, resets counting variable, and updates column variable
        # This function is called whenever we see an element from the same batch but another
        # column of zij
        def _step_column():
            new_actives = actives.write(actives_idx, [cnt, considered])
            return new_actives, actives_idx + 1, batch_idx, value_col_idx, _get_new_cnt(0), 1

        # Helper function that records current count, resets counting variable, and updates column and batch variable
        # This function is called whenever we see an element from a new batch of zij
        def _step_batch():
            new_actives = actives.write(actives_idx, [cnt, considered])
            return new_actives, actives_idx + 1, value_batch_idx, value_col_idx, _get_new_cnt(0), 1

        actives, actives_idx, batch_idx, col_idx, cnt, considered = tf.cond(
            tf.equal(batch_idx, value_batch_idx),
            true_fn=lambda: tf.cond(
                tf.equal(col_idx, value_col_idx),
                true_fn=_increase_cnt,
                false_fn=_step_column
            ),
            false_fn=_step_batch
        )

        # Go to next value in the values_ta
        return value_idx + 1, actives, actives_idx, batch_idx, col_idx, cnt, considered

    _, active_counts, actives_count_idx, _, _, cnt, considered = tf.while_loop(
        cond=lambda batch_idx, *args: batch_idx < number_of_values,
        body=_count_loop,
        loop_vars=[0, active_counts, 0, first_batch, first_col, 0, 0]
    )

    # Store the last count in the active_counts tensor array since this will
    # not be recorded in the while loop.
    active_counts = active_counts.write(actives_count_idx, [cnt, considered])

    # Prepare bias by splitting it in to a tensor array
    bias_ta = tf.TensorArray(tf.float32, tf.size(bias), clear_after_read=False).unstack(bias)

    # Prepare tensor array to record new zijs
    new_values = tf.TensorArray(tf.float32, number_of_values)

    # The loop body to distribute the bias among the active zijs
    def _distribute_loop(value_idx, new_values, actives_idx, used):
        # Get info about current sample
        value_batch_idx, value_col_idx, value_value = _get_value(value_idx)

        # Get the active neuron count to distribute bias among and the number of elements
        # considered in the current column of zij
        count_and_considered = active_counts.read(actives_idx)
        active_count = count_and_considered[0]
        considered = count_and_considered[1]

        # Helper function to be called whenever we encounter an active zij
        # It adds the proper fraction of the associated bias to the current value
        def _distribute_bias():
            # TODO we could avoid dividing bias with `active_count` for every single value
            # Calculate bias for current value by dividing bias with the active count
            bias_for_value = bias_ta.read(value_col_idx) / tf.cast(active_count, tf.float32)
            # Caclulate new value by adding the right amount of bias
            new_value = value_value + bias_for_value

            return new_value

        # Helper function that leaves the value untouched
        # This function is called whenever `value_value` is 0
        def _not_active():
            return value_value

        new_value_value = tf.cond(
            tf.equal(value_value, 0),
            true_fn=_not_active,
            false_fn=_distribute_bias
        )

        # Remember that we have used another one of the considered values of the current column
        used += 1

        # Store the new value
        new_values = new_values.write(value_idx, new_value_value)

        # Updated index for active count if we have used all the considered values
        actives_idx, used = tf.cond(tf.equal(used, considered),
                                    true_fn=lambda: (actives_idx + 1, 0),
                                    false_fn=lambda: (actives_idx, used))

        # Go to next value
        return value_idx + 1, new_values, actives_idx, used

    _, new_values, *_ = tf.while_loop(
        cond=lambda t, *_: t < number_of_values,
        body=_distribute_loop,
        loop_vars=[0, new_values, 0, 0]
    )

    # Stack tensor array to tensor
    # Shape: (number_of_values,)
    new_values = new_values.stack()

    # Create new Sparse Tensor with same indices and shape but the new values
    zijs_new = tf.SparseTensor(zijs.indices, new_values, zijs.dense_shape)

    # Transpose back to shape (batch_size, input_width, output_width)
    zijs_new = tf.sparse_transpose(zijs_new, [0, 2, 1])

    return zijs_new


def _sparse_epsilon(config, Rs, predictions_per_sample, zijs, bias):
    # Zj has shape (batch_size, 1, output_width) dense tensor
    zj = tf.sparse_reduce_sum(zijs, 1, keep_dims=True)

    # Prepare sparse tensor with duplicated bias for addition with zj
    if bias is not None:
        zj = zj + bias + config.epsilon

    # Distribute bias according to config
    zijs = _sparse_distribute_bias(config, zijs, bias)

    # construct bias to add to zj
    fractions = zijs / zj

    # Distribute the relevance according to the fractions
    R_new = _sparse_distribute_relevances(Rs, zijs.dense_shape[0], zijs.dense_shape[1], predictions_per_sample,
                                          fractions)

    return R_new


def _sparse_alpha(config, Rs, predictions_per_sample, zijs, bias):
    def _selective_Rs(selection):
        selection_values = selection(zijs.values)
        zijs_selection = tf.SparseTensor(zijs.indices, selection_values, zijs.dense_shape)

        # Sum over the input dimension to get the Zjs
        zj = tf.sparse_reduce_sum(zijs_selection, 1, keep_dims=True)
        b = bias
        # If there is actually
        if b is not None:
            # Filter bias
            b = selection(b)
            zj = zj + b

        # Add stabilizer
        zj = zj + EPSILON

        # Distribute bias according to the current configuration
        zijs_selection = _sparse_distribute_bias(config, zijs_selection, b)

        # Shape (batch_size, in_size, out_size)
        fractions = zijs_selection / zj

        # Distribute the relevance according to the fractions and return
        return _sparse_distribute_relevances(Rs, zijs.dense_shape[0], zijs.dense_shape[1],
                                             predictions_per_sample, fractions)

    # Scale the positive relevances by the alpha value of the configuration
    R_positive = _selective_Rs(lrp_util.replace_negatives_with_zeros) * config.alpha

    # Scale the negative relevances by the beta value of the configuration
    R_negative = _selective_Rs(lrp_util.replace_positives_with_zeros) * config.beta

    # Return the sum of the positive and the negative relevances
    return tf.sparse_add(R_positive, R_negative)


def sparse_dense_linear(router, R):
    # Sum the potentially multiple relevances from the upper layers
    R = lrp_util.sum_relevances(R)
    current_operation = router.get_current_operation()

    # As default set current_operation as matmul_tensor
    matmul_operation = current_operation

    # Start by assuming the activation tensor is the output of a sparse_tensor_dense_matmul
    # (i.e. not an addition with a bias)
    bias = None

    # If the activation tensor is the output of an addition (i.e. the above assumption does not hold),
    # move through the graph to find the output of the nearest matrix multiplication.
    if current_operation.type == 'Add':
        bias = lrp_util.get_input_bias_from_add(current_operation.outputs[0])
        matmul_tensor = lrp_util.find_first_tensor_from_type(current_operation.outputs[0], 'SparseTensorDenseMatMul')
        matmul_operation = matmul_tensor.op

    # Extract tensors from input to the sparse matmul operation
    (sparse_indices, sparse_values, sparse_shape, dense_input_weights) = matmul_operation.inputs

    # Construct sparse tensor from the inputs
    sparse_input_tensor = tf.SparseTensor(sparse_indices, sparse_values, sparse_shape)

    # Find tensors holding the different dimensionalities of input and output tensors
    R_shape = tf.shape(R)
    batch_size = tf.cast(R_shape[0], tf.int64)
    predictions_per_sample = tf.cast(R_shape[1], tf.int64)
    out_size = R_shape[2]

    # We know that sparse tensor to sparse_tensor_dense_matmul must be two dimensional
    in_size = sparse_input_tensor.dense_shape[1]

    # Create tensor array for the weights. This is used in the following while_loop.
    ta = tf.TensorArray(tf.float32, out_size)

    # Unstack the columns of the weights to be able to "manually" broadcast them
    # over the input
    # each element in ta has shape: (input_width,)
    ta = ta.unstack(tf.transpose(dense_input_weights))

    # Create tensor array to hold all Zij's values calculated in the following while_loop
    all_values = tf.TensorArray(tf.float32, 1, dynamic_size=True)

    # Create tensor array to hold all indices of the Zij's calculated in the following while_loop
    all_indices = tf.TensorArray(tf.int64, 1, dynamic_size=True)

    def _zij_loop(t, av, ai, ta, offset):
        # Broad cast one column of the weights through the input tensor (over the batch_size dimension)
        # tmp_sparse shape: (batch_size, input_width)
        tmp_sparse = sparse_input_tensor * ta.read(t)

        # Count how many elements we are about to add to the tensor arrays
        value_cnt = tf.shape(tmp_sparse.values)[0]

        # Calculate all the indexes to scatter the calculated values and indices into in the tensor arrays
        scatter_range = tf.range(offset, offset + value_cnt, dtype=tf.int32)

        # Scatter the values as is into the av ('all_values') array
        av = av.scatter(scatter_range, tmp_sparse.values)

        # Append the current column of the weights to the indices of the results to be able to
        # get the shape (batch_size, input_width, output_width) when creating the sparse tensor
        # that will later hold the fractions
        new_indices = tf.pad(tmp_sparse.indices, [[0, 0], [0, 1]], constant_values=tf.cast(t, dtype=tf.int64))

        # Scatter the indices like we did with with the values above
        ai = ai.scatter(scatter_range, new_indices)

        # Go to next column in the weights
        return t + 1, av, ai, ta, offset + value_cnt

    _, all_values, all_indices, *_ = tf.while_loop(
        cond=lambda t, *_: t < out_size,
        body=_zij_loop,
        loop_vars=[0, all_values, all_indices, ta, 0])

    # Stack all the values into one list; shape: (values_length,)
    all_values = all_values.stack()

    # Stack all the indices into one list; shape: (values_length, 3)
    all_indices = tf.cast(all_indices.stack(), dtype=tf.int64)

    # Create new sparse tensor holding all Zij's across all outpus
    # Zijs shape: (batch_size, in_size, out_size)
    in_size = tf.cast(in_size, tf.int64)
    out_size = tf.cast(out_size, tf.int64)

    # Construct a sparse tensor of the zijs
    zijs = tf.SparseTensor(all_indices, all_values, (batch_size, in_size, out_size))

    # Make sure that the zijs are ordered properly since it is expected by `_sparse_distribute_bias`
    zijs = tf.sparse_reorder(zijs)

    # Move the predictions_per_sample dimension up to be able to unstack it
    R = tf.transpose(R, [1, 0, 2])

    # Cast predictions per sample to int32
    predictions_per_sample = tf.cast(predictions_per_sample, tf.int32)

    # Unstack R to get a tensor array containing elements of shape (batch_size, out_size)
    Rs = tf.TensorArray(tf.float32, predictions_per_sample, clear_after_read=False).unstack(R)

    # Get the configuration for sparse linear from the router
    config = router.get_configuration(LAYER.SPARSE_LINEAR)

    # Extract the config type
    config_rule = config.type

    handler = None
    # Find rule to distribute relevance by
    if config_rule == RULE.EPSILON:
        handler = _sparse_epsilon
    elif config_rule == RULE.ALPHA_BETA:
        handler = _sparse_alpha
    elif config_rule == RULE.FLAT:
        handler = _sparse_flat
    elif config_rule == RULE.WW:
        handler = _sparse_ww

    # Distribute the relevance with the appropriate handler
    R_new = handler(config, Rs, predictions_per_sample, zijs, bias)

    # Transpose R_new to get the proper shape
    # R_new shape after transpose: (batch_size, predictions_per_sample, in_width)
    R_new = tf.sparse_transpose(R_new, [1, 0, 2])

    # Mark operation as handled
    router.mark_operation_handled(current_operation)
    router.mark_operation_handled(matmul_operation)

    # Find unique operations to send relevance to by making a
    # small dictionary with operation._id as keys.
    relevance_destinations = {}
    for inp in matmul_operation.inputs:
        relevance_destinations[inp.op._id] = inp.op

    # Extract the actual operations as destinations
    destinations = relevance_destinations.values()

    # Forward new Relevance to the proper operation
    for op in destinations:
        router.forward_relevance_to_operation(R_new, matmul_operation, op)


def _sparse_distribute_relevances(Rs, batch_size, in_size, predictions_per_sample, fractions):
    # Prepare tensor arrays for holding values and indices for calculated relevances
    all_values = tf.TensorArray(tf.float32, predictions_per_sample, dynamic_size=True)
    all_indices = tf.TensorArray(tf.int64, predictions_per_sample, dynamic_size=True)

    def _prediction_loop(t, av, ai, Rs, offset):
        # current_R shape: (batch_size, out_size)
        current_R = Rs.read(t)

        # Prepare batch for current prediction_per_sample to be broadcasted over
        # the input dimension
        # current_R shape: (batch_size, 1, out_size)
        current_R = tf.expand_dims(current_R, 1)

        # Scale fractions with relevances for current prediction_per_sample
        distributed_relevances = fractions * current_R

        # Reduce sum the get the relevances for the individual in_dimensions
        # new_relevances shape: (batch_size, in_size)
        new_relevances = tf.sparse_reduce_sum_sparse(distributed_relevances, 2)

        # Count how many values and indices to add to the tensor arrays
        value_cnt = tf.shape(new_relevances.values)[0]
        # Calculate range of indexes in the tensor arrays to write the values and indices to
        scatter_range = tf.range(offset, offset + value_cnt, dtype=tf.int32)

        # Scatter the values of the new relevances
        av = av.scatter(scatter_range, new_relevances.values)

        # Prepend the prediction_per_sample dimension to be able to make a
        # sparse tensor of shape (predictions_per_sample, batch_size, in_width) after
        # the while loop
        new_indices = tf.pad(new_relevances.indices, [[0, 0], [1, 0]], constant_values=tf.cast(t, dtype=tf.int64))

        # Scatter the indices of the new relevances
        ai = ai.scatter(scatter_range, new_indices)

        # Go to next prediction_per_sample
        return t + 1, av, ai, Rs, offset + value_cnt

    _, all_values, all_indices, *_ = tf.while_loop(
        cond=lambda t, *_: t < predictions_per_sample,
        body=_prediction_loop,
        loop_vars=[0, all_values, all_indices, Rs, 0]
    )
    # Stack the values in the all_values tensor array to get
    # R_values shape: (value_length,)
    R_values = all_values.stack()
    # Stack the indices in the all_indices tensor array to get
    # R_indices shape: (value_length, 3)
    R_indices = tf.cast(all_indices.stack(), dtype=tf.int64)
    # Create sparse tensor for R_new
    # R_new shape: (predictions_per_sample, batch_size, in_width)
    predictions_per_sample = tf.cast(predictions_per_sample, tf.int64)
    R_new = tf.SparseTensor(R_indices, R_values, (predictions_per_sample, batch_size, in_size))
    return R_new
