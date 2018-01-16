import tensorflow as tf
from . import lrp_util
from .configuration import LAYER, RULE, BIAS_STRATEGY
from .constants import BIAS_DELTA, EPSILON


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
    Epsilon rule implementation
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
    if bias is not None and config.bias_strategy != BIAS_STRATEGY.IGNORE:
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

    output_sign = tf.sign(output)
    output_sign = tf.where(tf.equal(output_sign, 0), tf.ones_like(output_sign), output_sign)

    if bias is not None and config.bias_strategy != BIAS_STRATEGY.IGNORE:
        # Find the bias to divide among the rows (This includes the stabilizer: epsilon)
        # Shape: (output_width) or (batch, output_width)
        bias_to_divide = (BIAS_DELTA * bias + config.epsilon * output_sign)

        zs = _divide_bias_among_zs(config, zs, bias_to_divide)

    # Add stabilizer to denominator to avoid dividing with 0
    # Shape of denominator: (batch, output_width)
    denominator = output + config.epsilon * output_sign

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


def _linear_zb(R, input, weights, config, bias=None):
    """
    Implementation of Zb rule
    :param R: tensor of relevance to distribute. Shape: (batch_size, output_width)
    :param input: The input tensor. Shape: (batch_size, input_width)
    :param weights: The kernel weights. Shape: (input_width, output_width)
    :param output: Optional output tensor. Shape: (batch_size, output_width)
           If none output is calculated as input times weights plus bias.
    :param bias: Optional tensor with bias. Shape: (output_width) or (batch_size, output_width)
    :return: Redistributed relevance. Shape: (batch_size, input_width)
    """
    # Shape (batch_size, input_width, output_width)
    positive_weights = lrp_util.replace_negatives_with_zeros(weights) * config.low
    negative_weights = lrp_util.replace_positives_with_zeros(weights) * config.high

    # Prepare batch for elementwise multiplication
    # Shape of input: (batch_size, input_width)
    # Shape of input after expand_dims: (batch_size, input_width, 1)
    input = tf.expand_dims(input, -1)

    # Perform elementwise multiplication of input, weights to get z_kij which is the contribution from
    # feature i to neuron j for input k
    # Shape of zs: (batch_size, input_width, output_width)
    zs = tf.multiply(input, weights)

    # Shape of zs_lwp_hwn: (batch_size, input_width, output_width)
    zs_lwp_hwn = zs - positive_weights - negative_weights

    # Shape of output: (batch_size, 1, output_width)
    output = tf.reduce_sum(zs_lwp_hwn, axis=1)

    # Only add bias when bias is not none
    if bias is not None and config.bias_strategy != BIAS_STRATEGY.IGNORE:
        output += bias

    # When bias is given divide it equally among the i's to avoid relevance loss
    output_sign = tf.sign(output)
    output_sign = tf.where(tf.equal(output_sign, 0), tf.ones_like(output_sign), output_sign)

    if bias is not None and config.bias_strategy != BIAS_STRATEGY.IGNORE:
        # Find the bias to divide among the rows (This includes the stabilizer: epsilon)
        # Shape: (output_width) or (batch, output_width)
        bias_to_divide = bias + output_sign * 1e-12
        zs_lwp_hwn = _divide_bias_among_zs(config, zs_lwp_hwn, bias_to_divide)

    # Add stabilizer to denominator to avoid dividing with 0
    # Shape of denominator: (batch, output_width)
    denominator = output + 1e-12 * output_sign

    # Expand the second to last dimension to be able to divide the denominator through the rows of zs
    # Shape after expand_dims: (batch, 1, output_width)
    denominator = tf.expand_dims(denominator, -2)

    # Find the relative contribution from feature i to neuron j for input k
    # Shape of fractions: (batch_size, input_width, output_width)
    fractions = tf.divide(zs_lwp_hwn, denominator)

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
        if bias is not None and config.bias_strategy != BIAS_STRATEGY.IGNORE:
            # Filter elements in bias to either positives of negatives according to selection callable
            bias_filtered = selection(bias)

            # Divide the bias according to the current configuration
            zijs = _divide_bias_among_zs(config, zijs, bias_filtered)

            # Add the sum of the z_ij^+'s and the positive bias (i.e. find the z_j^+'s)
            zj_sum = tf.add(zj_sum, bias_filtered)

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

    # If we should just pass relevance right through. Do so.
    layer_config = router.get_configuration(LAYER.ELEMENTWISE_LINEAR)
    if layer_config.type == RULE.IDENTITY:
        # Mark handled operations
        router.mark_operation_handled(current_operation)
        router.mark_operation_handled(multensor.op)

        # Forward relevance
        router.forward_relevance_to_operation(R, multensor.op, input.op)
        return

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

    R_new = linear_with_config(R, new_input, weights, layer_config, bias)

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
    elif config_rule == RULE.ZB:
        handler = _linear_zb

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

    # Mark handled operations
    router.mark_operation_handled(tensor.op)
    router.mark_operation_handled(matmultensor.op)

    # Forward relevance
    router.forward_relevance_to_operation(R_new, matmultensor.op, input.op)


def _sparse_flat(config, Rs, predictions_per_sample, zijs, bias):
    # TODO should we implement these? They will end up being huge, since all parts of the input (also zeros) will
    # get relevance
    raise NotImplementedError("Flat relevance distribution is not implemented for sparse matrix multiplications")


def _sparse_ww(config, Rs, predictions_per_sample, zijs, bias):
    # Zj has shape (batch_size, 1, output_width) dense tensor
    zj = tf.sparse_reduce_sum(zijs, 1, keep_dims=True)

    # Stabilizer
    zj_sign = tf.sign(zj)
    zj_sign = tf.where(tf.equal(zj, 0), tf.ones_like(zj_sign), zj_sign)
    zj += zj_sign * EPSILON

    # construct bias to add to zj
    fractions = zijs / zj

    # Distribute the relevance according to the fractions
    R_new = _sparse_distribute_relevances(Rs, zijs.dense_shape[0], zijs.dense_shape[1], predictions_per_sample,
                                          fractions)

    return R_new


def _sparse_distribute_bias(config, zijs, bias):
    # Return if bias or the bias strategy is none or throw error if it is all
    # since all will kill the memory (remember we are in sparse land here ;) )
    if bias is None or config.bias_strategy == BIAS_STRATEGY.IGNORE:
        return zijs
    if config.bias_strategy == BIAS_STRATEGY.NONE:
        return zijs
    elif config.bias_strategy == BIAS_STRATEGY.ALL:
        raise NotImplementedError("BIAS_STRATEGY.ALL is not implemented for sparse matmul")

    # Dense Shape (batch_size, input_width, output_width)
    indicators = tf.SparseTensor(zijs.indices,
                                 tf.where(tf.equal(zijs.values, 0), tf.zeros_like(zijs.values),
                                          tf.ones_like(zijs.values)),
                                 zijs.dense_shape)

    # Count all the indicators in each column
    # Shape: (batch_size, 1, output_width)
    counts = tf.sparse_reduce_sum(indicators, axis=1, keep_dims=True)

    # Hack to avoid dividing by zero (doesn't matter for the final result.
    counts = tf.where(tf.equal(counts, 0), tf.ones_like(counts), counts)

    # Divide the bias by broadcasting it to every sample in the batch
    bias_divided = bias / counts

    # Scale the indicators by the bias
    # Dense shape: (batch_size, input_width, output_width)
    bias_to_add = indicators * bias_divided

    # Create new zijs with the divided bias
    zijs_new = tf.SparseTensor(zijs.indices, zijs.values + bias_to_add.values, zijs.dense_shape)

    return zijs_new


def _sparse_epsilon(config, Rs, predictions_per_sample, zijs, bias):
    # Zj has shape (batch_size, 1, output_width) dense tensor
    zj = tf.sparse_reduce_sum(zijs, 1, keep_dims=True)

    # Prepare sparse tensor with duplicated bias for addition with zj
    if bias is not None and config.bias_strategy != BIAS_STRATEGY.IGNORE:
        zj = zj + bias

    zj_sign = tf.sign(zj)
    zj_sign = tf.where(tf.equal(zj, 0), tf.ones_like(zj_sign), zj_sign)
    zj += zj_sign * config.epsilon

    # Distribute bias according to config
    zijs = _sparse_distribute_bias(config, zijs, bias)

    # construct bias to add to zj
    fractions = zijs / zj

    # Distribute the relevance according to the fractions
    R_new = _sparse_distribute_relevances(Rs, zijs.dense_shape[0], zijs.dense_shape[1], predictions_per_sample,
                                          fractions)

    return R_new


def _sparse_alpha(config, Rs, predictions_per_sample, zijs, bias):
    def _selective_Rs(selection, stabilizer):
        selection_values = selection(zijs.values)
        zijs_selection = tf.SparseTensor(zijs.indices, selection_values, zijs.dense_shape)

        # Sum over the input dimension to get the Zjs
        zj = tf.sparse_reduce_sum(zijs_selection, 1, keep_dims=True)
        b = bias
        # If there is actually
        if b is not None and config.bias_strategy != BIAS_STRATEGY.IGNORE:
            # Filter bias
            b = selection(b)
            zj = zj + b

        # Add stabilizer
        zj = stabilizer(zj, EPSILON)

        # Distribute bias according to the current configuration
        zijs_selection = _sparse_distribute_bias(config, zijs_selection, b)

        # Shape (batch_size, in_size, out_size)
        fractions = zijs_selection / zj

        # Distribute the relevance according to the fractions and return
        return _sparse_distribute_relevances(Rs, zijs.dense_shape[0], zijs.dense_shape[1],
                                             predictions_per_sample, fractions)

    # Scale the positive relevances by the alpha value of the configuration
    R_positive = _selective_Rs(lrp_util.replace_negatives_with_zeros, tf.add) * config.alpha

    # Scale the negative relevances by the beta value of the configuration
    R_negative = _selective_Rs(lrp_util.replace_positives_with_zeros, tf.subtract) * config.beta

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

    # Get the configuration for sparse linear from the router
    config = router.get_configuration(LAYER.SPARSE_LINEAR)

    # Extract the config type
    config_rule = config.type

    if config_rule == RULE.WW:
        sparse_indicator = tf.SparseTensor(sparse_input_tensor.indices,
                                           tf.ones_like(sparse_input_tensor.values),
                                           sparse_input_tensor.dense_shape)
        squared_weights = tf.square(dense_input_weights)
        zijs = _sparse_calculate_zijs(batch_size, sparse_indicator, squared_weights, in_size, out_size)
    else:
        zijs = _sparse_calculate_zijs(batch_size, sparse_input_tensor, dense_input_weights, in_size, out_size)

    # Move the predictions_per_sample dimension up to be able to unstack it
    R = tf.transpose(R, [1, 0, 2])

    # Cast predictions per sample to int32
    predictions_per_sample = tf.cast(predictions_per_sample, tf.int32)

    # Unstack R to get a tensor array containing elements of shape (batch_size, out_size)
    Rs = tf.TensorArray(tf.float32, predictions_per_sample, clear_after_read=False).unstack(R)

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


def _sparse_calculate_zijs(batch_size, sparse_input_tensor, dense_input_weights, in_size, out_size):
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
    return zijs


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
