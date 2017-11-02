import tensorflow as tf
from lrp import lrp_util
from lrp.configuration import LINEAR_LAYER, ALPHA_BETA_RULE, EPSILON_RULE
from constants import BIAS_DELTA, EPSILON


def linear_epsilon(R, input, weights, bias=None, output=None):
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
    # If no output tensor given; construct one by multiplying input and weights
    if output is None:
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
        # Number of input features to divide relevance among (cast to float32 from int to perform the division below)
        input_shape = tf.shape(input)
        input_features = tf.cast(input_shape[1], tf.float32)

        # Find the bias to divide equally among the rows (This includes the stabilizer: epsilon)
        # Shape: (output_width) or (batch, output_width)
        bias_to_divide = (BIAS_DELTA * bias + EPSILON * tf.sign(output))

        # Divide the bias (and stabilizer) equally between the `input_features` (rows of zs)
        # Shape: (output_width) or (batch, output_width)
        bias_per_feature = bias_to_divide / input_features

        # Expand the second to last dimension to be able to add the bias through the rows of zs
        # Shape: (1, output_width) or (batch, 1, output_width)
        bias_per_feature = tf.expand_dims(bias_per_feature, -2)

        # Add bias to zs
        # Shape of zs: (batch, input_width, output_width)
        zs = zs + bias_per_feature

    # Add stabilizer to denominator to avoid dividing with 0
    # Shape of denominator: (batch, output_width)
    denominator = output + EPSILON * tf.sign(output)

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


# TODO Should this function also take output as an optional input to be consistent with other rules?
def linear_alpha(R, input, weights, bias=None):
    # Prepare batch for elementwise multiplication
    # Shape of input: (batch_size, input_width)
    # Shape of input after expand_dims: (batch_size, input_width, 1)
    input = tf.expand_dims(input, -1)

    # Perform elementwise multiplication of input, weights to get z_kij which is the contribution from
    # feature i to neuron j for input k
    # Shape of zs: (batch_size, input_width, output_width)
    zs = tf.multiply(input, weights)

    # Replace the negative elements with zeroes to only have the positive z's left (i.e. z_kij^+)
    # Shape of zp: (batch_size, input_width, output_width)
    zp = lrp_util.replace_negatives_with_zeros(zs)

    # Take the sum of each column of z_kij^+'s
    # Shape of zp_sum: (batch_size, 1, output_width)
    zp_sum = tf.reduce_sum(zp, axis=1, keep_dims=True)

    # Find and add the positive parts of an eventual bias (i.e. find and add the b^+s).
    if bias is not None:
        # Replace the negative elements in the bias with zeroes
        bias_positive = lrp_util.replace_negatives_with_zeros(bias)

        # Add the sum of the z_ij^+'s and the positive bias (i.e. find the z_j^+'s)
        zp_sum = tf.add(zp_sum, bias_positive)

    # Add stabilizer to the denominator
    zp_sum += EPSILON

    # Find the relative contribution from feature i to neuron j for input k
    # Shape of fractions: (batch_size, input_width, output_width)
    fractions = tf.divide(zp, zp_sum)

    # Prepare the fractions for the matmul below
    # Shape of fractions after transpose: (batch_size, output_width, input_width)
    fractions = tf.transpose(fractions, perm=[0, 2, 1])

    # Multiply relevances with fractions to find relevance per feature in input
    # In other words: Calculate the lower layer relevances (a combination of equation 60 and 62 in Bach 2015)
    # Shape of R: (batch_size, predictions_per_sample, output_width)
    # Shape of fractions: (batch_size, output_width, input_width)
    # Shape of R_new: (batch_size, predictions_per_sample, input_width)
    R_new = tf.matmul(R, fractions)

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
        bias = lrp_util._get_input_bias_from_add(current_tensor)
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
    new_input, R = tf.cond(tf.equal(tf.rank(input),  2),
                       true_fn=_rank2,
                       false_fn=_higher_rank)

    layer_type = router.get_configuration(LINEAR_LAYER).type
    if layer_type == ALPHA_BETA_RULE:
        # Calculate new relevances with the alpha rule
        R_new = linear_alpha(R, input, weights, bias=bias)
    elif layer_type == EPSILON_RULE:
        # Calculate new relevances with the epsilon rule
        R_new = linear_epsilon(R, input, weights, bias=bias)

    # Turn the calculated relevances into the correct form if the rank of the input was > 2
    def _revert_rank2():
        return R_new

    def _revert_higher_rank():
        return tf.reshape(R_new, tf.concat(([batch_size], [predictions_per_sample], tf.shape(input)[1:]), 0))

    # Test if the rank of the input is > 2
    R_new = tf.cond(tf.equal(tf.rank(input),  2),
                       true_fn=_revert_rank2,
                       false_fn=_revert_higher_rank)

    # Mark handled operations
    router.mark_operation_handled(current_operation)
    router.mark_operation_handled(multensor.op)

    # Forward relevance
    router.forward_relevance_to_operation(R_new, multensor.op, input.op)


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
        bias = lrp_util._get_input_bias_from_add(tensor)
        matmultensor = lrp_util.find_first_tensor_from_type(tensor, 'MatMul')

    # Find the inputs to the matrix multiplication
    (input, weights) = matmultensor.op.inputs

    layer_type = router.get_configuration(LINEAR_LAYER).type
    if layer_type == ALPHA_BETA_RULE:
        # Calculate new relevances with the alpha rule
        R_new = linear_alpha(R, input, weights, bias=bias)
    elif layer_type == EPSILON_RULE:
        # Calculate new relevances with the epsilon rule
        R_new = linear_epsilon(R, input, weights, bias=bias)

    # Mark handled operations
    router.mark_operation_handled(tensor.op)
    router.mark_operation_handled(matmultensor.op)

    # Forward relevance
    router.forward_relevance_to_operation(R_new, matmultensor.op, input.op)


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
        bias = lrp_util._get_input_bias_from_add(current_operation.outputs[0])
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
    ta = ta.unstack(tf.transpose(dense_input_weights))

    # Create tensor array to hold all Zij's values calculated in the following while_loop
    all_values = tf.TensorArray(tf.float32, 1, dynamic_size=True)

    # Create tensor array to hold all indices of the Zij's calculated in the following while_loop
    all_indices = tf.TensorArray(tf.int64, 1, dynamic_size=True)

    def _loop(t, av, ai, ta, offset):
        # Broad cast one column of the weights through the input tensor (over the batch_size dimension)
        # tmp_sparse shape: (batch_size, input_width, output_width)
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
        body=_loop,
        loop_vars=[0, all_values, all_indices, ta, 0])

    # Stack all the values into one list; shape: (values_length,)
    all_values = all_values.stack()

    # Stack all the indices into one list; shape: (values_length, 3)
    all_indices = tf.cast(all_indices.stack(), dtype=tf.int64)

    # Create new sparse tensor holding all Zij's across all outpus
    # Zijs shape: (batch_size, in_size, out_size)
    in_size = tf.cast(in_size, tf.int64)
    out_size = tf.cast(out_size, tf.int64)
    positive_values = lrp_util.replace_negatives_with_zeros(all_values)
    zijs = tf.SparseTensor(all_indices, positive_values, (batch_size, in_size, out_size))
    # in_size = tf.cast(in_size, tf.int64)
    # out_size = tf.cast(out_size, tf.int64)

    # Sum over the input dimension get the Zjs
    zj = tf.sparse_reduce_sum(zijs, 1, keep_dims=True)

    # Add positive bias is there is any
    if bias is not None:
        zj = zj + lrp_util.replace_negatives_with_zeros(bias)

    # Add stabilizer
    zj = zj + EPSILON

    # Shape (batch_size, in_size, out_size)
    fractions = zijs / zj

    # Move the predictions_per_sample dimension up to be able to unstack it
    R = tf.transpose(R, [1, 0, 2])

    predictions_per_sample = tf.cast(predictions_per_sample, tf.int32)

    # Unstack R to get a tensor array containing elements of shape (batch_size, out_size)
    Rs = tf.TensorArray(tf.float32, predictions_per_sample).unstack(R)

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
    R_values = tf.Print(R_values, [R_values], message="\n\nR_values:\n", summarize=100)
    # Stack the indices in the all_indices tensor array to get
    # R_indices shape: (value_length, 3)
    R_indices = tf.cast(all_indices.stack(), dtype=tf.int64)

    # Create sparse tensor for R_new
    # R_new shape: (predictions_per_sample, batch_size, in_width)
    predictions_per_sample = tf.cast(predictions_per_sample, tf.int64)
    R_new = tf.SparseTensor(R_indices, R_values, (predictions_per_sample, batch_size, in_size))

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
        print("Writing relevance to ", op.type, " - ", op.name)
        router.forward_relevance_to_operation(R_new, current_operation, op)
