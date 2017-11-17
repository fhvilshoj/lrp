import lrp.lrp_util
import tensorflow as tf


def tile(router, R):
    # Sum the relevances received from upper layers
    R = lrp_util.sum_relevances(R)

    # Get the current operation, i.e. the tile operation we are currently handling
    current_operation = router.get_current_operation()

    # Get the input to the tiling operation and find the shape of it
    input_to_current_operation = current_operation.inputs[0]
    input_shape = tf.shape(input_to_current_operation)

    # Get the size of the 'predictions_per_sample' dimension from R
    R_shape = tf.shape(R)
    predictions_per_sample = R_shape[1]
    # Reshape to a list so it can be used in the concat below
    predictions_per_sample = tf.reshape(predictions_per_sample, (1,))

    # Get the tensor that tells how many times the input has been duplicated for each dimension of the input
    copies_per_dimension = current_operation.inputs[1]

    # Get the number of dimensions of the input
    rank_input = tf.size(copies_per_dimension)

    # Transpose R from shape (batch_size, predictions_per_sample, ....) to shape
    # (predictions_per_sample, batch_size, ...) since the predictions_per_sample dimensions is left untouched in
    # the processing below
    remaining_axes = tf.range(2, rank_input + 1)
    perm = tf.concat([[1, 0], remaining_axes], 0)
    R = tf.transpose(R, perm)

    # Reshape R to shape (copies_dim_0, input_size_dim_0, ... copies_dim_(r-1), input_size_dim(r-1))
    double_rank = rank_input * 2
    zipped_dims = tf.reshape(tf.transpose([copies_per_dimension, input_shape]), (double_rank,))
    zipped_dims = tf.concat([predictions_per_sample, zipped_dims], 0)
    R = tf.reshape(R, zipped_dims)

    # Transpose R to shape (input_size_dim_0, copies_dim_0 ... input_size_dim(r-1), copies_dim_(r-1))
    perm1 = tf.range(2, double_rank + 1, 2)
    perm2 = tf.range(1, double_rank + 1, 2)
    zipped_perm = tf.reshape(tf.transpose([perm1, perm2]), (double_rank,))
    zipped_perm = tf.concat([[0], zipped_perm], 0)
    R = tf.transpose(R, zipped_perm)

    # Reduce sum for R over dimensions 'input_size_dim_0', ... 'input_size_dim(r-1)'
    R_new = tf.reduce_sum(R, perm1)

    # Transpose R back from shape (predictions_per_sample, batch_size ....) to shape
    # (batch_size, predictions_per_sample, ...)
    remaining_axes = tf.range(2, rank_input + 1)
    perm = tf.concat([[1,0], remaining_axes], 0)
    R_new = tf.transpose(R_new, perm)

    # Mark the tiling operation as handled
    router.mark_operation_handled(current_operation)

    # Forward the relevance to input to the tile operation
    router.forward_relevance_to_operation(R_new, current_operation, input_to_current_operation.op)
