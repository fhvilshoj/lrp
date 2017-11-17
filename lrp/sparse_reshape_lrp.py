import tensorflow as tf
import lrp.lrp_util

def sparse_reshape(router, R):
    # Sum the potentially multiple relevances from the upper layers
    R = lrp_util.sum_relevances(R)

    # Get the current operation (i.e. the sparse reshape operation we are currently taking care of)
    current_operation = router.get_current_operation()

    # Get the shape of the input to the sparse reshape operation
    input_shape = current_operation.inputs[1]

    # Split the shape of the input into 'batch_size' and everything else
    batch_size, input_shape_without_batch_size = tf.split(input_shape, [1, -1], 0)

    # Get the shape of the relevances
    relevances_shape = tf.shape(R)

    # Find the size of the predictions_per_sample dimension
    predictions_per_sample = relevances_shape[1]

    # Cast 'predictions_per_sample' to int64 to be able to concatenate with the dimensions from the sparse
    # tensor which are int64
    predictions_per_sample = tf.cast(predictions_per_sample, tf.int64)

    # Concatenate the dimensions to get the new shape of the relevances
    relevances_new_shape = tf.concat([batch_size, [predictions_per_sample], input_shape_without_batch_size], 0)

    # Reshape R to the same shape as the input to the reshaping operation
    R_reshaped = tf.sparse_reshape(R, relevances_new_shape)

    # Tell the router that we handled this operation
    router.mark_operation_handled(current_operation)

    # Forward relevance to the operation of the input to the current operation
    for input in current_operation.inputs:
        router.forward_relevance_to_operation(R_reshaped, current_operation, input.op)
