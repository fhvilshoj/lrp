import tensorflow as tf
import lrp.lrp_util as lrp_util

def shaping(router, R):
    # Sum the potentially multiple relevances from the upper layers
    R = lrp_util.sum_relevances(R)

    # Get the current operation (i.e. the shaping operation we are currently taking care of)
    current_operation = router.get_current_operation()

    # Get the input to the shaping operation
    input_to_current_operation = current_operation.inputs[0]

    # Get the shape of the input to the shaping operation
    input_shape = tf.shape(input_to_current_operation)

    # Split the shape of the input into 'batch_size' and everything else
    batch_size, input_shape_without_batch_size = tf.split(input_shape, [1, -1], 0)

    # Get the shape of the relevances
    relevances_shape = tf.shape(R)

    # Find the size of the predictions_per_sample dimension
    predictions_per_sample = relevances_shape[1]

    # Concatenate the dimensions to get the new shape of the relevances
    relevances_new_shape = tf.concat([batch_size, [predictions_per_sample], input_shape_without_batch_size], 0)

    # Reshape R to the same shape as the input to the reshaping operation that created the tensor
    # but leave the two first dimensions untouched since they are batch_size, predictions_per_sample
    # which stay constant all the way through the framework
    R_reshaped = tf.reshape(R, relevances_new_shape)

    # Tell the router that we handled this operation
    router.mark_operation_handled(current_operation)

    # Forward relevance to the operation of the input to the current operation
    router.forward_relevance_to_operation(R_reshaped, current_operation, input_to_current_operation.op)


