import tensorflow as tf
import lrp_util

def shaping(router, R):
    # Sum the potentially multiple relevances from the upper layers
    R = lrp_util.sum_relevances(R)

    # Reshape R to the same shape as the input to the
    # reshaping operation that created the tensor
    current_operation = router.get_current_operation()
    input_to_current_operation = current_operation.inputs[0]
    R_reshaped = tf.reshape(R, tf.shape(input_to_current_operation))

    # Check if there has been added an extra dimension (for multiple predictions per sample) in
    # which case we have to add the dimension again after the reshape
    if router.starting_point_relevances_did_not_have_predictions_per_sample_dimension():
        R_reshaped = tf.expand_dims(R_reshaped, 1)

    # Tell the router that we handled this operation
    router.mark_operation_handled(current_operation)

    # Forward relevance to the operation of the input to the current operation
    router.forward_relevance_to_operation(R_reshaped, current_operation, input_to_current_operation.op)
