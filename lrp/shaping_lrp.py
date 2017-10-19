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

    # Tell the router that we handled this operation
    router.mark_operation_handled(current_operation)

    # Forward relevance to the operation of the input to the current operation
    router.forward_relevance_to_operation(R_reshaped, current_operation, input_to_current_operation.op)
