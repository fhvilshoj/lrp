import lrp.lrp_util
import tensorflow as tf

# TODO: Big fat todo. We don't handle slicing with sizes of -1 at the moment
def slicing(router, R):
    # Sum the relevances
    R = lrp_util.sum_relevances(R)

    # Get the current operation
    current_operation = router.get_current_operation()

    # Get the input to the slicing operation
    input_to_slicing_operation = current_operation.inputs[0]

    # Cut off the batch size and maybe predictions pr. sample if exists from the tensor shapes
    # since we never want to pad the batch dimension
    free_dimensions = 2 - (tf.rank(R) - tf.rank(input_to_slicing_operation))
    def _relevant_dims(tensor):
        return tensor[free_dimensions:]

    # Get the shape of the input to the slicing operation
    input_shape = _relevant_dims(tf.shape(input_to_slicing_operation))

    # Find starting point of the slice operation
    starting_point = _relevant_dims(current_operation.inputs[1])

    # Find size of the slice operation
    size_of_slice = _relevant_dims(current_operation.inputs[2])

    # Find the number of zeros to insert after the relevances
    end_zeros = input_shape - (starting_point + size_of_slice)

    # For each axis, insert 'starting_point' zeros before the relevances and
    # 'size_of_split'-('starting_point' + 'size_of_split') zeros after the relevances
    padding = tf.transpose(tf.stack([starting_point, end_zeros]))

    # Never use any padding for the first two dimensions, since these are always batch_size, predictions_per_sample
    # and should stay constant all the way through the framework
    batch_and_sample_padding = tf.zeros((2, 2), dtype=tf.int32)
    padding = tf.concat([batch_and_sample_padding, padding], axis=0)

    R_new = tf.pad(R, padding)

    # Mark operations as handled
    router.mark_operation_handled(current_operation)

    # Forward the calculated relevances
    router.forward_relevance_to_operation(R_new, current_operation, input_to_slicing_operation.op)
