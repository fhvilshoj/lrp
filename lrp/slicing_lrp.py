import lrp_util
import tensorflow as tf

def slicing(router, R):
    # Sum the relevances
    R = lrp_util.sum_relevances(R)

    # Get the current operation
    current_operation = router.get_current_operation()

    # Get the input to the slicing operation
    input_to_slicing_operation = current_operation.inputs[0]

    # Get the shape of the input to the slicing operation
    input_shape = tf.shape(input_to_slicing_operation)
    print(input_shape)

    # Find starting point of the slice operation
    starting_point = current_operation.inputs[1]
    print(starting_point)

    # Find size of the slice operation
    size_of_slice = current_operation.inputs[2]
    print(size_of_slice)

    # Find the number of zeros to insert after the relevances
    end_zeros = input_shape - (starting_point + size_of_slice)

    print(end_zeros)

    # For each axis, insert 'starting_point' zeros before the relevances and
    # 'size_of_split'-('starting_point' + 'size_of_split') zeros after the relevances
    padding = tf.transpose(tf.stack([starting_point, end_zeros]))
    R = tf.pad(R, padding)

    # Mark operations as handled
    router.mark_operation_handled(current_operation)

    # Forward the calculated relevances
    router.forward_relevance_to_operation(R, current_operation, input_to_slicing_operation.op)

