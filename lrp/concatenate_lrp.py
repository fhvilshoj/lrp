import tensorflow as tf
import lrp_util


# When we see a concatenate we want to split the incoming relevances accordingly
def concatenate(router, R):
    # Sum the potentially multiple relevances from the upper layers
    R = lrp_util.sum_relevances(R)

    # Get the current concatenate operation
    current_operation = router.get_current_operation()

    # Find axis that the concatenation was over
    axis = current_operation.inputs[-1]

    # Split relevances in same order. Start by initializing empty arrays to hold respectively the sizes that the
    # relevances shall be split in and the receivers of the relevances
    split_sizes = []
    input_operations = []
    # Run through the inputs to the current operation except the last (the last input is the "axis" input)
    for inp in current_operation.inputs[:-1]:
        # Add the operation to the array
        input_operations.append(inp.op)
        # Find the shape of the operation
        shape = tf.shape(inp)
        # Find and add the size of the input in the "axis" dimension
        split_sizes.append(shape[axis])

    # Adjust the axis to split over, since we in the lrp router have added either one extra dimension for
    # predictions_per_sample (if the starting point relevances had shape (batch_size, predictions_per_sample, classes))
    # or two dimensions for predictions_per_sample (if the starting point relevances had shape
    # (batch_size, predictions_per_sample, classes)) to the relevances
    if router.starting_point_relevances_did_not_have_predictions_per_sample_dimension():
        axis += 2
    else:
        axis += 1

    # Split the relevances over the "axis" dimension according to the found split sizes
    R_splitted = tf.split(R, split_sizes, axis)

    # Tell the router that we handled the concatenate operation
    router.mark_operation_handled(current_operation)

    # Forward relevance to the operation of the input to the concatenate operation
    for input_index, relevance in enumerate(R_splitted):
        router.forward_relevance_to_operation(relevance, current_operation, input_operations[input_index])
