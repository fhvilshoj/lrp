import tensorflow as tf
from constants import *


# When we see a split we want to concatenate the incoming relevances
def split(router, R):
    # Get the current split operation
    current_operation = router.get_current_operation()

    # Put all the received relevances in a single dictionary for fast lookup
    relevances_from_ops = dict()
    for r in R:
        # We put all the relevances in arrays to be able to hold multiple
        # relevances from the same producer
        if not r[RELEVANCE_PRODUCER] in relevances_from_ops:
            # When we haven't seen any relevance from the producer yet we add
            # the given relevance to an empty array and put it in the dictionary
            relevances_from_ops[r[RELEVANCE_PRODUCER]] = [r[RELEVANCE]]
        else:
            # When we saw relevance from the producer before we look op the list
            # and append the new relevance to the end of the list
            relevances_from_ops[r[RELEVANCE_PRODUCER]].append(r[RELEVANCE])

    # Fill the relevances for each output with zeros to make
    # sure that we end up with relevance of the right shape
    relevances_to_sum = []
    for output in current_operation.outputs:
        # Find the shape of the output and adjust it, since we in the lrp router have added either one extra
        # dimension for predictions_per_sample (if the starting point relevances had shape
        # (batch_size, predictions_per_sample, classes)) or two dimensions for predictions_per_sample
        # (if the starting point relevances had shape (batch_size, predictions_per_sample, classes)) to the relevances
        shape = tf.expand_dims(output, 1)
        if router.did_add_extra_dimension_for_multiple_predictions_per_sample():
            shape = tf.expand_dims(shape, 1)

        relevances_to_sum.append([tf.zeros_like(shape)])


    ### This section distributes relevances to the correct output of the split operation

    # Helper function which returns the index of the output tensor
    # corresponding to the input or -1 else
    def _get_output_index_for_tensor(tensor):
        for index, output in enumerate(current_operation.outputs):
            if output is tensor:
                return index
        return -1

    # Iterate over all the relevances one operation at a time
    for key, value in relevances_from_ops.items():
        # Find the opration that the relevances in `value` came from
        producer_operation = current_operation.graph._nodes_by_id[key]

        # Pointer to where in the current relevance array we are
        relevance_idx = 0

        # Iterate over all the inputs of the producer operation to
        # find all those corresponding to outputs of the current split
        # operation and append the relevance to the appropriate lists
        # in `relevances_to_sum`. We take advantage of the fact that
        # operations store the relevances they forward in the same order
        # as the order of their input (e.g. the relevance belonging
        # to the first input is stored first)
        for input in producer_operation.inputs:
            # Sanity check to make sure we do not get out of bounds
            # and to improve speed a bit by early exit
            if relevance_idx == len(value):
                break

            # Get the index of of current_split operations output
            # from the input tensor
            output_index = _get_output_index_for_tensor(input)
            # If there is a match between input and output (i.e `output_index` is not -1)
            if output_index >= 0:
                # Append relevance for later summing
                relevances_to_sum[output_index].append(value[relevance_idx])
                # Update relevance index
                relevance_idx += 1


    # Sum relevances for each output
    relevances_to_concatenate = []
    for relevance_list in relevances_to_sum:
        # Start the sum with the zeroes added under initialization
        relevance_sum = relevance_list[0]

        # Iterate over the existing relevances if any
        for relevance in relevance_list[1:]:
            # Accumulate the sum
            relevance_sum += relevance
        relevances_to_concatenate.append(relevance_sum)

    # Find axis to concatenate on
    axis = current_operation.inputs[0]

    # Adjust the axis to concatenate over, since we in the lrp router have added either one extra dimension for
    # predictions_per_sample (if the starting point relevances had shape (batch_size, predictions_per_sample, classes))
    # or two dimensions for predictions_per_sample (if the starting point relevances had shape
    # (batch_size, predictions_per_sample, classes)) to the relevances
    if router.did_add_extra_dimension_for_multiple_predictions_per_sample():
        axis += 2
    else:
        axis += 1

    # Concatenate relevances
    R_concatenated = tf.concat(relevances_to_concatenate, axis)

    # Tell the router that we handled this operation
    router.mark_operation_handled(current_operation)

    # Forward relevance to the operation of the input to the current operation
    router.forward_relevance_to_operation(R_concatenated, current_operation, current_operation.inputs[1].op)
