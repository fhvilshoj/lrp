import tensorflow as tf
import lrp_util


# When we see a concatenate we want to split the incoming relevances accordingly
def concatenate(router, R):
    # Sum the potentially multiple relevances from the upper layers
    R = lrp_util.sum_relevances(R)

    # Get the current concatenate operation
    current_operation = router.get_current_operation()

    # Find axis and order of concatenate
    print(current_operation)

    # Split relevances in same order

    axis = current_operation.inputs[-1]
    split_sizes = []
    input_operations = []
    for inp in current_operation.inputs[:-1]:
        input_operations.append(inp.op)
        shape = tf.shape(inp)
        split_sizes.append(shape[axis])

    R_splitted = tf.split(R, split_sizes, axis)

    # Tell the router that we handled this operation
    router.mark_operation_handled(current_operation)

    # Forward relevance to the operation of the input to the current operation
    for input_index, relevance in enumerate(R_splitted):
        router.forward_relevance_to_operation(relevance, current_operation, input_operations[input_index])
