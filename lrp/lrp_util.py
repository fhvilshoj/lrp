import tensorflow as tf
from constants import *


# Helper function that takes a tensor, goes back to the operation that created it,
# and determines which of the operation's inputs lead in the direction of the input
# to the network (i.e. the output of a previous layer)
def find_path_towards_input(tensor):
    # Find the inputs of the operation that created the tensor
    inputs = tensor.op.inputs
    # Expects minimum one input for the operation
    if (len(inputs) == 0):
        raise Exception('find path expected at least one input to the operation')
    # If there is only one input to the operation, return that input
    elif (len(inputs) == 1):
        return inputs[0]

    # Run through the inputs until finding an input that is not a variable
    # (accessed with the operation identity) or a constant
    for tens in inputs:
        if tens.op.type not in ['Identity', 'Const']:
            return tens

    # In case everything else fails, return the first input
    return inputs[0]


# Helper function that takes a tensor and a type of operation (or a list of types)
# as input, goes back to the operation that created the tensor, and then goes through
# the network (in the direction towards the input) until it finds a tensor created by
# an operation of the specified type
def find_first_tensor_from_type(tensor, t):
    if not isinstance(t, list):
        t = [t]
    # If there are no inputs to the operation that created the tensor, return nothing
    if not tensor.op.inputs:
        return None

    # Run through the inputs to the operation that created the tensor
    for inp in tensor.op.inputs:
        # If a tensor that is created by the specified type of operation is found, return that tensor
        if inp.op.type in t:
            return inp

    # Move one layer (in the direction towards the input to the network) and continue the search
    return find_first_tensor_from_type(find_path_towards_input(tensor), t)


# Determine which type of operation an addition (the input) is associated with
def addition_associated_with(tensor):
    known_types = ['Conv2D', 'MatMul', 'Mul']
    found_tensor = find_first_tensor_from_type(tensor, known_types)
    return found_tensor.op.type


# Helper function that takes a tensor and replaces all negative entries with zeroes
def replace_negatives_with_zeros(tensor):
    return tf.where(tf.greater(tensor, 0), tensor, tf.zeros_like(tensor))


# Finds the one of the two inputs to an add that is given as a constant or a variable
def _get_input_bias_from_add(tensor):
    # Find bias tensor by identifying which of the two inputs to the addition are a variable (accessed
    # with the operation identity) or a constant
    if tensor.op.inputs[0].op.type in ['Identity', 'Const']:
        bias = tensor.op.inputs[0]
    else:
        bias = tensor.op.inputs[1]
    return bias


def _logical_or(l1, l2):
    return [t[0] or t[1] for t in zip(l1, l2)]

# Helper function that sums all relevances in an array
def sum_relevances(relevances):
    # relevances are dictionaries with keys producer and relevance
    summed_relevances = relevances[0][RELEVANCE]

    # Check if there are more than one relevance in the array
    if len(relevances) > 1:
        # Check if the relevances are sparse
        # TODO: There is no test case that checks if multiple sparse relevances are added correctly
        if isinstance(relevances[0][RELEVANCE], tf.SparseTensor):
            # Add all the relevances to the sum using tf's sparse_add operation
            for i in range(1, len(relevances)):
                summed_relevances = tf.sparse_add(summed_relevances, relevances[i][RELEVANCE])
        # If the relevances are dense, we can use tf's normal add operation
        else:
            for i in range(1, len(relevances)):
                summed_relevances = tf.add(summed_relevances, relevances[i][RELEVANCE])

    return summed_relevances


def _find_operations_in_LSTM(first_operation_in_LSTM, between_ops):
    # Create new path that contains all operations belonging to the LSTM
    LSTM_path = []

    # Create a variable for remembering the start of the LSTM
    start_of_LSTM = None

    # Create a new queue that holds operations in the LSTM that have not been handled yet
    # and add the first operation in the LSTM as the starting element
    operations_to_be_handled = [first_operation_in_LSTM]

    # Create an array of flags that indicate if we have looked at a operation before during
    # our LSTM traversal (to avoid getting stuck in loops)
    g = first_operation_in_LSTM.graph

    # Make a list of indicators telling if we have considered a given node before.
    reached_ops = [False] * (g._last_id + 1)

    # Keep processing operations from the queue of operations in the LSTM that have not been handled yet
    # as long as there are elements in the queue
    while operations_to_be_handled:
        # Pop the front element
        current_operation = operations_to_be_handled.pop(0)

        # Mark the operation as reached
        reached_ops[current_operation._id] = True

        # Add the operation to the list of operations that are part of the LSTM
        LSTM_path.append(current_operation)

        # Check if the operation is a transpose that has a TensorArrayScatter as consumer
        # in which case it indicates the end of the LSTM
        if current_operation.type == "Transpose":
            is_start_of_LSTM = False
            for output in current_operation.outputs:
                for consumer in output.consumers():
                    if "TensorArrayScatter" in consumer.type:
                        is_start_of_LSTM = True
                        break
            if is_start_of_LSTM:
                # Remember the transpose operation that is the start of the LSTM
                start_of_LSTM = current_operation
                # Move on to the next operation in the queue (we do not want to add the inputs of the current
                # operation to the queue because they are not part of the LSTM)
                continue

        # Push all non-visited input operations that are part of the overall path from the input to the output
        # to the queue of operations in the LSTM that have not been handled yet
        for input in current_operation.inputs:
            input_operation = input.op
            if between_ops[input_operation._id] and not reached_ops[input_operation._id]:
                operations_to_be_handled.append(input_operation)

    # Find the next operation to consider after the LSTM (i.e. the operation that produced the input to the LSTM)
    # If we did not find the beginning of the LSTM, raise an error
    if not start_of_LSTM:
        raise ValueError("Did not find the beginning op the LSTM")

    # Return the list of operations in the LSTM, the transpose that starts the LSTM, and the indicators
    # of which operations we reached in this LSTM
    return LSTM_path, start_of_LSTM, reached_ops


# Add the operation right after the LSTM to the queue of operations to take care of and begin
# a new path (the path that is before the LSTM in the original computational graph)

def _rearrange_op_list(output, between_ops):
    g = output.op.graph

    # Make a list of indicators telling if we have considered a given node before.
    reached_ops = [False] * (g._last_id + 1)

    # Initialize list for holding the contexts
    context_list = []

    # Current path to append operations to
    current_path = []

    # Create a queue and push the operation that created the output of the graph
    queue = [output.op]

    # Work through the queue
    while queue:

        # Pop the first element of the queue (i.e. the first operation)
        op = queue.pop(0)

        # Check if the operation is a transpose
        if op.type == 'Transpose':
            # If the operation is a transpose, check if it has a TensorArrayGather operation as input
            has_ta_gather_input = False
            for input in op.inputs:
                if "TensorArrayGather" in input.op.type:
                    has_ta_gather_input = True
                    break

            # If the operation has a TensorArrayGather operation as input, this is the start of a LSTM, so
            # we are at the end of the current path
            if has_ta_gather_input:
                # Close current path, add it to the current context and append the context to the list of
                # contexts
                context_list.append({CONTEXT_PATH: current_path, CONTEXT_TYPE: NON_LSTM_CONTEXT_TYPE})

                # Pass the responsibility of handling the LSTM to the appropriate handler
                LSTM_path, start_of_LSTM, reached_ops_from_LSTM = _find_operations_in_LSTM(op, between_ops)

                # Extract the operation responsible for input to the LSTM
                operation_before_LSTM = start_of_LSTM.inputs[0].op

                # Merge current reached ops with those found in the LSTM handler
                reached_ops = _logical_or(reached_ops, reached_ops_from_LSTM)

                # Append the new LSTM context to the list of contexts
                context_list.append({CONTEXT_PATH: LSTM_path,
                                     CONTEXT_TYPE: LSTM_CONTEXT_TYPE,
                                     EXTRA_CONTEXT_INFORMATION:
                                         {LSTM_BEGIN_TRANSPOSE_OPERATION: start_of_LSTM,
                                          LSTM_INPUT_OPERATION: operation_before_LSTM}})

                # Add the other side of the LSTM to the queue
                queue.append(operation_before_LSTM)

                # Open new path
                current_path = []

                # Skip to next operation in the queue
                continue

        # Add the operation to the ordered list of operations
        current_path.append(op)

        # Remember that we have handled the operation
        reached_ops[op._id] = True

        # Run through the inputs to the current operation
        for i in op.inputs:
            # Check if the input is part of the path between input and output
            if between_ops[i.op._id]:

                # Check if all consumers of of the operation that created the input have been handled:
                consumers_handled_already = True
                # Run through all outputs of the operation
                for out in i.op.outputs:
                    # Stop if any of the consumers of the outputs have not been handled
                    for c in out.consumers():
                        # Check if the consumer is in the path between input and output and check if
                        # we have not reached it before
                        if between_ops[c._id] and not reached_ops[c._id]:
                            consumers_handled_already = False

                    # If we found a consumer not handled we cannot add
                    # the operation of the input to the queue quite yet.
                    if not consumers_handled_already:
                        break

                # If all consumers have been taken care of and we haven't looked at the operation before,
                # add it to the queue
                if consumers_handled_already and not reached_ops[i.op._id]:
                    queue.append(i.op)

    # Append the last context to the final list of contexts
    context_list.append({CONTEXT_PATH: current_path, CONTEXT_TYPE: NON_LSTM_CONTEXT_TYPE})
    return context_list


# Helper function that traverses computation graph through all nodes
# to find the path from the output back to the input.
def get_operations_between_output_and_input(input, output):
    g = output.op.graph

    # Make a list of indicators telling if we have considered a given node before.
    reached_ops = [False] * (g._last_id + 1)

    def _markReachedOps(from_ops, reached_ops):
        # Make queue of the elements of the from ops argument. Both
        # handeling lists and single objects.
        queue = [from_ops.op] if not isinstance(from_ops, list) else [t.op for t in from_ops]

        # Run until we considered all paths from the input out into
        # the graph
        while queue:
            op = queue.pop(0)

            # Only consider nodes in the graph that we have not seen before
            if not reached_ops[op._id]:
                # Remember that we saw the current node
                reached_ops[op._id] = True

                # Add its consumers to the queue
                for output in op.outputs:
                    for consumer in output.consumers():
                        queue.append(consumer)

    # Find all nodes in graph reachable from the input
    _markReachedOps(input, reached_ops)

    # Make new list of boolean indicators for backwards pass
    between_ops = [False] * (g._last_id + 1)

    # List used for holding all the nodes in graph that we visit
    # On the path from output to input. Note that nodes with no
    # direct path to input might be included when there are while
    # loops in the graph.
    between_op_list = []
    queue = [output.op]
    while queue:
        op = queue.pop(0)
        # We are interested in this operation only if we saw the
        # node in the pass from input towards output.
        if reached_ops[op._id]:
            # This indicates that current node is part of path
            # between ouput and input, so remember it.
            between_ops[op._id] = True
            between_op_list.append(op)

            # Clear the boolean so we won't consider it again.
            reached_ops[op._id] = False

            # Add the operation's inputs to the queue
            if "Reshape" in op.type:
                # If the operation is a reshape, we only care about its first input
                # since that is the input that stems from the input to the network while all other inputs to the
                # reshape operation are irrelevant for us
                queue.append(op.inputs[0].op)
            else:
                # For all other operations, we add all its inputs to the queue
                for inp in op.inputs:
                    queue.append(inp.op)

    between_op_list = _rearrange_op_list(output, between_ops)

    return between_ops, between_op_list


# Helper function that takes a tensor, finds the operation that created it,
# and recursively prints the inputs to the operation
def _print(tensor, index=''):
    print(index + str(tensor.op.type))
    for inp in tensor.op.inputs:
        _print(inp, index + ' | ')


def patches_to_images(patches, batch_size, rows_in, cols_in, channels, rows_out, cols_out, ksize_r, ksize_c,
                      stride_r, stride_c, padding):

    if padding == 'SAME':
        rows_out = tf.cast((tf.ceil(rows_in / stride_r)), tf.int32)
        cols_out = tf.cast((tf.ceil(cols_in / stride_c)), tf.int32)
        pad_rows = ((rows_out - 1) * stride_r + ksize_r - rows_in) // 2
        pad_cols = ((cols_out - 1) * stride_c + ksize_c - cols_in) // 2

    elif padding == 'VALID':
        rows_out = tf.cast(tf.ceil((rows_in - ksize_r + 1) / stride_r), tf.int32)
        cols_out = tf.cast(tf.ceil((cols_in - ksize_c + 1) / stride_c), tf.int32)
        pad_rows = 0  # (rows_out - 1) * stride_r + ksize_r - rows_in
        pad_cols = 0  # (cols_out - 1) * stride_c + ksize_c - cols_in


    # Reshape the patches from shape (batch_size, out_height, out_width, kernel_height * kernel_width * in_channels) to
    # shape (batch_size, out_height, out_width, kernel_height,kernel_width, in_channels)
    patch_shape = tf.stack([batch_size, rows_out, cols_out, ksize_r, ksize_c, channels])
    patches = tf.reshape(patches, patch_shape)

    # Transpose the patches to shape (out_height, out_width, kernel_height,kernel_width, batch_size, in_channels)
    patches = tf.transpose(patches, (1, 2, 3, 4, 0, 5))

    # Reshape the patches to shape (out_height * out_width * kernel_height * kernel_width, batch_size * in_channels)
    patches = tf.reshape(patches, tf.stack([-1, batch_size * channels]))

    sparse_indexes_size = rows_out * cols_out * ksize_r * ksize_c
    sparse_indexes = tf.TensorArray(tf.int64, size=sparse_indexes_size, dynamic_size=True)

    batch_size = tf.cast(batch_size, dtype=tf.int64)
    ksize_r = tf.cast(ksize_r, dtype=tf.int64)
    ksize_c = tf.cast(ksize_c, dtype=tf.int64)
    channels = tf.cast(channels, dtype=tf.int64)
    rows_out = tf.cast(rows_out, dtype=tf.int64)
    cols_out = tf.cast(cols_out, dtype=tf.int64)
    rows_in = tf.cast(rows_in, dtype=tf.int64)
    cols_in = tf.cast(cols_in, dtype=tf.int64)
    pad_rows = tf.cast(pad_rows, dtype=tf.int64)
    pad_cols = tf.cast(pad_cols, dtype=tf.int64)

    def _loop_over_rows_out(r_out_center, sparse_indexes, tensor_array_index):

        #############################################################################
        def _loop_over_columns_out(c_out_center, sparse_indexes, tensor_array_index):
            r_low = r_out_center * stride_r - pad_rows
            c_low = c_out_center * stride_c - pad_cols
            r_high = r_low + ksize_r
            c_high = c_low + ksize_c

            #############################################################################
            def _loop_over_kernel_rows(kernel_row_index, current_input_row, sparse_indexes, tensor_array_index):

                #############################################################################
                def _loop_over_kernel_columns(kernel_column_index, current_input_column, sparse_indexes, tensor_array_index):
                    def _add_tuple():
                        index_tuple = (current_input_row * (cols_in) + current_input_column,
                                       r_out_center * (cols_out * ksize_r * ksize_c) +
                                       c_out_center * (ksize_r * ksize_c) +
                                       kernel_row_index * (ksize_c) +
                                       kernel_column_index)
                        return sparse_indexes.write(tensor_array_index, index_tuple), tf.add(tensor_array_index, 1)

                    def _do_not_add_tuple():
                        return sparse_indexes, tensor_array_index

                    sparse_indexes, tensor_array_index = tf.cond(
                        tf.logical_and(
                            tf.logical_and(
                                current_input_column >= 0,
                                current_input_column < cols_in
                            ),
                            tf.logical_and(
                                current_input_row >= 0,
                                current_input_row < rows_in
                            )
                        ),
                        true_fn=_add_tuple,
                        false_fn=_do_not_add_tuple)

                    return tf.add(kernel_column_index, 1), tf.add(current_input_column,
                                                                  1), sparse_indexes, tensor_array_index

                #############################################################################


                *_, sparse_indexes, tensor_array_index = tf.while_loop(
                    cond=lambda _, t, *__: tf.less(t, c_high),
                    body=_loop_over_kernel_columns,
                    loop_vars=[tf.constant(0, dtype=tf.int64), c_low, sparse_indexes, tensor_array_index]
                )

                return tf.add(kernel_row_index, 1), tf.add(current_input_row, 1), sparse_indexes, tensor_array_index

            #############################################################################
            *_, sparse_indexes, tensor_array_index = tf.while_loop(
                cond=lambda _, t, *__: tf.less(t, r_high),
                body=_loop_over_kernel_rows,
                loop_vars=[tf.constant(0, dtype=tf.int64), r_low, sparse_indexes, tensor_array_index]
            )

            return tf.add(c_out_center, 1), sparse_indexes, tensor_array_index

        #############################################################################

        _, sparse_indexes, tensor_array_index = tf.while_loop(
            cond=lambda t, *_: tf.less(t, cols_out * stride_c),
            body=_loop_over_columns_out,
            loop_vars=[tf.constant(0, dtype=tf.int64), sparse_indexes, tensor_array_index]
        )
        return tf.add(r_out_center, 1), sparse_indexes, tensor_array_index

    #############################################################################
    _, sparse_indexes, tensor_array_index = tf.while_loop(
        cond=lambda t, *_: tf.less(t, rows_out * stride_r),
        body=_loop_over_rows_out,
        loop_vars=[tf.constant(0, dtype=tf.int64), sparse_indexes, 0]
    )


    tensor_array_index = tf.cast(tensor_array_index, dtype=tf.int32)

    sparse_indexes = sparse_indexes.gather(tf.range(tensor_array_index))

    columns = rows_out * cols_out * ksize_r * ksize_c
    rows = rows_in * cols_in
    dense_shape = tf.stack([rows, columns])

    transformation_matrix = tf.SparseTensor(
        sparse_indexes,
        tf.ones((tensor_array_index,)),
        dense_shape
    )

    batch_size = tf.cast(batch_size, dtype=tf.int32)
    ksize_r = tf.cast(ksize_r, dtype=tf.int32)
    ksize_c = tf.cast(ksize_c, dtype=tf.int32)
    channels = tf.cast(channels, dtype=tf.int32)
    rows_out = tf.cast(rows_out, dtype=tf.int32)
    cols_out = tf.cast(cols_out, dtype=tf.int32)
    rows_in = tf.cast(rows_in, dtype=tf.int32)
    cols_in = tf.cast(cols_in, dtype=tf.int32)

    new_relevances = tf.sparse_tensor_dense_matmul(transformation_matrix, patches)

    new_relevances = tf.reshape(new_relevances, tf.stack((rows_in, cols_in, batch_size, channels)))

    new_relevances = tf.transpose(new_relevances, (2, 0, 1, 3))

    return new_relevances


# Helper function that uses tensorflow's Print function to print the value of a tensor
def print_value(input, name):
    str = '\n --------------     ' + name + ':     -------------- \n'
    return tf.Print(input, [input], summarize=1000, message=str)
