import tensorflow as tf
from math import ceil
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.framework import sparse_tensor, ops

EPSILON = 1e-12
BIAS_DELTA = 1  # (0 if we shouldn't consider bias in epsilon rule) 1 if we should


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


def sum_relevances(relevances):
    # relevances are dictionaries with keys producer and relevance
    summed_relevances = relevances[0]['relevance']
    if len(relevances) > 1:
        for i in range(1, len(relevances)):
            summed_relevances = tf.add(summed_relevances, relevances[i]['relevance'])

    return summed_relevances


def _rearrange_op_list(output, between_ops):
    g = output.op.graph

    # Make a list of indicators telling if we have considered a given node before.
    reached_ops = [False] * (g._last_id + 1)

    # Initialize list for holding the ordered operations
    between_op_list = []

    # Create a queue and push the operation that created the output of the graph
    queue = [output.op]

    # Work through the queue
    while queue:

        # Pop the first element of the queue (i.e. the first operation)
        op = queue.pop(0)
        # Add the operation to the ordered list of operations
        between_op_list.append(op)

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

    return between_op_list

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

            # Add inputs to the queue
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


# Helper function borrowed from
# https://github.com/VigneshSrinivasan10/interprettensor/blob/master/interprettensor/modules/convolution.py#L209
# TODO: Do we want to change this to something we develop ourself?
def patches_to_images(patches, batch_size, rows_in, cols_in, channels, rows_out, cols_out, ksize_r, ksize_c,
                      stride_r, stride_h, padding):
    ksize_r_eff = ksize_r  # + (ksize_r - 1) * (rate_r - 1)
    ksize_c_eff = ksize_c  # + (ksize_c - 1) * (rate_c - 1)

    if padding == 'SAME':
        rows_out = int(ceil(rows_in / stride_r))
        cols_out = int(ceil(cols_in / stride_h))
        pad_rows = ((rows_out - 1) * stride_r + ksize_r_eff - rows_in) // 2
        pad_cols = ((cols_out - 1) * stride_h + ksize_c_eff - cols_in) // 2

    elif padding == 'VALID':
        rows_out = int(ceil((rows_in - ksize_r_eff + 1) / stride_r))
        cols_out = int(ceil((cols_in - ksize_c_eff + 1) / stride_h))
        pad_rows = (rows_out - 1) * stride_r + ksize_r_eff - rows_in
        pad_cols = (cols_out - 1) * stride_h + ksize_c_eff - cols_in

    pad_rows, pad_cols = max(0, pad_rows), max(0, pad_cols)

    grad_expanded = array_ops.transpose(
        array_ops.reshape(patches, (batch_size, rows_out,
                                    cols_out, ksize_r, ksize_c, channels)),
        (1, 2, 3, 4, 0, 5)
    )
    grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))

    row_steps = range(0, rows_out * stride_r, stride_r)
    col_steps = range(0, cols_out * stride_h, stride_h)

    idx = []
    for i in range(rows_out):
        for j in range(cols_out):
            r_low, c_low = row_steps[i] - pad_rows, col_steps[j] - pad_cols
            r_high, c_high = r_low + ksize_r_eff, c_low + ksize_c_eff

            idx.extend([(r * (cols_in) + c,
                         i * (cols_out * ksize_r * ksize_c) +
                         j * (ksize_r * ksize_c) +
                         ri * (ksize_c) + ci)
                        for (ri, r) in enumerate(range(r_low, r_high))
                        for (ci, c) in enumerate(range(c_low, c_high))
                        if 0 <= r and r < rows_in and 0 <= c and c < cols_in
                        ])

    sp_shape = (rows_in * cols_in,
                rows_out * cols_out * ksize_r * ksize_c)

    sp_mat = sparse_tensor.SparseTensor(
        array_ops.constant(idx, dtype=ops.dtypes.int64),
        array_ops.ones((len(idx),), dtype=ops.dtypes.float32),
        sp_shape
    )

    jac = sparse_ops.sparse_tensor_dense_matmul(sp_mat, grad_flat)

    result = array_ops.reshape(
        jac, (rows_in, cols_in, batch_size, channels)
    )
    result = array_ops.transpose(result, (2, 0, 1, 3))

    return result


# Helper function that uses tensorflow's Print function to print the value of a tensor
def print_value(input, name):
    str = '\n --------------     ' + name + ':     -------------- \n'
    return tf.Print(input, [input], summarize=1000, message=str)
