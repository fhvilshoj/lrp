import tensorflow as tf
from math import ceil
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.framework import sparse_tensor, ops

# Helper function that takes a tensor, goes back to the operation that created it, and determines which of the operation's inputs lead in the direction of the input to the network (i.e. the output of a previous layer)
def find_path_towards_input(tensor):
    # Find the inputs of the operation that created the tensor
    inputs = tensor.op.inputs
    # Expects minimum one input for the operation
    if (len(inputs) == 0):
        raise Exception('find path expected at least one input to the operation')
    # If there is only one input to the operation, return that input
    elif (len(inputs) == 1):
        return inputs[0]

    # Run through the inputs until finding an input that is not a variable (accessed with the operation identity) or a constant
    for tens in inputs:
        if tens.op.type not in ['Identity', 'Const']:
            return tens

    # In case everything else fails, return the first input
    return inputs[0]


# Helper function that takes a tensor and a type of operation as input, goes back to the operation that created the tensor, and then goes through the network (in the direction towards the input) until it finds a tensor created by an operation of the specified type
def find_first_tensor_from_type(tensor, t):
    # If there are no inputs to the operation that created the tensor, return nothing
    if not tensor.op.inputs:
        return None

    # Run through the inputs to the operation that created the tensor
    for input in tensor.op.inputs:
        # If a tensor that is created by the specified type of operation is found, return that tensor
        if input.op.type == t:
            return input

    # Move one layer (in the direction towards the input to the network) and continue the search
    return find_first_tensor_from_type(find_path_towards_input(tensor), t)

# Helper function that takes a tensor and replaces all negative entries with zeroes
def replace_negatives_with_zeros(tensor):
    return tf.where(tf.greater(tensor, 0), tensor, tf.zeros_like(tensor))

# Helper function that takes a tensor, finds the operation that created it, and recursively prints the inputs to the operation
def _print(tensor, R, index=''):
    print(index + str(tensor.op.type))
    for inp in tensor.op.inputs:
        _print(inp, R, index + '  ')

# Helper function borrowed from https://github.com/VigneshSrinivasan10/interprettensor/blob/master/interprettensor/modules/convolution.py#L209
# TODO: Do we want to change this to something we develop ourself?
def patches_to_images(patches, batch_size, rows_in, cols_in, channels, rows_out, cols_out, ksize_r, ksize_c,
                      stride_h, stride_r, padding):

    ksize_r_eff = ksize_r #+ (ksize_r - 1) * (rate_r - 1)
    ksize_c_eff = ksize_c #+ (ksize_c - 1) * (rate_c - 1)

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