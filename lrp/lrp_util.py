import tensorflow as tf

def find_path_towards_input(tensor):
    # Expects minimum one input for the operation
    inputs = tensor.op.inputs
    if (len(inputs) == 0):
        raise Exception('find path expected at least one input to the operation')
    elif (len(inputs) == 1):
        return inputs[0]

    for tens in inputs:
        if tens.op.type not in ['Identity', 'Const']:
            return tens

    return inputs[0]


def find_first_tensor_from_type(tensor, t):
    if not tensor.op.inputs:
        return None

    for input in tensor.op.inputs:
        if input.op.type == t:
            return input

    return find_first_tensor_from_type(find_path_towards_input(tensor), t)


def replace_negatives_with_zeros(tensor):
    return tf.where(tf.greater(tensor, 0), tensor, tf.zeros_like(tensor))


def _print(tensor, R, index=''):
    print(index + str(tensor.op.type))
    for inp in tensor.op.inputs:
        _print(inp, R, index + '  ')
