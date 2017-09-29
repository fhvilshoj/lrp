import tensorflow as tf

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
