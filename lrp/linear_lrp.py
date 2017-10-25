import tensorflow as tf
from lrp import lrp_util
from constants import BIAS_DELTA, EPSILON

def linear_epsilon(R, input, weights, bias=None, output=None):
    """
    Simple linear layer used for partial computations of LSTM
    :param R: tensor of relevance to distribute. Shape: (batch_size, output_width)
    :param input: The input tensor. Shape: (batch_size, input_width)
    :param weights: The kernel weights. Shape: (input_width, output_width)
    :param output: Optional output tensor. Shape: (batch_size, output_width)
           If none output is calculated as input times weights plus bias.
    :param bias: Optional tensor with bias. Shape: (output_width) or (batch_size, output_width)
    :return: Redistributed relevance. Shape: (batch_size, input_width)
    """
    # If no output tensor given; construct one by multiplying input and weights
    if output is None:
        output = tf.matmul(input, weights)
        # Only add bias when bias is not none
        if bias is not None:
            output += bias

    # Prepare batch for elementwise multiplication. New input shape: (batch_size, input_width, 1)
    input = tf.expand_dims(input, -1)

    # Find Z_kij's. Shape: (batch, input_width, output_width)
    zs = tf.multiply(input, weights)

    # When bias is given divide it equally among the i's to avoid relevance loss
    if bias is not None:
        # Number of input features to divide relevance among
        input_features = input.get_shape().as_list()[1]

        # Find the bias to divide equally among the rows (This includes the stabilizer: epsilon)
        # Shape: (output_width) or (batch, output_width)
        bias_to_divide = (BIAS_DELTA * bias + EPSILON * tf.sign(output))

        # Divide the bias (and stabilizer) equally between the `input_features` (rows of zs)
        # Shape: (output_width) or (batch, output_width)
        bias_per_feature = bias_to_divide / input_features

        # Expand the second to last dimension to be able to add the bias through the rows of zs
        # Shape: (1, output_width) or (batch, 1, output_width)
        bias_per_feature = tf.expand_dims(bias_per_feature, -2)

        # Add bias through rows of zs
        # Shape: (batch, input_width, output_width)
        zs = zs + bias_per_feature

    # Add stabilizer to denominator to avoid dividing with 0
    # Shape: (batch, output_width)
    denominator = output + EPSILON * tf.sign(output)

    # Expand the second to last dimension to be able to divide the denominator through the rows of zs
    # Shape after expand_dims: (batch, 1, output_width)
    denominator = tf.expand_dims(denominator, -2)

    # Find the relative contribution from feature i to neuron j for input k
    # Shape: (batch, input_width, output_width)
    fractions = tf.divide(zs, denominator)

    # Transpose the fractions to be able to do matrix multiplication with R
    # Shape after transpose: (batch_size, output_width, input_width)
    fractions = tf.transpose(fractions, [0, 2, 1])

    # Expand the second to last dimension to be able to do matrix multiplication with fractions
    # Shape after expand_dims: (batch_size, 1, output_width)
    R = tf.expand_dims(R, -2)

    # Calculate relevance by doing matrix multiplication of R and fractions
    # R_new shape: (batch_size, 1, input_width)
    R_new = tf.matmul(R, fractions)

    # Remove the extra dimension added to R above
    # Final shape: (batch_size, input_width)
    R_new = tf.squeeze(R_new, -2)

    return R_new


# TODO Should this function also take output as an optional input to be consistent with other rules?
def linear_alpha(R, input, weights, bias=None):

    # Prepare batch for elementwise multiplication
    input = tf.expand_dims(input, -1)

    # Perform elementwise multiplication of input, weights to get z_kij which is the contribution from
    # feature i to neuron j for input k
    zs = input * weights

    # Replace the negative elements with zeroes to only have the positive z's left (i.e. z_kij^+)
    zp = lrp_util.replace_negatives_with_zeros(zs)

    # Take the sum of each column of z_kij^+'s
    zp_sum = tf.reduce_sum(zp, axis=1, keep_dims=True)

    # Find and add the positive parts of an eventual bias (i.e. find and add the b^+s).
    if bias is not None:
        # Replace the negative elements in the bias with zeroes
        bias_positive = lrp_util.replace_negatives_with_zeros(bias)

        # Add the sum of the z_ij^+'s and the positive bias (i.e. find the z_j^+'s)
        zp_sum = tf.add(zp_sum, bias_positive)

    # Add stabilizer to the denominator
    zp_sum += EPSILON

    # Find the relative contribution from feature i to neuron j for input k
    fractions = tf.divide(zp, zp_sum)

    # Prepare the fractions for the matmul below
    fractions = tf.transpose(fractions, perm=[0, 2, 1])

    # Add an extra dimension since we in the lrp router have added either one extra dimension for
    # predictions_per_sample (if the starting point relevances had shape (batch_size, predictions_per_sample, classes))
    # or two dimensions for predictions_per_sample (if the starting point relevances had shape
    # (batch_size, predictions_per_sample, classes)) to the relevances
    fractions = tf.expand_dims(fractions, 1)

    # Multiply relevances with fractions to find relevance per feature in input
    # In other words: Calculate the lower layer relevances (a combination of equation 60 and 62 in Bach 2015)
    R_new = tf.matmul(R, fractions)

    return R_new


def element_wise_linear(router, R):
    """
    Used to handle element wise multiplications (by transforming them into matrices)
    :param router: the router object to report changes to
    :param R: the list of tensors containing the relevances from the upper layers
    """
    # Tensor is the output of the current operation (i.e. Add, or Mul)
    current_operation = router.get_current_operation()
    current_tensor = current_operation.outputs[0]

    # Sum the potentially multiple relevances from the upper layers
    R = lrp_util.sum_relevances(R)

    multensor = current_tensor
    bias = None
    if current_tensor.op.type == 'Add':
        bias = lrp_util._get_input_bias_from_add(current_tensor)
        multensor = lrp_util.find_first_tensor_from_type(current_tensor, 'Mul')

    # Find the inputs to the matrix multiplication
    (input, weights) = multensor.op.inputs
    weights = tf.diag(weights)

    # Calculate new relevances with the alpha rule
    R_new = linear_alpha(R, input, weights, bias=bias)

    # Mark handled operations
    router.mark_operation_handled(current_operation)
    router.mark_operation_handled(multensor.op)

    # Forward relevance
    router.forward_relevance_to_operation(R_new, multensor.op, input.op)


def linear(router, R):
    """
    linear lrp
    :param router: the router object to report changes to
    :param R: the list of tensors containing the relevances from the upper layers
    """
    # Tensor is the output of the current operation (i.e. Add, or MatMul)
    tensor = router.get_current_operation().outputs[0]
    # Sum the potentially multiple relevances from the upper layers
    R = lrp_util.sum_relevances(R)

    # Start by assuming the activation tensor is the output of a matrix multiplication
    # (i.e. not an addition with a bias)
    # Tensor shape: (1, upper layer size)
    matmultensor = tensor
    bias = None

    # If the activation tensor is the output of an addition (i.e. the above assumption does not hold),
    # move through the graph to find the output of the nearest matrix multiplication.
    if tensor.op.type == 'Add':
        bias = lrp_util._get_input_bias_from_add(tensor)
        matmultensor = lrp_util.find_first_tensor_from_type(tensor, 'MatMul')

    # Find the inputs to the matrix multiplication
    (input, weights) = matmultensor.op.inputs

    # Calculate new relevances with the alpha rule
    R_new = linear_alpha(R, input, weights, bias=bias)


    # Mark handled operations
    router.mark_operation_handled(tensor.op)
    router.mark_operation_handled(matmultensor.op)

    # Forward relevance
    router.forward_relevance_to_operation(R_new, matmultensor.op, input.op)
