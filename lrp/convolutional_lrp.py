from lrp import lrp_util
from linear_lrp import linear_with_config
from lrp.configuration import LAYER
import tensorflow as tf


def convolutional(router, R):
    """
    Convolutional lrp
    :param router: the router object to report changes to
    :param R: the list of tensors containing the relevances from the upper layers
    """
    # Sum the potentially multiple relevances from the upper layers
    # Shape of R: (batch_size, predictions_per_sample, out_height, out_width, out_depth)
    R = lrp_util.sum_relevances(R)

    # Start by assuming the activation tensor is the output
    # of a convolution (i.e. not an addition with a bias)
    # Shape of current_tensor and convolution_tensor: (batch_size, out_height, out_width, out_depth)
    current_operation = router.get_current_operation()
    current_tensor = convolution_tensor = current_operation.outputs[0]

    # Remember that there was no bias
    with_bias = False

    bias_tensor = None
    # If the top operation is an addition (i.e. the above assumption
    # does not hold), move through the graph to find the output of the nearest convolution
    if current_operation.type in ['BiasAdd', 'Add']:
        # Shape of convolution_tensor: (batch_size, out_height, out_width, out_depth)
        convolution_tensor = lrp_util.find_first_tensor_from_type(current_tensor, 'Conv2D')
        bias_tensor = lrp_util._get_input_bias_from_add(current_tensor)
        with_bias = True

    # Find the inputs to the convolution
    (conv_input, filters) = convolution_tensor.op.inputs

    # Find the padding and strides that were used in the convolution
    padding = convolution_tensor.op.get_attr("padding").decode("UTF-8")
    strides = convolution_tensor.op.get_attr("strides")

    # Extract dimensions of the filters
    filter_sh = filters.get_shape().as_list()
    filter_height = filter_sh[0]
    filter_width = filter_sh[1]

    # Get shape of the input
    input_shape = tf.shape(conv_input)
    batch_size = input_shape[0]
    input_height = input_shape[1]
    input_width = input_shape[2]
    input_channels = input_shape[3]

    # Get the shape of the output of the convolution
    convolution_tensor_shape = tf.shape(convolution_tensor)
    output_height = convolution_tensor_shape[1]
    output_width = convolution_tensor_shape[2]
    output_channels = convolution_tensor_shape[3]

    # Extract every patch of the input (i.e. portion of the input that a filter looks at a
    # time), to get a tensor of shape
    # (batch_size, out_height, out_width, filter_height*filter_width*input_channels)
    image_patches = tf.extract_image_patches(conv_input, [1, filter_height, filter_width, 1],
                                             strides, [1, 1, 1, 1], padding)

    # Reshape patches to suit linear
    # shape: (batch_size * out_height * out_width, filter_height * filter_width * input_channels)
    linear_input = tf.reshape(image_patches,
                              (batch_size * output_height * output_width,
                               filter_height * filter_width * input_channels))

    # Reshape finters to suit linear
    linear_filters = tf.reshape(filters, (-1, output_channels))

    # Find the number of predictions per sample from R
    relevances_shape = tf.shape(R)
    predictions_per_sample = relevances_shape[1]

    # Transpose relevances to suit linear
    # Shape goes from (batch_size, predictions_per_sample, out_height, out_width, out_channels)
    # to: (batch_size, out_height, out_width, predictions_per_sample, out_channels)
    R_shape = tf.shape(R)

    # Make transpose order (0, 2, .. , 1, last_dim)
    # This in necessary because for conv1d the output might have been expanded which
    # makes the output size partially unknown
    transpose_order = tf.concat([[0], tf.range(2, tf.size(R_shape) - 1), [1], [tf.size(R_shape) - 1]], 0)

    # Do the actual transpose
    linear_R = tf.transpose(R, transpose_order)

    # Reshape linear_R to have three dimensions
    linear_R = tf.reshape(linear_R,
                          (batch_size * output_height * output_width, predictions_per_sample, output_channels))

    # TODO: refactor to call with appropriate configuration
    # Fetch configuration for linear_lrp
    config = router.get_configuration(LAYER.CONVOLUTIONAL)

    # Pass the responsibility to linear_lrp
    # Shape of linear_R_new:
    # (batch_size * out_height * out_width, predictions_per_sample, filter_height * filter_width * input_channels)
    linear_R_new = linear_with_config(linear_R, linear_input, linear_filters, config, bias=bias_tensor)

    # Shape back to be able to restitch
    linear_R_new = tf.reshape(linear_R_new, (
        batch_size, output_height, output_width, predictions_per_sample, filter_height * filter_width * input_channels))

    # Transpose back to be able to restitch
    # New shape:
    # (batch_size, predictions_per_sample, out_height, out_width, filter_height * filter_width * input_channels)
    linear_R_new = tf.transpose(linear_R_new, [0, 3, 1, 2, 4])

    # Gather batch_size and predictions_per_sample
    # New shape:
    # (batch_size * predictions_per_sample, out_height, out_width, filter_height * filter_width * input_channels)
    linear_R_new = tf.reshape(linear_R_new,
                              (batch_size * predictions_per_sample, output_height, output_width,
                               filter_height * filter_width * input_channels))

    # Restitch relevances to the input size
    R_new = lrp_util.patches_to_images(linear_R_new, batch_size * predictions_per_sample, input_height, input_width,
                                       input_channels, output_height, output_width, filter_height, filter_width,
                                       strides[1], strides[2], padding)

    # Reshape the calculated relevances from
    # (batch_size * predictions_per_sample, input_height, input_width, input_channels) to new shape:
    # (batch_size, predictions_per_sample, input_height, input_width, input_channels)
    R_new = tf.reshape(R_new, (batch_size, predictions_per_sample, input_height, input_width, input_channels))

    # Report handled operations
    router.mark_operation_handled(current_tensor.op)
    router.mark_operation_handled(convolution_tensor.op)

    # In case of 1D convolution we need to skip the squeeze operation in
    # the path towards the input
    if with_bias and current_tensor.op.inputs[0].op.type == 'Squeeze':
        router.mark_operation_handled(current_tensor.op.inputs[0].op)

    # Forward the calculated relevance to the input of the convolution
    router.forward_relevance_to_operation(R_new, convolution_tensor.op, conv_input.op)
