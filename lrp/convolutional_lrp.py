from lrp import lrp_util
import tensorflow as tf
from constants import EPSILON


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

    # Initialize a bias tensor with 'out_depth' zeros
    positive_bias_tensor = tf.zeros_like(tf.shape(R)[-1], dtype=tf.float32)

    # Remember that there was no bias
    with_bias = False

    # If the top operation is an addition (i.e. the above assumption
    # does not hold), move through the graph to find the output of the nearest convolution
    if current_operation.type in ['BiasAdd', 'Add']:
        # Shape of convolution_tensor: (batch_size, out_height, out_width, out_depth)
        convolution_tensor = lrp_util.find_first_tensor_from_type(current_tensor, 'Conv2D')
        bias_tensor = lrp_util._get_input_bias_from_add(current_tensor)
        positive_bias_tensor = lrp_util.replace_negatives_with_zeros(bias_tensor)
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

    # Reshape the extracted patches to get a tensor of shape
    # (batch_size, out_height, out_width, filter_height, filter_width, input_channels)
    image_patches = tf.reshape(image_patches,
                               [batch_size, output_height, output_width, filter_height, filter_width, input_channels])

    # Add an extra dimesion to the filters to get shape:
    # (1, filter_height, filter_width, input_channels, output_channels)
    filters = tf.expand_dims(filters, 0)

    # Add an extra dimension to the image_patches to get shape:
    # (batch_size, out_height, out_width, filter_height, filter_width, input_channels, 1)
    image_patches = tf.expand_dims(image_patches, -1)

    # Multiply each patch by each filter to get the z_ijk's in a tensor zs with shape:
    # (batch_size, out_height, out_width, filter_height, filter_width, input_channels, output_channels)
    zs = tf.multiply(filters, image_patches)

    # Replace the negative elements with zeroes to only have the positive z's left (i.e. z_ijk^+)
    # Shape is still: (batch_size, out_height, out_width, filter_height, filter_width, input_channels, output_channels)
    zp = lrp_util.replace_negatives_with_zeros(zs)

    # Sum over each patch and add the positive parts of bias to get z_jk+'s
    # Shape of zp_sum: (batch_size, out_height, out_width, 1, 1, 1, output_channels)
    zp_sum = tf.reduce_sum(zp, [3, 4, 5], keep_dims=True)

    # Add stabilizer to the sum to avoid dividing by 0
    # Shape is still: (batch_size, out_height, out_width, 1, 1, 1, output_channels)
    zp_sum += EPSILON

    # Add a dimension to the bias and add it to the z's
    # Shape of zp_sum: (batch_size, out_height, out_width, 1, 1, 1, output_channels)
    zp_sum += tf.expand_dims(positive_bias_tensor, 0)

    # Find the number of predictions per sample from R
    relevances_shape = tf.shape(R)
    predictions_per_sample = relevances_shape[1]

    # Find the relative contribution of each feature in the input to the activations,
    # i.e. the ratio between the z_ijk's and the z_jk's
    # shape of fractions:
    # (batch_size, out_height, out_width, kernel_height, kernel_width, input_channels, output_channels)
    fractions = (zp / zp_sum)

    # Add the predictions_per_sample dimension to be able to broadcast fractions over the different
    # predictions for the same sample
    # Shape after expand_dims:
    # (batch_size, predictions_per_sample=1, out_height, out_width, kernel_height, kernel_width, input_channels, output_channels)
    fractions = tf.expand_dims(fractions, 1)

    # Reshape the relevance from the upper layer
    # shape of upper_layer_relevance:
    # (batch_size, predictions_per_sample, output_height, output_width, 1, 1, 1, output_channels)
    upper_layer_relevance = tf.reshape(R,
                                       [batch_size, predictions_per_sample, output_height, output_width, 1, 1,
                                        1, output_channels])


    # Find the absolute contribution of each feature
    # Shape of relevance:
    # (batch_size, predictions_per_sample, out_height, out_width, kernel_height, kernel_width, input_channels, output_channels)
    relevance = fractions * upper_layer_relevance

    # Sum the relevances over the filters
    # Shape of R_new:
    # (batch_size, predictions_per_sample, out_height, out_width, kernel_height, kernel_width, input_channels)
    R_new = tf.reduce_sum(relevance, 7)

    # Put the batch size and predictions_per_sample on the same dimension to be able to use the patches_to_images tool.
    # Also rearrange patches back to lists from the "small images".
    # Shape of relevances after reshape:
    # (batch_size * predictions_per_sample, out_height, out_width, filter_height * filter_width * input_channels)
    R_new = tf.reshape(R_new, [batch_size*predictions_per_sample, output_height, output_width,
                               filter_height * filter_width * input_channels])

    # Reconstruct the shape of the input, thereby summing the relevances for each individual feature
    R_new = lrp_util.patches_to_images(R_new, batch_size*predictions_per_sample, input_height, input_width,
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
