from lrp import lrp, lrp_util
import tensorflow as tf


def convolutional(path, R):
    """
    Convolutional lrp
    :param tensor: the tensor of the upper activation (the output of the convolution)
    :param R: The upper layer relevance
    :return: lower layer relevance (i.e. relevance distributed to the input to the convolution)
    """

    # Start by assuming the activation tensor is the output
    # of a convolution (i.e. not an addition with a bias)
    # Tensor shape: (upper_layer_height, upper_layer_width, upper_layer_depth)
    tensor = convolution_tensor = path[0].outputs[0]
    positive_bias_tensor = tf.zeros_like((R.shape[-1]), dtype=tf.float32)
    with_bias = False

    # If the top operation is an addition (i.e. the above assumption
    # does not hold), move through the graph to find the output of the nearest convolution.
    if path[0].type in ['BiasAdd', 'Add']:
        convolution_tensor = lrp_util.find_first_tensor_from_type(tensor, 'Conv2D')
        bias_tensor = lrp_util.get_input_bias_from_add(tensor)
        positive_bias_tensor = lrp_util.replace_negatives_with_zeros(bias_tensor)
        with_bias = True

    # Find the inputs to the convolution
    (conv_input, filters) = convolution_tensor.op.inputs

    # Find the padding and strides that were used in the convolution
    padding = convolution_tensor.op.get_attr("padding").decode("UTF-8")
    strides = convolution_tensor.op.get_attr("strides")

    # Extract dimensions of the filters
    (filter_height, filter_width, input_channels, output_channels) = filters.get_shape().as_list()

    # Get shape of the input
    (batch, input_height, input_width, input_channels) = conv_input.get_shape().as_list()

    # Get the shape of the output of the convolution
    (_, output_height, output_width, _) = convolution_tensor.get_shape().as_list()

    # Extract every patch of the input (i.e. portion of the input that a filter looks at a
    # time), to get a tensor of shape
    # (batch, out_height, out_width, filter_height*filter_width*input_channels)
    image_patches = tf.extract_image_patches(conv_input, [1, filter_height, filter_width, 1],
                                             strides, [1, 1, 1, 1], padding)

    # Reshape the extracted patches to get a tensor I of shape
    # (batch, out_height, out_width, filter_height, filter_width, input_channels)
    image_patches = tf.reshape(image_patches,
                               [batch, output_height, output_width, filter_height, filter_width, input_channels])

    # Multiply each patch by each filter to get the z_ijk's in a tensor zs
    zs = tf.multiply(tf.expand_dims(filters, 0), tf.expand_dims(image_patches, -1))

    # Replace the negative elements with zeroes to only have the positive z's left (i.e. z_ijk^+)
    zp = lrp_util.replace_negatives_with_zeros(zs)

    # Sum over each patch and add the positive parts of bias to get z_jk+'s
    zp_sum = tf.reduce_sum(zp, [3, 4, 5], keep_dims=True)

    # Add stabilizer to the sum to avoid dividing by 0
    zp_sum += 1e-12 * tf.ones_like(zp_sum)

    zp_sum += tf.expand_dims(positive_bias_tensor, 0)

    # Reshape the relevance from the upper layer
    upper_layer_relevance = tf.reshape(R, [batch, input_height, input_width, 1, 1, 1, output_channels])

    # Find the contribution of each feature in the input to the activations,
    # i.e. the ratio between the z_ijk's and the z_jk's
    division = (zp / zp_sum)

    # Find the relevance of each feature
    relevance = division * upper_layer_relevance

    # Sum the relevances over the filters
    R_new = tf.reduce_sum(relevance, 6)

    # Reshape the relevance tensor, so each patch becomes a vector
    R_new = tf.reshape(R_new, [batch, output_height, output_width, filter_height * filter_width * input_channels])

    # Reconstruct the shape of the input, thereby summing the relevances for each individual pixel
    R_new = lrp_util.patches_to_images(R_new, batch, input_height, input_width, input_channels, output_height,
                                       output_width, filter_height, filter_width, strides[1], strides[2], padding)

    # Skip forward in graph according to the use of bias or not
    skip = 2 if with_bias else 1

    # In case of 1D convolution we need to skip the squeeze operation in
    # the path towards the input.
    skip += 1 if path[1].type == 'Squeeze' else 0

    return path[skip:], R_new
