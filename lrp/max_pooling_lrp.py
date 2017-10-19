from lrp import lrp_util
import tensorflow as tf

# TODO: We do not currently support max pooling where kernels span over the depts dimension (k: [1,1,1,d])
def max_pooling(router, R):
    """
    Max pooling lrp
    :param router: the router object to report changes to
    :param R: the list of tensors containing the relevances from the upper layers
    """
    # Sum the potentially multiple relevances from the upper layers
    R = lrp_util.sum_relevances(R)

    # Get current operation from the router
    current_operation = router.get_current_operation()

    # max pool output as current tensor
    current_tensor = current_operation.outputs[0]
    assert current_tensor.shape == R.shape, "Tensor and R should have same shape"

    # Find the input to the max pooling
    max_pool_input = current_operation.inputs[0]

    # Find the padding and strides that were used in the max pooling
    padding = current_operation.get_attr("padding").decode("UTF-8")
    strides = current_operation.get_attr("strides")
    kernel_size = current_operation.get_attr("ksize")

    # Get shape of the input
    (batch, input_height, input_width, input_channels) = max_pool_input.get_shape().as_list()

    # Get the shape of the output of the output of the max pool
    (_, output_height, output_width, output_channels) = current_tensor.get_shape().as_list()

    # Replace the negative elements with zeroes to only have the positive entries left
    max_pool_input_positive = lrp_util.replace_negatives_with_zeros(max_pool_input)

    # Extract every patch of the input (i.e. portion of the input that the kernel looks at a
    # time), to get a tensor of shape
    # (batch, out_height, out_width, kernel_height*kernel_width*input_channels)
    image_patches = tf.extract_image_patches(max_pool_input_positive, kernel_size,
                                             strides, [1, 1, 1, 1], padding)
    image_patches = tf.reshape(image_patches, [batch, output_height, output_width, kernel_size[1], kernel_size[2], input_channels])

    # Find the largest elements in each patch and set all other entries to zero (to find z_ijk+'s)
    max_elems = tf.reduce_max(image_patches, axis=[3, 4], keep_dims=True)
    zs = tf.where(tf.equal(image_patches, max_elems), image_patches, tf.zeros_like(image_patches))

    # Find the contribution of each feature in the input to the activations,
    # i.e. the ratio between the z_ijk's and the z_jk's (plus a small stabilizer to avoid division by zero)
    # TODO We do not take care of two equaly big entries (currently doubles given relevance)
    fraction = zs / (max_elems + lrp_util.EPSILON)

    # Find the relevance of each feature
    R = tf.reshape(R, [batch, output_height, output_width, 1, 1, output_channels])

    relevances = fraction * R
    relevances = tf.reshape(relevances, (batch, output_height, output_width, kernel_size[1]*kernel_size[2] * input_channels))

    # Reconstruct the shape of the input, thereby summing the relevances for each individual pixel
    R_new = lrp_util.patches_to_images(relevances, batch, input_height, input_width, input_channels, output_height,
                                       output_width, kernel_size[1], kernel_size[2], strides[1], strides[2], padding)

    # Report handled operations
    router.mark_operation_handled(current_operation)

    # Forward the calculated relevance to the input of the convolution
    router.forward_relevance_to_operation(R_new, current_operation, max_pool_input.op)


