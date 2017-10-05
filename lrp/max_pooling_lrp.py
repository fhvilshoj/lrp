from lrp import lrp, lrp_util
import tensorflow as tf


def max_pooling(tensor, R):
    """
    Max pooling lrp
    :param tensor: the tensor of the upper activation of the max pooling layer
    :param R: The upper layer relevance
    :return: lower layer relevance
    """

    assert tensor.shape == R.shape, "Tensor and R should have same shape"

    # Find the input to the max pooling
    max_pool_input = tensor.op.inputs[0]

    # Find the padding and strides that were used in the max pooling
    padding = tensor.op.get_attr("padding").decode("UTF-8")
    strides = tensor.op.get_attr("strides")
    kernel_size = tensor.op.get_attr("ksize")

    # Get shape of the input
    (batch, input_height, input_width, input_channels) = max_pool_input.get_shape().as_list()

    # Get the shape of the output of the output of the max pool
    (_, output_height, output_width, _) = tensor.get_shape().as_list()

    # Replace the negative elements with zeroes to only have the positive entries left
    max_pool_input = lrp_util.replace_negatives_with_zeros(max_pool_input)

    # Extract every patch of the input (i.e. portion of the input that the kernel looks at a
    # time), to get a tensor of shape
    # (batch, out_height, out_width, kernel_height*kernel_width*kernel_depth)
    image_patches = tf.extract_image_patches(max_pool_input, kernel_size,
                                             strides, [1, 1, 1, 1], padding)

    image_patches = tf.Print(image_patches, [image_patches], message="IMAGE_PATCHES", summarize=1000)

    # Find the largest elements in each patch and set all other entries to zero (to find z_ijk+'s)
    max_elems = tf.reduce_max(image_patches, axis=3, keep_dims=True)
    zs = tf.where(tf.equal(image_patches, max_elems), image_patches, tf.zeros_like(image_patches))


    # Find the contribution of each feature in the input to the activations,
    # i.e. the ratio between the z_ijk's and the z_jk's (plus a small stabilizer to avoid division by zero)
    fraction = zs / (max_elems + 1e-12)

    # Find the relevance of each feature
    relevances = fraction * R

    shape1 = tf.shape(relevances)
    relevances = tf.Print(relevances, [relevances, shape1], message="Before stich", summarize=1000)
    # Reconstruct the shape of the input, thereby summing the relevances for each individual pixel
    R_new = lrp_util.patches_to_images_for_max_pool(relevances, batch, input_height, input_width, input_channels, output_height,
                                       output_width, kernel_size[1], kernel_size[2], strides[1], strides[2], padding)


    shape2 = tf.shape(R_new)
    R_new = tf.Print(R_new, [R_new, shape2, kernel_size, strides], message="After stich", summarize=1000)
    # Recursively find the relevance of the next layer in the network
    return lrp._lrp(max_pool_input, R_new)


