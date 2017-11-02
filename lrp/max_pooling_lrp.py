from lrp import lrp_util
import tensorflow as tf
from constants import EPSILON

# TODO: We do not currently support max pooling where kernels span over the depts dimension (k: [1,1,1,d])
def max_pooling(router, R):
    """
    Max pooling lrp
    :param router: the router object to report changes to
    :param R: the list of tensors containing the relevances from the upper layers
    """
    # Sum the potentially multiple relevances from the upper layers
    # Shape of R: (batch_size, predictions_per_sample, out_height, out_width, out_depth)
    R = lrp_util.sum_relevances(R)

    # Get current operation from the router
    current_operation = router.get_current_operation()

    # Get the output from the max pool operation
    # Shape of current_tensor: (batch_size, out_height, out_width, out_depth)
    current_tensor = current_operation.outputs[0]

    # Find the input to the max pooling
    # Shape of current_tensor: (batch_size, in_height, in_width, in_depth)
    max_pool_input = current_operation.inputs[0]

    # Find the padding and strides that were used in the max pooling
    padding = current_operation.get_attr("padding").decode("UTF-8")
    strides = current_operation.get_attr("strides")
    kernel_size = current_operation.get_attr("ksize")

    # Get shape of the input
    max_pooling_input_shape = tf.shape(max_pool_input)
    batch_size = max_pooling_input_shape[0]
    input_height = max_pooling_input_shape[1]
    input_width = max_pooling_input_shape[2]
    input_channels = max_pooling_input_shape[3]

    # (batch_size, input_height, input_width, input_channels) = max_pool_input.get_shape().as_list()

    # Get the shape of the output of the max pool operation
    current_tensor_shape = tf.shape(current_tensor)
    output_height = current_tensor_shape[1]
    output_width = current_tensor_shape[2]
    output_channels = current_tensor_shape[3]

    # (_, output_height, output_width, output_channels) = current_tensor.get_shape().as_list()

    # Replace the negative elements with zeroes to only have the positive entries left
    # Shape of max_pool_input_positive: (batch_size, in_height, in_width, in_depth)
    max_pool_input_positive = lrp_util.replace_negatives_with_zeros(max_pool_input)

    # Extract every patch of the input (i.e. portion of the input that the kernel looks at a time)
    # Shape of image_patches: (batch, out_height, out_width, kernel_height*kernel_width*input_channels)
    image_patches = tf.extract_image_patches(max_pool_input_positive, kernel_size,
                                             strides, [1, 1, 1, 1], padding)

    # Reshape image patches to "small images" instead of lists
    # Shape of image_patches after reshape:
    # (batch_size, out_height, out_width, kernel_height, kernel_width, input_channels)
    image_patches = tf.reshape(image_patches, [batch_size, output_height, output_width, kernel_size[1], kernel_size[2], input_channels])

    # Find the largest elements in each patch and set all other entries to zero (to find z_ijk+'s)
    # Shape of max_elems: (batch_size, out_height, out_width, 1, 1, input_channels)
    # max_elems = tf.reduce_max(image_patches, axis=[3, 4], keep_dims=True)
    max_elems = tf.reshape(current_tensor, (batch_size, output_height, output_width, 1, 1, input_channels))

    # Select maximum in each patch and set all others to zero
    # Shape of zs: (batch_size, out_height, out_width, kernel_height, kernel_width, input_channels)
    zs = tf.where(tf.equal(image_patches, max_elems), image_patches, tf.zeros_like(image_patches))

    # Find the contribution of each feature in the input to the activations,
    # i.e. the ratio between the z_ijk's and the z_jk's (plus a small stabilizer to avoid division by zero)
    # TODO We do not take care of two equaly big entries (currently doubles given relevance)
    # Shape of fractions: (batch_size, out_height, out_width, kernel_height, kernel_width, input_channels)
    fractions = zs / (max_elems + EPSILON)

    # Add the predictions_per_sample dimension to be able to broadcast fractions over the different
    # predictions for the same sample
    # Shape after expand_dims:
    # (batch_size, predictions_per_sample=1, out_height, out_width, kernel_height, kernel_width, input_channels)
    fractions = tf.expand_dims(fractions, 1)

    # Extract information about R for later reshapes
    relevances_shape = tf.shape(R)
    predictions_per_sample = relevances_shape[1]
    batch_size_times_predictions_per_sample = batch_size * predictions_per_sample

    # Put the relevance for each patch in the dimension that corresponds to the "input_channel" dimension
    # of the fractions
    # Shape of R after reshape: (batch_size, predictions_per_sample, out_height, out_width, 1, 1, out_channels)
    R = tf.reshape(R, [batch_size, predictions_per_sample, output_height, output_width, 1, 1, output_channels])

    # Distribute the relevance onto athe fractions
    # Shape of new relevances: (batch_size, predictions_per_sample, out_height, out_width, kernel_height, kernel_width, input_channels)
    relevances = fractions * R

    # Put the batch size and predictions_per_sample on the same dimension to be able to use the patches_to_images tool.
    # Also rearrange patches back to lists from the "small images".
    # Shape of relevances after reshape:
    # (batch_size * predictions_per_sample, out_height, out_width, kernel_height * kernel_width * input_channels)
    relevances = tf.reshape(relevances, (batch_size_times_predictions_per_sample, output_height, output_width,
                                         kernel_size[1]*kernel_size[2] * input_channels))

    # Reconstruct the shape of the input, thereby summing the relevances for each individual feature
    R_new = lrp_util.patches_to_images(relevances, batch_size_times_predictions_per_sample, input_height, input_width, input_channels, output_height,
                                       output_width, kernel_size[1], kernel_size[2], strides[1], strides[2], padding)

    # Reshape the relevances back to having batch_size and predictions_per_sample as the first two dimensions
    # (batch_size, predictions_per_sample, input_height, input_width, input_channels) rather than
    # (batch_size * predictions_per_sample, input_height, input_width, input_channels)
    R_new = tf.reshape(R_new, (batch_size, predictions_per_sample, input_height, input_width, input_channels))

    # Report handled operations
    router.mark_operation_handled(current_operation)

    # Forward the calculated relevance to the input of the convolution
    router.forward_relevance_to_operation(R_new, current_operation, max_pool_input.op)


