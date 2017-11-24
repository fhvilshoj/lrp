from configuration import LAYER, RULE, LRPConfiguration, EpsilonConfiguration, BIAS_STRATEGY
from lrp.linear_lrp import linear_with_config
from lrp import lrp_util
import tensorflow as tf
from lrp.constants import EPSILON

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

    # Extract information about R for later reshapes
    R_shape = tf.shape(R)
    predictions_per_sample = R_shape[1]

    batch_size_times_predictions_per_sample = batch_size * predictions_per_sample

    # (_, output_height, output_width, output_channels) = current_tensor.get_shape().as_list()

    # Extract every patch of the input (i.e. portion of the input that the kernel looks at a time)
    # Shape of image_patches: (batch, out_height, out_width, kernel_height*kernel_width*input_channels)
    image_patches = tf.extract_image_patches(max_pool_input, kernel_size,
                                             strides, [1, 1, 1, 1], padding)
    def _winners_takes_all():
        # Reshape image patches to "small images" instead of lists
        # Shape of image_patches after reshape:
        # (batch_size, out_height, out_width, kernel_height, kernel_width, input_channels)
        image_patches_reshaped = tf.reshape(image_patches, [batch_size, output_height, output_width, kernel_size[1], kernel_size[2], input_channels])

        # Find the largest elements in each patch and set all other entries to zero (to find z_ijk+'s)
        # Shape of max_elems: (batch_size, out_height, out_width, 1, 1, input_channels)
        max_elems = tf.reshape(current_tensor, (batch_size, output_height, output_width, 1, 1, input_channels))

        # Select maximum in each patch and set all others to zero
        # Shape of zs: (batch_size, out_height, out_width, kernel_height, kernel_width, input_channels)
        zs = tf.where(tf.equal(image_patches_reshaped, max_elems), tf.ones_like(image_patches_reshaped), tf.zeros_like(image_patches_reshaped))

        # Count how many zijs had the maximum value for each patch
        max_counts = tf.reduce_sum(zs, axis=[3, 4], keep_dims=True)

        # Find the contribution of each feature in the input to the activations,
        # i.e. the ratio between the z_ijk's and the z_jk's
        # Shape of fractions: (batch_size, out_height, out_width, kernel_height, kernel_width, input_channels)
        fractions = zs / max_counts

        # Add the predictions_per_sample dimension to be able to broadcast fractions over the different
        # predictions for the same sample
        # Shape after expand_dims:
        # (batch_size, predictions_per_sample=1, out_height, out_width, kernel_height, kernel_width, input_channels)
        fractions = tf.expand_dims(fractions, 1)

        # Put the relevance for each patch in the dimension that corresponds to the "input_channel" dimension
        # of the fractions
        # Shape of R after reshape: (batch_size, predictions_per_sample, out_height, out_width, 1, 1, out_channels)
        R_distributed = tf.reshape(R, [batch_size, predictions_per_sample, output_height, output_width, 1, 1, output_channels])

        # Distribute the relevance onto athe fractions
        # Shape of new relevances: (batch_size, predictions_per_sample, out_height, out_width, kernel_height, kernel_width, input_channels)
        return fractions * R_distributed


    def _distribute_relevances():
        linear_input_columns = kernel_size[1] * kernel_size[2] * input_channels
        linear_input = tf.reshape(image_patches, (batch_size * output_height * output_width, linear_input_columns))
        linear_filter = tf.ones((linear_input_columns, 1), dtype=tf.float32)

        # Make transpose order (0, 2, .. , 1, last_dim)
        # This in necessary because for conv1d the output might have been expanded which
        # makes the output size partially unknown
        transpose_order = tf.concat([[0], tf.range(2, tf.size(R_shape) - 1), [1], [tf.size(R_shape) - 1]], 0)
        linear_R = tf.transpose(R, transpose_order)

        # Reshape linear_R to have three dimensions
        linear_R = tf.reshape(linear_R,
                              (batch_size * output_height * output_width, predictions_per_sample, output_channels))

        # Construct synthetic configuration
        config = EpsilonConfiguration(bias_strategy=BIAS_STRATEGY.IGNORE)

        # Pass the responsibility to linear_lrp
        # Shape of linear_R_new:
        # (batch_size * out_height * out_width, predictions_per_sample, filter_height * filter_width * input_channels)
        linear_R_new = linear_with_config(linear_R, linear_input, linear_filter, config, bias=None)

        # Shape back to be able to restitch
        linear_R_new = tf.reshape(linear_R_new, (batch_size, output_height, output_width,
                                                 predictions_per_sample, linear_input_columns))

        # Transpose back to be able to restitch
        # New shape:
        # (batch_size, predictions_per_sample, out_height, out_width, filter_height * filter_width * input_channels)
        linear_R_new = tf.transpose(linear_R_new, [0, 3, 1, 2, 4])

        # Gather batch_size and predictions_per_sample
        # New shape:
        # (batch_size * predictions_per_sample, out_height, out_width, filter_height * filter_width * input_channels)
        linear_R_new = tf.reshape(linear_R_new,
                                  (batch_size * predictions_per_sample,
                                   output_height, output_width, linear_input_columns))

        return linear_R_new

    config = router.get_configuration(LAYER.MAX_POOOING)
    if config.type == RULE.WINNER_TAKE_ALL:
        relevances = _winners_takes_all()
    else:
        relevances = _distribute_relevances()

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


