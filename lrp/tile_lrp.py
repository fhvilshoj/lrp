import lrp_util
import tensorflow as tf


def tile(router, R):
    # Sum the relevances
    R = lrp_util.sum_relevances(R)

    # Get the current operation, i.e. the tile operation we are currently handling
    current_operation = router.get_current_operation()

    # Get the input to the tiling operation
    input_to_current_operation = current_operation.inputs[0]

    # Get the size of the 'batch_size' dimension from the input
    input_shape = tf.shape(input_to_current_operation)
    batch_size = input_shape[0]

    # Get the size of the 'predictions_per_sample' dimension from R
    R_shape = tf.shape(R)
    predictions_per_sample = R_shape[1]

    # Find the shape of the input
    input_shape = tf.shape(input_to_current_operation)

    # Concatenate the dimensions to get the new shape of the relevances
    relevances_new_shape = tf.concat([predictions_per_sample, input_shape], 0)


    # Get the tensor that tells how many times the input has been duplicated for each dimension of the input
    copies_per_dimension = tf.cast(current_operation.inputs[1], tf.int32)

    # Get the total number of times the input was duplicated by the tile operation
    number_of_copies = tf.reduce_prod(copies_per_dimension, 0)

    # Get the number of dimensions of the input
    rank_input = tf.size(copies_per_dimension)

    # Transpose R from shape (batch_size, predictions_per_sample, ....) to shape
    # (predictions_per_sample, batch_size, ...) since the predictions_per_sample dimensions is left untouched in
    # the while loop below
    first_two_axes = tf.constant([1, 0])
    remaining_axes = tf.range(2, rank_input + 1)
    perm = tf.concat([first_two_axes, remaining_axes], 0)
    R = tf.transpose(R, perm)

    def _method(T, d):
        lol = 1
        def _return_T(T):
            return T

        def _iterate_elements_in_ta(T, d):
            ta = tf.TensorArray(dtype=tf.float32, size=copies_per_dimension[d])
            ta.split(T, [input_shape[d]] * copies_per_dimension[d])

            d = tf.add(d, 1)
            zeroes = tf.zeros(relevances_new_shape)

            def _add(idx, sum, ta):
                elem = ta.read(idx)
                tensor_sum = _method(elem, d)
                new_sum = tf.add(sum, tensor_sum)
                return tf.add(idx, 1), new_sum, ta

            _, sum, _ = tf.while_loop(
                cond=lambda idx, _: tf.less(idx, ta.size()),
                body=_add,
                loop_vars=[0, zeroes, ta]
            )

            return sum

        T = tf.cond(tf.less_equal(d, rank_input), true_fn=lambda : _iterate_elements_in_ta(T, d),
                    false_fn=lambda : _return_T(T, d))

        return T

    R_new = _method(R, 1)

    # Transpose R back from shape (predictions_per_sample, batch_size, ...)
    # to shape (batch_size, predictions_per_sample, ....)
    first_two_axes = tf.constant([0, 1])
    remaining_axes = tf.range(2, rank_input + 1)
    perm = tf.concat([first_two_axes, remaining_axes], 0)
    R = tf.transpose(R_new, perm)

    # Mark the tiling operation as handled
    router.mark_operation_handled(current_operation)

    # Forward the relevance to input to the tile operation
    router.forward_relevance_to_operation(R, current_operation, input_to_current_operation.op)
