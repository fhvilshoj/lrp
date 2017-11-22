import tensorflow as tf

from config_selection import logger


class Pertuber(object):
    def __init__(self, X, R, **kwargs):
        self.args = kwargs

        self.X = X
        self.R = R
        self.num_iterations = kwargs['pertubations'] if 'pertubations' in kwargs else 100

        self.X_actives = tf.size(self.X.values)
        self.R_actives = tf.size(self.R.values)

        self._prepare_priority_array()

    # Prevares priority tensor arrays for each sample
    def _prepare_priority_array(self):
        logger.debug("Preparing priority array")

        self.batch_size = self.args['batch_size']

        # Find the active count for each sample by first making an indicator tensor
        indicatores = tf.SparseTensor(self.R.indices, tf.ones_like(self.R.values), self.R.dense_shape)

        # Then summing over everything but the batch dimension
        self.counts = tf.cast(tf.sparse_reduce_sum(indicatores, axis=[1, 2]), dtype=tf.int32)

        # Split indices of R into a tensor array with respect to the sizes of the samples
        indices_ta = tf.TensorArray(tf.int64, self.batch_size, dynamic_size=False, clear_after_read=True,
                                    infer_shape=False, element_shape=(None, 3))
        indices_ta = indices_ta.split(self.R.indices, self.counts)

        # Split values of R into a tensor array with respect to the sizes of the samples
        values_ta = tf.TensorArray(tf.float32, self.batch_size, dynamic_size=False, clear_after_read=True,
                                   infer_shape=False, element_shape=(None,))
        values_ta = values_ta.split(self.R.values, self.counts)

        # Prepare tensorarray to hold ordered indices for each sample
        ordered_indices_ta = tf.TensorArray(tf.int64, self.batch_size, dynamic_size=False, clear_after_read=True,
                                            infer_shape=True)

        def _sort_body(t, ordered_ta):

            # Read indices and values from tensor arrays
            sample_indices = indices_ta.read(t)
            sample_values = values_ta.read(t)

            # Find size of values
            sample_size = tf.size(sample_values)

            # If more values than num_iterations stick with num_iterations
            sample_size = tf.minimum(sample_size, self.num_iterations)

            # Find the ordered top relevances
            _, top_indices = tf.nn.top_k(sample_values, k=sample_size)

            # Store the associated indices to use later
            indices = tf.gather(sample_indices, top_indices)

            def _do_padding():
                to_pad = self.num_iterations - sample_size
                padding = tf.tile(tf.constant([[-1, -1, -1]], dtype=tf.int64), [to_pad, 1])
                return tf.concat([indices, padding], axis=0)

            def _skip_padding():
                return indices

            # Indices shape: (num_iterations, 3)
            indices = tf.cond(tf.less(sample_size, self.num_iterations),
                              true_fn=_do_padding,
                              false_fn=_skip_padding)

            ordered_ta = ordered_ta.write(t, indices)

            return t + 1, ordered_ta

        _, ordered_indices_ta = tf.while_loop(
            cond=lambda t, _: tf.less(t, self.batch_size),
            body=_sort_body,
            loop_vars=[0, ordered_indices_ta]
        )

        # Ordered_indices_stackes shape: (batch_size, num_iterations, 3)
        ordered_indices_stacked = ordered_indices_ta.stack()

        # Transpose to group respectively the most important, next most important, etc. together
        ordered_indices_transposed = tf.transpose(ordered_indices_stacked, [1, 0, 2])

        # Reshape to have shape (batch_size * num_iterations, 3)
        self.priority_indices = tf.reshape(ordered_indices_transposed, (self.batch_size * self.num_iterations, 3))

    def build_pertubation_graph(self, template):
        logger.debug("Building pertubation_graph")
        results_ta = tf.TensorArray(tf.float32, self.num_iterations + 1)
        priority_indices_ta = tf.TensorArray(tf.int64, self.batch_size * self.num_iterations,
                                          dynamic_size=False,
                                          clear_after_read=True).unstack(self.priority_indices)

        def _pertubation_loop_body(i, X_values, res_ta):
            #### Do and record forward pass
            X = tf.SparseTensor(self.X.indices, X_values, self.X.dense_shape)
            model = template(X)
            res_ta = res_ta.write(i, model['y_hat'])

            def _prepare_next_iteration():
                # Loop for pertubing batch_size times
                def _pertubation_body(j, current_selection):
                    ####  Do pertubation
                    # Find index to set to 0

                    def _select_index():
                        idx = i * self.batch_size + j
                        index = priority_indices_ta.read(idx)

                        # Make selection from index and indices in X by first finding all partial matches with the index:
                        # Selection shape: (sequence_length, 3)
                        selection = tf.where(tf.equal(X.indices, index), tf.ones_like(X.indices), tf.zeros_like(X.indices))

                        # Next sum matches to se if we can get a sum of 3 indicating that all three dimensions, i.e.
                        # (sample, time_step, feature) are matching.
                        # selection shape: (sequence_length,)
                        selection = tf.reduce_sum(selection, axis=1)

                        # Filter out that one index to remove by setting that index to 0
                        selection = tf.where(tf.equal(selection, 3),
                                             tf.ones_like(selection, dtype=tf.float32),
                                             tf.zeros_like(selection, dtype=tf.float32))
                        return selection

                    def _skip_index():
                        skipping = tf.zeros_like(current_selection)
                        return skipping

                    # Only select index if there are more values in the given sample
                    selection = tf.cond(
                        tf.less(i, self.counts[j]),
                        _select_index,
                        _skip_index
                    )

                    # Used the filter to multiply elementwise between old values and the filter
                    return j + 1, current_selection + selection

                _, selection = tf.while_loop(
                    cond=lambda j, _: tf.less(j, self.batch_size),
                    body=_pertubation_body,
                    loop_vars=[0, tf.zeros_like(X_values)]
                )

                # Invert selection to be able to multiply it with the X_values
                selection = tf.where(tf.equal(selection, 0),
                                     tf.ones_like(selection),
                                     tf.zeros_like(selection))


                # Remove selected values
                return X_values * selection

            def _last_iteration():
                # The return of this is never used so it could be whatever
                return X_values

            X_values_new = tf.cond(tf.equal(i, self.num_iterations),
                                   true_fn=_last_iteration,
                                   false_fn=_prepare_next_iteration)

            return i + 1, X_values_new, res_ta

        *_, results_ta = tf.while_loop(
            cond=lambda t, *_: tf.less_equal(t, self.num_iterations),
            body=_pertubation_loop_body,
            loop_vars=[0, self.X.values, results_ta]
        )

        # Stack results to a tensor
        # shape: (num_iterations, batch_size, 2)
        range = tf.range(0, self.num_iterations + 1)
        results = results_ta.gather(range)

        # Transpose result to shape (batch_size, num_iterations, num_classes)
        result = tf.transpose(results, [1, 0, 2])
        return result
