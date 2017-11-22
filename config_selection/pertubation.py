import tensorflow as tf

from config_selection import logger

class Pertuber(object):
    def __init__(self, X, R, **kwargs):

        self.X = X
        self.R = R
        self.num_iterations = kwargs['pertubations'] if 'pertubations' in kwargs else 100

        self.X_actives = tf.size(self.X.values)
        self.R_actives = tf.size(self.R.values)

        self._prepare_priority_array()

    # Prevares priority tensor arrays for each sample
    def _prepare_priority_array(self):
        logger.debug("Preparing priority array")

        # Find number of pertubations to do
        active_indices = tf.size(self.R.values)
        self.pertubations = tf.minimum(active_indices, self.num_iterations)

        # Prepare a tensorarray with elements condaining the ordered indices
        # of the most relevant features
        ta = tf.TensorArray(tf.int64, self.pertubations, clear_after_read=True, infer_shape=False, element_shape=(3,))

        # Find the ordered top relevances
        _, top_indices = tf.nn.top_k(self.R.values, k=self.pertubations)

        # Store the associated indices to use later
        indices = tf.gather(self.R.indices, top_indices)
        self.priority_indices = ta.unstack(indices)

    def build_pertubation_graph(self, template):
        logger.debug("Building pertubation_graph")
        results_ta = tf.TensorArray(tf.float32, self.num_iterations + 1)

        def _pertubation_loop_body(t, X_values, res_ta):
            #### Do and record forward pass
            # res shape: (2,)
            X = tf.SparseTensor(self.X.indices, X_values, self.X.dense_shape)
            model = template(X)
            res_ta = res_ta.write(t, model['y_hat'])

            def _prepare_next_X():
                ####  Do pertubation
                # Find index to set to 0
                index = self.priority_indices.read(t)

                # Make selection from index and indices in X by first finding all partial matches with the index:
                # Selection shape: (sequence_length, 3)
                selection = tf.where(tf.equal(X.indices, index), tf.ones_like(X.indices), tf.zeros_like(X.indices))

                # Next sum matches to se if we can get a sum of 3 indicating that all three dimensions, i.e.
                # (sample, time_step, feature) are matching.
                # selection shape: (sequence_length,)
                selection = tf.reduce_sum(selection, axis=1)

                # Filter out that one index to remove by setting that index to 0
                selection = tf.where(tf.equal(selection, 3),
                                     tf.zeros_like(selection, dtype=tf.float32),
                                     tf.ones_like(selection, dtype=tf.float32))

                # Used the filter to multiply elementwise between old values and the filter
                return X.values * selection

            def _last_iteration():
                return X.values

            X_values_new = tf.cond(tf.less(t, self.pertubations),
                            true_fn=_prepare_next_X,
                            false_fn=_last_iteration)

            return t + 1, X_values_new, res_ta

        *_, results_ta = tf.while_loop(
            cond=lambda t, *_: tf.less_equal(t, self.pertubations),
            body=_pertubation_loop_body,
            loop_vars=[0, self.X.values, results_ta]
        )

        # Stack results to a tensor
        # shape: (pertubations, 2)
        results = results_ta.gather(tf.range(0, self.pertubations + 1))
        result = tf.squeeze(results, 1)
        result = tf.transpose(result)
        return result




