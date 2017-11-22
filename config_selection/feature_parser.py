import tensorflow as tf
from data_loader import sparse_merge


class FeatureParser(object):
    def __init__(self, feature_file, single_feature_size, context_size, batch_size):
        self.feature_file = feature_file
        self.batch_size = batch_size

        self.record_count = None
        self.get_record_count()

        self.records_read = 0

        # Create a queue with the input images
        self.queue = tf.train.string_input_producer(self.feature_file, num_epochs=self.record_count, shuffle=False)
        self.feature_size = single_feature_size
        self.context_size = context_size
        self.reader = tf.TFRecordReader(options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))
        _, self.serialized_example = self.reader.read(self.queue)

    def has_next(self):
        return self.records_read + self.batch_size < self.get_record_count()

    def next_batch(self):
        batch = tf.train.batch([self.serialized_example], self.batch_size)

        features = tf.parse_example(batch, features={
            'fea_indices_row': tf.VarLenFeature(tf.int64),
            'fea_indices_col': tf.VarLenFeature(tf.int64),
            'fea_values': tf.VarLenFeature(tf.float32),
            'seq_len': tf.FixedLenFeature([1], tf.int64),
            'label': tf.FixedLenFeature([1], tf.int64),
            'forloeb': tf.FixedLenFeature([1], tf.int64),
            'ctx_indices': tf.VarLenFeature(tf.int64),
            'ctx_values': tf.VarLenFeature(tf.float32),
            'text_indices_row': tf.VarLenFeature(tf.int64),
            'text_indices_col': tf.VarLenFeature(tf.int64),
            'text_indices_vals': tf.VarLenFeature(tf.int64),
            'text_len_idx': tf.VarLenFeature(tf.int64),
            'text_len': tf.VarLenFeature(tf.int64),
            'doc_ids': tf.VarLenFeature(tf.int64)
        })

        seq_len = tf.sparse_reduce_max(features['fea_indices_row'], axis=1) + 10
        max_seq_len = tf.reduce_max(
            tf.stack([
                tf.reduce_max(seq_len) - 1,
                tf.reduce_max(features['text_indices_row'].values)
            ])) + 1

        fea_dim = [max_seq_len, self.feature_size]

        # Read values and construct sparse tensor
        sparse_indices_row = features['fea_indices_row']
        sparse_indices_col = features['fea_indices_col']
        sparse_vals = features['fea_values']

        feature_tensor = sparse_merge.sparse_merge([sparse_indices_row, sparse_indices_col], sparse_vals, fea_dim)

        text_indices_row = features['text_indices_row']
        text_indices_col = features['text_indices_col']
        text_indices_vals = features['text_indices_vals']
        max_text_len = tf.cond(tf.equal(text_indices_col.dense_shape[1], tf.constant(0, tf.int64)),
                               lambda: tf.constant(0, tf.int64),
                               lambda: tf.add(tf.reduce_max(text_indices_col.values), 1))
        text_tensor = sparse_merge.sparse_merge([text_indices_row, text_indices_col], text_indices_vals,
                                                [max_seq_len, max_text_len])

        text_len_tensor = sparse_merge.sparse_merge([features['text_len_idx']], features['text_len'], [max_seq_len])
        doc_ids_tensor = sparse_merge.sparse_merge([features['text_len_idx']], features['doc_ids'], [max_seq_len])

        ctx_tensor = tf.sparse_merge(features['ctx_indices'], features['ctx_values'], self.context_size)

        return {
            'features': feature_tensor,
            'context': ctx_tensor,
            'seq_len': seq_len,
            'label': features['label'],
            'forloeb': features['forloeb'],
            'text': text_tensor,
            'text_len': text_len_tensor,
            'doc_ids': doc_ids_tensor
        }

    def get_record_count(self):
        """
        Counts the number of records in tf_record_file
        """
        if self.record_count is not None:
            return self.record_count

        count = 0
        print(self.feature_file)
        for file in self.feature_file:
            for _ in tf.python_io.tf_record_iterator(file, options=tf.python_io.TFRecordOptions(
                    tf.python_io.TFRecordCompressionType.GZIP)):
                count += 1
        self.record_count = count
        print(self.record_count)
        return self.record_count

    def samples_read(self):
        return self.records_read

    def did_read_sample(self):
        self.records_read += self.batch_size
