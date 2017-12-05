import tensorflow as tf
import numpy as np

class NpyFeatureParser(object):
    def __init__(self, feature_file, batch_size, *args, **kwargs):
        self.tensors = np.load(feature_file[0])
        self.batch_size = batch_size
        self.tensors = self._merge_tensors_to_batch_size(self.tensors, self.batch_size)
        self.record_count = len(self.tensors)
        self.records_read = 0

    def _merge_models(self, m1, m2):
        sparse_merges = ['context', 'features']
        dense_merges = ['seq_len', 'label', 'forloeb']
        ignores = ['test', 'doc_ids', 'text_len']

        new_model = {}
        
        for key in sparse_merges:
            new_shape = np.concatenate([[m1[key].dense_shape[0] + m2[key].dense_shape[0]], m1[key].dense_shape[1:]], axis=0)
            new_indices = m1[key].indices
            to_append = m2[key].indices
            to_append[:,0] += m1[key].dense_shape[0]
            new_indices = np.concatenate([new_indices, to_append], axis=0)
            new_values = np.concatenate([m1[key].values, m2[key].values], axis=0)
            
            new_model[key] = tf.SparseTensorValue(new_indices, new_values, new_shape)
        for key in dense_merges:
            new_model[key] = np.concatenate([m1[key], m2[key]], axis=0)
        for key in ignores:
            new_model[key] = None

        return new_model

    def _merge_tensors_to_batch_size(self, tensors, batch_size):
        new_tensors = []

        current_tensor = tensors[0]
        current_batch_size = 1
        for t in tensors[1:]:
            if current_batch_size == batch_size:
                new_tensors.append(current_tensor)
                current_batch_size = 1
                current_tensor = t
            else:
                current_tensor = self._merge_models(current_tensor, t)
                current_batch_size += 1

        new_tensors.append(current_tensor)
        return new_tensors

    def has_next(self):
        return self.records_read < self.record_count

    def next_batch(self):
        to_read = self.records_read
        self.records_read += 1
        return self.tensors[to_read]

    def get_record_count(self):
        return self.record_count
    
    def samples_read(self):
        return self.records_read

    def did_read_batch(self):
        pass

if __name__ == '__main__':
    fp = NpyFeatureParser(['./to_remove/test.npy'], 3)
    
