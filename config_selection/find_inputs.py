import tensorflow as tf
import numpy as np
import argparse
import re
from os import listdir, makedirs
from os.path import isfile, join, exists
from config_selection.feature_parser import FeatureParser

# Disable tf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

dir_name_re = r'(.*\/).*'
single_feature_size = 4512
context_size = 123
batch_size = 1

# File name
def _create_dir_if_not_exists(destination_file):
    if exists(destination_file):
        raise ValueError("{} already exists".format(destination_file))
    else:
        dir = re.sub(dir_name_re, r'\1', destination_file)
        if not exists(dir):
            makedirs(dir)
        if not destination_file.endswith(".npy"):
            destination_file = re.sub(r'(.*)\..*', r'\1.npy', destination_file)
        return destination_file

def _read_sparse_tensor(spt):
    with open(spt, 'r') as s:
        shape = [int(i) for i in s.readline().split()]
        s.readline()
        indices = []
        for i in range(len(shape)):
            dimension = [int(d) for d in s.readline().split()]
            indices.append(dimension)
        indices = np.array(indices)
        indices = indices.transpose()
        s.readline()
        values = np.array([float(d) for d in s.readline().split()])
        return indices, values, shape
        

def _read_sparse_tensors(spt):
    sparse_indices = []
    sparse_values = []
    sparse_shapes = []
    
    for sp in spt:
        i, v, s = _read_sparse_tensor(sp)
        first_dim = i.transpose().reshape(-1)[:len(i)]
        
        for sample in range(s[0]):
            selection = np.where(first_dim == sample)
            selected_indices = i[selection]
            selected_indices[:,0] = 0
            sparse_indices.append(selected_indices)
            sparse_values.append(v[selection])
            sparse_shapes.append(np.concatenate([[1], s[1:]], axis=0))
    return sparse_indices, sparse_values, sparse_shapes


def _search_for_tensors(indices, values, shapes, feature_file, **kwargs):

    with tf.Graph().as_default():
        f_parser = FeatureParser(feature_file, single_feature_size, context_size, batch_size)
        next_batch = f_parser.next_batch()
        found_tensors = []
        permutation = []
        with tf.Session() as s:
            s.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=s, coord=coord)

            sum = 0
            i = 0
            found = 0
            while f_parser.has_next():
                r = s.run(next_batch)
                labels = r['label'].transpose()
                sum += np.sum(labels)

                read_vals = r['features'].values
                for j, val in enumerate(values):
                    if len(val) == len(read_vals) and np.allclose(val, read_vals):
                        print("Found sample at index {}".format(i))
                        r['permutation'] = j
                        found_tensors.append(r)
                        found += 1

                if found == len(values):
                    break                
                i += 1
                f_parser.did_read_batch()

            coord.request_stop()
            coord.join()
            
            found_tensors = found_tensors[::-1]
            found_tensors.sort(key=lambda x: x['permutation'])
           
            print("Found {} matching".format(found))
            print(found_tensors[0]['forloeb'])
            return found_tensors

def _write_found_tensors_to_file(found_tensors, destination):
    print("Writing tensors into %s \n -- In sorted order" % destination)
    np.save(destination, found_tensors)
    

def main():
    parser = argparse.ArgumentParser(description='Use this tool to read .spt files and search .txt files for equal tensors')
    parser.add_argument('-f', '--feature-file', type=str, nargs=1,
                        default='E:/frederikogbenjamin/sepsis/data/TODO')
    parser.add_argument('-s', '--spt', type=str, nargs='+',
                        default=['E:/fraderikogbenjamin/benchmarks/TODO'])
    parser.add_argument('-d', '--destination', type=str,
                        default='E:/frederikogbenjamin/benchmarks/pickles/TODO')

    args = parser.parse_args()

    dest = _create_dir_if_not_exists(args.destination)

    si, sv, ss = _read_sparse_tensors(args.spt)

    found_tensors = _search_for_tensors(si, sv, ss, **vars(args))

    _write_found_tensors_to_file(found_tensors, dest)

if __name__ == '__main__':
    main()
