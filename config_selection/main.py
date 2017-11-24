# Config
from config_selection import logger
from config_selection.feature_parser import FeatureParser
from config_selection.result_file_writer import ResultWriter
from config_selection.pertubation import Pertuber
from config_selection.random_permutation import get_random_relevance
from config_selection.sensitivity_analysis import get_sensitivity_analysis
from config_selection.configurations import get_configurations

# LRP
from lrp import lrp

# Sepsis
import sirs_classifier

# External
import argparse
import tensorflow as tf
import numpy as np
from tabulate import tabulate

# The size of the input and the number of classes

INPUT_SIZE = 4512
CONTEXT_SIZE = 123
NUM_CLASSES = 10

# The number of epochs and the size of each mini batch
NUM_EPOCHS = None

# Don't use autoencoder, text or doc_vec since we are dealing with images
USE_AUTOENCODER = False
USE_TEXT = False
USE_DOC2VEC = False

# The model to restore from
SAVED_MODEL = "MNIST_trained"

class ConfigSelection(object):
    def __init__(self, input_features, model, destination, batch_size, **kwargs):
        # Remember the batch_size
        self.batch_size = batch_size
        self.confs = kwargs

        # Remember the file containing the model
        self.model_file = model

        # Dict used to holde the input elements (as tensors)
        self.features_batch = {
            'features': None,
            'context': None,
            'seq_len': None,
            'label:': None,
            'forloeb': None,
        }

        # Result writer used to append benchmark results to files according to destination directory
        self.writer = ResultWriter(destination)

        # Gets all the different configurations
        self.configurations = get_configurations()

        # Count the configurations and add sensitivity analysis and random
        self.num_configurations = len(self.configurations) + 2

        # Prepare separate graph for reading input
        self.input_graph = tf.Graph()
        with self.input_graph.as_default():
            self.parser = FeatureParser(input_features, INPUT_SIZE, CONTEXT_SIZE, batch_size)
            self.next_batch = self.parser.next_batch()

            # Required variable in order to use tf.make_template()
            global_step = tf.Variable(0, name='global_step', trainable=False)

            logger.debug('Making sirs model for the input')

            self.input_model = sirs_classifier.create_model(None, global_step, self.next_batch, False, INPUT_SIZE, CONTEXT_SIZE, NUM_CLASSES,
                                         USE_TEXT, use_char_autoencoder=USE_AUTOENCODER, use_doc2vec=USE_DOC2VEC)

            y_hat = self.input_model['y_hat']
            y_maxes = tf.reduce_max(y_hat, axis=2)
            y_argmax = tf.cast(tf.argmax(y_maxes, axis=1), tf.int32)

            self.input_model['offsets'] = tf.pad(tf.expand_dims(y_argmax, axis=1), [[0, 0], [3, 1]])

            self.input_session = tf.Session(graph=self.input_graph)
            self.input_session.run([tf.local_variables_initializer()])

            # Create a tf Saver that can be used to restore a pre-trained model below
            saver = tf.train.Saver()
            self.restore_checkpoint(self.input_session, saver)

        logger.info("Testing {} samples from {}".format(self.parser.get_record_count(), input_features))
        logger.info("Writing results to folder: {}".format(destination))
        logger.info("Model used: {}".format(model))
        logger.info("Batch size: {}".format(batch_size))

    def __call__(self, *args, **kwargs):
        # This is the main entrance to run configuration testing
        logger.debug("In the __call__ function")

        # Read through all samples in the input file
        while self.parser.has_next():
            logger.info('Starting new sample')

            # Read input from file
            self._read_next_input()

            batch_number = self.parser.samples_read() // self.batch_size

            logger.info('Testing configuration {}/{} for batch {}'.format(1, self.num_configurations, batch_number))
            self._test_configuration("random")

            logger.info('Testing configuration {}/{} for batch {}'.format(2, self.num_configurations, batch_number))
            self._test_configuration("sensitivity_analysis")

            # Run test for each configuration in the configuration list.
            for idx, config in enumerate(self.configurations[:2]):
                logger.info('Testing configuration {}/{} for batch {}'.format(idx + 3, self.num_configurations, batch_number))
                self._test_configuration(config)

        logger.info("Done testing all of the input to the given file")

    def _read_next_input(self):
        logger.debug("Starting input session")

        # Start queue runners for the reader queue to work
        tf.train.start_queue_runners(sess=self.input_session)

        # Read the next input
        self.features_read, self.input_selection, y_hat = self.input_session.run([self.next_batch,
                                                                                  self.input_model['offsets'],
                                                                                  self.input_model['y_hat']])

        # Reduce max sequence length
        self.max_sequence = y_hat.shape[1]

        # Tell the parser that we read a batch
        self.parser.did_read_batch()

        logger.info("Read sample {} with label \n{}".format(self.parser.samples_read(), self.features_read['label']))


    def _test_configuration(self, config):
        # Start new graph for this configuration
        graph = tf.Graph()
        with graph.as_default():

            logger.debug("Start of new test graph with config {}".format(config))

            # Dictionary to hold all the 'feed_dict' parameters for session.run
            to_feed = dict()

            # Construct sparse tensors from placeholders
            # X read from the input (this is a SparseTensorValue , i.e. numpy like)
            X_read = self.features_read['features']

            # Placeholders to reconstruct X in this new graph
            X_indices = tf.placeholder(tf.int64, X_read.indices.shape)
            X_values = tf.placeholder(tf.float32, X_read.values.shape)
            X_shape = tf.placeholder(tf.int64, np.size(X_read.dense_shape))

            X = tf.SparseTensor(X_indices, X_values, X_shape)

            # Do sparse reorder to ensure that LRP (and other sparse matrix operations) works
            X_reordered = tf.sparse_reorder(X)

            # Fill actual values into the three placeholders above
            to_feed[X_indices] = X_read.indices
            to_feed[X_values] = X_read.values
            to_feed[X_shape] = X_read.dense_shape

            # Do the same sparse tensor reconstruction trick for the context
            # C read from the input (this is a SparseTensorValue, i.e. numpy like)
            C_read = self.features_read['context']

            # Placeholders to reconstruct C in this new graph
            C_indices = tf.placeholder(tf.int64, C_read.indices.shape)
            C_values = tf.placeholder(tf.float32, C_read.values.shape)
            C_shape = tf.placeholder(tf.int64, np.size(C_read.dense_shape))

            C = tf.SparseTensor(C_indices, C_values, C_shape)

            # Fill actual values into the three placeholders for C
            to_feed[C_indices] = C_read.indices
            to_feed[C_values] = C_read.values
            to_feed[C_shape] = C_read.dense_shape

            # Store sparse context tensor
            self.features_batch['context'] = C

            # Same circus for seq_len
            seq_len = tf.placeholder(tf.int64, (None,))
            self.features_batch['seq_len'] = seq_len

            # Same circus for label
            label = tf.placeholder(tf.int64, self.features_read['label'].shape)
            self.features_batch['label'] = label

            # Same circus for forloeb
            forloeb = tf.placeholder(tf.int64, self.features_read['forloeb'].shape)
            self.features_batch['forloeb'] = forloeb

            # Fill actual values into seq_len, label, forloeb
            to_feed[seq_len] = self.features_read['seq_len']
            to_feed[label] = self.features_read['label']
            to_feed[forloeb] = self.features_read['forloeb']

            # Prepare template (that uses parameter sharing across calls to sirs_template)
            self.first_template_use = True
            sirs_template = tf.make_template('', self.create_model)

            logger.debug("Building graph for forward pass")
            # Compute the DRN graph
            model = sirs_template(X_reordered)

            should_write_input = False
            if isinstance(config, str):
                # The config is either random or SA
                if config == 'random':
                    # Compute random relevances
                    logger.debug("Building random graph")
                    R = get_random_relevance(X)
                    should_write_input = True
                else:
                    # Compute sensitivity analysis
                    logger.debug("Building SA graph")
                    R = get_sensitivity_analysis(X, model['y_hat'])
            else:
                logger.debug('Building lrp graph')
                R = lrp.lrp(X, model['y_hat'], config)
                logger.debug('Done building lrp graph')

            logger.debug("Instantiating pertubation class")
            # Make pertuber for X and R that prepares a number of pertubations of X
            pertuber = Pertuber(X, R, self.batch_size, **self.confs)

            # Build the pertubation graph
            benchmark = pertuber.build_pertubation_graph(sirs_template)

            # Create a tf Saver that can be used to restore a pre-trained model below
            saver = tf.train.Saver()

            with tf.Session(graph=graph) as s:
                logger.debug("Initializing vars and restoring model")
                # Initialize the local variables and restore the model that was trained earlier
                s.run([tf.local_variables_initializer()])
                self.restore_checkpoint(s, saver)

                logger.debug("Restored model. Now starting threads.")

                # Create the threads that run the model
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=s)

                try:
                    # Run the benchmarks. Shapes:
                    # Benchmark_result: batch_size, pertubations, num_classes
                    # y                 batch_size, 1
                    # y_hat             batch_size, num_classes
                    logger.debug("Starting session for benchmarks")
                    benchmark_result, expl, y, y_hat = self.run_model([benchmark, R, model['y'], model['y_hat']],
                                                                model,
                                                                feed_dict=to_feed,
                                                                session=s)
                    logger.debug("Session done")

                    # Remove extra dimension from y
                    # y shape: (batch_size,)
                    y = y[:, 0]

                    # Find argmax for y_hat
                    # y_hat shape: (batch_size,)
                    y_hat = np.argmax(y_hat, axis=1)

                    # Write results to file
                    logger.debug("Writing result to file")
                    self.writer.write_result(config, y, y_hat, benchmark_result)

                    logger.debug("Writing explanation to file")
                    self.writer.write_explanation(config, expl)

                    if should_write_input:
                        logger.debug("Writing input to file")
                        self.writer.write_input(X_read)

                except tf.errors.OutOfRangeError:
                    logger.debug("Done with the testing")
                except KeyboardInterrupt:
                    logger.debug("Process interrupted by user. Wrapping up.")
                finally:
                    coord.request_stop()

                logger.debug("Joining threads")
                coord.join(threads)
                logger.info("Done with test")

    # Helper function that creates the model
    def create_model(self, features):
        # First update the features stored in this class with the new features
        feature_batch = self.features_batch
        feature_batch['features'] = features

        # Run sirs_classifier to get the model for forward passes
        # Ignoring two first parameters (session and global_step) since session is never used and
        # global_step is only for training
        model = sirs_classifier.create_model(None, None, feature_batch, False, INPUT_SIZE, CONTEXT_SIZE,
                                             NUM_CLASSES,
                                             USE_TEXT, use_char_autoencoder=USE_AUTOENCODER, use_doc2vec=USE_DOC2VEC)

        y_hat_reshaped = tf.reshape(model['y_hat'], (1, 1, self.batch_size, self.max_sequence, NUM_CLASSES))

        # Each sample has shape (1, 1, max_sequence, num_classes)
        samples = tf.split(y_hat_reshaped, self.batch_size, 2)

        # each sample has shape (1, 1, 1, num_classes)
        to_concatenate = [tf.slice(sample, max_slice, [1, 1, 1, 1, NUM_CLASSES]) for (sample, max_slice) in zip(samples, self.input_selection)]

        # Shape (1, 1, 1, batch_size * num_classes)
        highest_predictions_concatenated = tf.concat(to_concatenate, axis=4)

        # Remove dimension 0, 1, 2
        model['y_hat'] = tf.reshape(highest_predictions_concatenated, (self.batch_size, NUM_CLASSES))

        return model

    # Helper function that runs the model
    def run_model(self, fetches, model, feed_dict, session):
        static = {
            model['reset_auc']: False,
            model['dropout_keep_prob']: 1.
        }

        # Merge static feed dict witbh feed_dict argument
        to_feed = {**feed_dict, **static}

        # Run the session
        res = session.run(fetches,
                          feed_dict=to_feed)
        return res

    # Helper function that restores the model from a checkpoint
    def restore_checkpoint(self, session, saver):
        saver.restore(session, self.model_file)


# Code so enable commandline execution of the configuration selection
def _main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test LRP configurations')
    parser.add_argument('-i', '--input_features', type=str, nargs='+',
                        default=['E:/frederikogbenjamin/sepsis/data/test.txt_compressed'],
                        help='the location of the input features')
    parser.add_argument('-m', '--model', type=str,
                        default='E:/frederikogbenjamin/sepsis/models/new_model4.ckpt',
                        help='trained model to use')
    parser.add_argument('-d', '--destination', type=str,
                        default='E:/frederikogbenjamin/sepsis/checkpoints/new_model4.ckpt',
                        help='Destination directory')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=10,
                        help='Batch size when testing')
    parser.add_argument('-p', '--pertubations', type=int,
                        default=100,
                        help='Pertubation iterations for each configuration')
    args = parser.parse_args()

    config_select = ConfigSelection(**vars(args))

    # Call config selection with gathered arguments
    config_select()


if __name__ == '__main__':
    _main()
