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
MINIBATCH_SIZE = 1

# Don't use autoencoder, text or doc_vec since we are dealing with images
USE_AUTOENCODER = False
USE_TEXT = False
USE_DOC2VEC = False

# The model to restore from
SAVED_MODEL = "MNIST_trained"

class ConfigSelection(object):
    def __init__(self, input_features, model, destination):

        self.input_features_file = input_features
        self.model_file = model if not isinstance(model, list) else model[0]

        self.features_batch = {
            'features': None,
            'context': None,
            'seq_len': None,
            'label:': None,
            'forloeb': None,
        }

        self.writer = ResultWriter(*destination)

        self.configurations = get_configurations()
        self.num_configurations = len(self.configurations) + 2 # add random and SA

        self.input_graph = tf.Graph()
        with self.input_graph.as_default():
            self.parser = FeatureParser(input_features, INPUT_SIZE, CONTEXT_SIZE)
            self.next_batch = self.parser.next_batch()
            self.input_session = tf.Session(graph=self.input_graph)
            self.input_session.run([tf.local_variables_initializer()])

            logger.info("Testing {} samples from {}".format(self.parser.get_record_count(), self.input_features_file))

    def __call__(self, *args, **kwargs):
        logger.debug("In the __call__ function")

        while self.parser.has_next() and self.parser.samples_read() < 2000:
            logger.info('Starting new sample')
            # Read input from file
            self._read_next_input()
            self.parser.did_read_sample()

            logger.info('Testing configuration {}/{}'.format(1, self.num_configurations))
            self._test_configuration("random")
            logger.info('Testing configuration {}/{}'.format(2, self.num_configurations))
            self._test_configuration("sensitivity_analysis")

            for idx, config in enumerate(self.configurations):
                logger.info('Testing configuration {}/{}'.format(idx + 3, self.num_configurations))
                self._test_configuration(config)

        logger.info("Done testing all of the input to the given file")

    def _read_next_input(self):
        logger.debug("Starting input session")

        tf.train.start_queue_runners(sess=self.input_session)
        self.features_read = self.input_session.run(self.next_batch)

        logger.info("Read sample {} with label {}".format(self.parser.samples_read(), self.features_read['label']))

    def _test_configuration(self, config):
        graph = tf.Graph()
        with graph.as_default():
            logger.debug("Start of new test graph with config {}".format(config))
            to_feed = dict()

            # Construct sparse tensors from placeholders
            # X
            X_read = self.features_read['features']
            X_indices = tf.placeholder(tf.int64, X_read.indices.shape)
            X_values = tf.placeholder(tf.float32, X_read.values.shape)
            X_shape = tf.placeholder(tf.int64, np.size(X_read.dense_shape))

            X = tf.SparseTensor(X_indices, X_values, X_shape)
            X_reordered = tf.sparse_reorder(X)

            to_feed[X_indices] = X_read.indices
            to_feed[X_values] = X_read.values
            to_feed[X_shape] = X_read.dense_shape

            self.features_batch['features'] = X_reordered

            # C
            C_read = self.features_read['context']
            C_indices = tf.placeholder(tf.int64, C_read.indices.shape)
            C_values = tf.placeholder(tf.float32, C_read.values.shape)
            C_shape = tf.placeholder(tf.int64, np.size(C_read.dense_shape))
            C = tf.SparseTensor(C_indices, C_values, C_shape)

            to_feed[C_indices] = C_read.indices
            to_feed[C_values] = C_read.values
            to_feed[C_shape] = C_read.dense_shape

            self.features_batch['context'] = C

            seq_len = tf.placeholder(tf.int64, (None,))
            self.features_batch['seq_len'] = seq_len

            label = tf.placeholder(tf.int64, self.features_read['label'].shape)
            self.features_batch['label'] = label

            forloeb = tf.placeholder(tf.int64, self.features_read['forloeb'].shape)
            self.features_batch['forloeb'] = forloeb

            to_feed[seq_len] = self.features_read['seq_len']
            to_feed[label] = self.features_read['label']
            to_feed[forloeb] = self.features_read['forloeb']

            sirs_template = tf.make_template('', self.create_model)
            model = sirs_template(X_reordered)

            if isinstance(config, str):
                # The config is either random or SA
                if config == 'random':
                    R = get_random_relevance(X)
                else:
                    R = get_sensitivity_analysis(X, model['y_hat'])
            else:
                logger.debug('building lrp graph')
                R = lrp.lrp(X, model['y_hat'], config)
                logger.debug('done building lrp graph')

            pertuber = Pertuber(X, R, 10)
            benchmark = pertuber.build_pertubation_graph(sirs_template)

            # Create a tf Saver that can be used to restore a pre-trained model below
            saver = tf.train.Saver()

            logger.debug("Starting session")

            with tf.Session(graph=graph) as s:
                # Initialize the local variables and restore the model that was trained earlier
                s.run([tf.local_variables_initializer()])
                self.restore_checkpoint(saver, self.model_file)

                logger.debug("Restored model. Now running graph.")

                # Create the threads that run the model
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=s)

                try:
                    benchmark_result, expl, y, y_hat = self.run_model([benchmark, R, model['y'], model['y_hat']],
                                                                model,
                                                                feed_dict=to_feed,
                                                                session=s)
                    y = np.squeeze(y)
                    y_hat = np.squeeze(np.argmax(y_hat, axis=1))

                    self.writer.write_result(config, y, y_hat, benchmark_result)

                    # TODO write explanation to a file

                except tf.errors.OutOfRangeError:
                    logger.debug("Done with the testing")
                except KeyboardInterrupt:
                    logger.debug("Process interrupted by user. Wrapping up.")
                finally:
                    coord.request_stop()

                coord.join(threads)

                logger.debug("Done with test")

    # Helper function that creates the model
    def create_model(self, features):

        feature_batch = self.features_batch
        feature_batch['features'] = features

        # Ignoring two first parameters (session and global_step) since session is never used and
        # global_step is only for training
        model = sirs_classifier.create_model(None, None, feature_batch, False, INPUT_SIZE, CONTEXT_SIZE,
                                             NUM_CLASSES,
                                             USE_TEXT, use_char_autoencoder=USE_AUTOENCODER, use_doc2vec=USE_DOC2VEC)

        # Shape: (batch_size, sequence_length, num_classes)
        y_hat = model['y_hat']

        # for_explanation shape: (1, 1, num_classes)
        for_explanation = tf.slice(y_hat, [0, tf.shape(y_hat)[1] - 1, 0], [1, 1, NUM_CLASSES])

        # for_explanation shape: (1, num_classes)
        for_explanation = tf.squeeze(for_explanation, axis=1)
        model['y_hat'] = for_explanation

        return model

    # Helper function that runs the model
    def run_model(self, fetches, model, feed_dict, session):
        static = {
            model['reset_auc']: False,
            model['dropout_keep_prob']: 1.
        }
        to_feed = {**feed_dict, **static}

        res = session.run(fetches,
                          feed_dict=to_feed)
        return res

    # Helper function that restores the model from a checkpoint
    def restore_checkpoint(self, saver, model_name):
        # saver.restore(tf.get_default_session(), "{}/model_{}.ckpt".format(CHECKPOINTS_DIR, model_name))
        saver.restore(tf.get_default_session(), model_name)


# Code so enable commandline execution of the configuration selection
def _main():
    parser = argparse.ArgumentParser(description='Test LRP configurations')
    parser.add_argument('-i', '--input_features', type=str, nargs=1,
                        help='the location of the input features')
    parser.add_argument('-m', '--model', type=str, nargs=1, help='trained model to use')
    parser.add_argument('-d', '--destination', type=str, nargs=1, help='Destination directory')
    args = parser.parse_args()

    config_select = ConfigSelection(**vars(args))

    # Call config selection with gathered arguments
    config_select()


if __name__ == '__main__':
    _main()
