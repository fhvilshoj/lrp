import argparse
import tensorflow as tf
import numpy as np

from config_selection import logger
import sirs_classifier
from feature_parser import FeatureParser

INPUT_SIZE = 4512
CONTEXT_SIZE = 123
NUM_CLASSES = 10

# The number of epochs and the size of each mini batch
NUM_EPOCHS = None

# Don't use autoencoder, text or doc_vec since we are dealing with images
USE_AUTOENCODER = False
USE_TEXT = False
USE_DOC2VEC = False


def _create_model(features):
    return sirs_classifier.create_model(None, None, features, False, INPUT_SIZE, CONTEXT_SIZE, NUM_CLASSES, USE_TEXT)

def _restore_checkpoint(sess, model_name):
    saver = tf.train.Saver()
    saver.restore(sess, model_name)

def _run_model(sess, model, fetches):
    return sess.run(fetches,
                    feed_dict={
                        model['reset_auc']: False,
                        model['dropout_keep_prob']: 1.0
                    })


def _evaluate_model(sess, tf_model, test_size, tf_scores, **kwargs):
    total_evals = 0
    all_scores = []

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    try:
        while (not coord.should_stop()) and total_evals < test_size:
            # Run training step
            calculated_scores = _run_model(sess, tf_model, tf_scores)
            all_scores.append(calculated_scores)

            if total_evals % (kwargs['batch_size'] * 10) == 0:
                logger.debug(calculated_scores)

            total_evals += kwargs['batch_size']

    except tf.errors.OutOfRangeError:
        print('Training completed')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    coord.join(threads)

    scores = np.array(all_scores)

    return np.mean(scores, axis=0)


def _do_test(test_file, **kwargs):
    with tf.Graph().as_default():

        parser = FeatureParser(test_file, INPUT_SIZE, CONTEXT_SIZE, **kwargs)
        features = parser.next_batch()
        num_records = parser.get_record_count()

        # Keep scope empty
        sirs_model = tf.make_template('', _create_model)
        model = sirs_model(features)

        valid_scores = []
        scores = []
        for s in kwargs['scores']:
            if s in model:
                valid_scores.append(s)

                m_score = model[s]
                if s == 'accuracy':
                    m_score = tf.reduce_mean(tf.cast(m_score, tf.float32))
                scores.append(m_score)
            else:
                logger.error("Score '{}' is not supported.".format(s))

        with tf.Session() as sess:
            # Initialize variables
            sess.run([tf.local_variables_initializer()])

            # Restore model
            _restore_checkpoint(sess, kwargs['model'])

            # Find scores
            evaluations = _evaluate_model(sess, model, num_records, scores, **kwargs)

        # Print scores
        logger.info("Scores: ")
        for sc in zip(valid_scores, evaluations):
            print(sc)
            logger.info("{:<10}: {:10f}".format(*sc))

def _main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Get model scores')
    parser.add_argument('-t', '--test_file', type=str, nargs=1,
                        default=['E:/frederikogbenjamin/sepsis_model/data/TODO.txt_compressed'],
                        help='The location of the input features')
    parser.add_argument('-m', '--model', type=str,
                        default='E:/frederikogbenjamin/sepsis_model/checkpoints/remaped_for_1.4.ckpt',
                        help='Trained model to use')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=10,
                        help='Batch size when testing')
    parser.add_argument('-s', '--scores', type=str, nargs='*',
                        default=['accuracy', 'f1', 'precision', 'recall'])
    args = parser.parse_args()

    # Call config selection with gathered arguments
    _do_test(**vars(args))


if __name__ == '__main__':
    _main()
