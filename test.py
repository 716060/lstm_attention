from __future__ import division

import argparse
import os
import sys

import h5py
import tensorflow as tf

import command
import models
import util

FLAGS = None


def score(batch_size, tuple_size, speakers):
    feature_size = speakers.shape[1]
    w = tf.reshape(speakers, [batch_size, tuple_size, feature_size])
    score_batch = tf.zeros([1])

    for idx in range(batch_size):
        wi_enroll = w[idx, 1:]
        wi_eval = w[idx, 0]
        normalize_wi_enroll = tf.nn.l2_normalize(wi_enroll, axis=1)
        c_k = tf.reduce_mean(normalize_wi_enroll, 0)
        normalize_ck = tf.nn.l2_normalize(c_k, axis=0)
        normalize_wi_eval = tf.nn.l2_normalize(wi_eval, axis=0)
        cos_similarity = tf.reduce_sum(tf.multiply(normalize_ck, normalize_wi_eval))
        score_batch = tf.concat([score_batch, [cos_similarity]], 0)
    return score_batch[1:]


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()

    FLAGS.is_training = False
    model_settings, audio_processor, input_audio_data = util.prepare_settings(FLAGS)

    dimension_linear_layer = FLAGS.dimension_linear_layer
    model_settings['weights'] = tf.Variable(
        tf.random_normal([FLAGS.dimension_projection, dimension_linear_layer], stddev=1),
        name='weights')
    model_settings['bias'] = tf.Variable(tf.random_normal([dimension_linear_layer], stddev=1), name='bias')
    dropout_prob_input = tf.placeholder(tf.float32, [], name='dropout_prob_input')
    model_settings['dropout_prob'] = dropout_prob_input
    model_settings['attention_layer'] = FLAGS.attention_layer
    model_settings['scoring_function'] = FLAGS.scoring_function
    model_settings['weights_pooling'] = FLAGS.weights_pooling
    model_architecture = FLAGS.model_architecture

    outputs = models.create_model(input_audio_data, model_settings, model_architecture)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))

    batch_size = FLAGS.batch_size
    test_dir = os.path.join('tmp', 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    features_read_buffer = h5py.File(os.path.join(FLAGS.data_dir, 'features.hdf5'), 'r')
    testing_tuples_read_buffer = open(os.path.join(FLAGS.data_dir, 'all_testing_tuples'), 'r')
    score_write_buffer = open(os.path.join(test_dir, 'score_eval'), 'w')

    batch_label = tf.placeholder(tf.int32, [batch_size])
    batch_score = score(batch_size=batch_size, tuple_size=FLAGS.num_utt_enrollment + 1, speakers=outputs)

    all_testing_tuples = testing_tuples_read_buffer.readlines()
    num_iteration = int(len(all_testing_tuples) / batch_size)

    for i in range(num_iteration):
        tuples_batch = all_testing_tuples[i * batch_size: (i + 1) * batch_size]
        voiceprint, label = audio_processor.get_data(tuples_batch, features_read_buffer)
        score_batch = sess.run(batch_score,
                               feed_dict={
                                   input_audio_data: voiceprint,
                                   dropout_prob_input: 0,
                                   batch_label: label
                               })

        for j in range(score_batch.shape[0]):
            score_write_buffer.write(str(score_batch[j]) + '\n')

    features_read_buffer.close()
    testing_tuples_read_buffer.close()
    score_write_buffer.close()


if __name__ == '__main__':
    pwd = os.getcwd()
    parser = argparse.ArgumentParser()
    command.prepare_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
