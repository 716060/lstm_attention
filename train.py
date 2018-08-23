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


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()

    FLAGS.is_training = True
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

    control_dependencies = []
    if FLAGS.check_nans:
        checks = tf.add_check_numerics_ops()
        control_dependencies = [checks]

    labels = tf.placeholder(tf.int64, [], name='labels')
    with tf.name_scope('train_loss'):
        loss = models.tuple_loss(batch_size=FLAGS.batch_size,
                                 tuple_size=1 + FLAGS.num_utt_enrollment,
                                 speakers=outputs,
                                 labels=labels)
    tf.summary.scalar('train_loss', loss)
    with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
        learning_rate_input = tf.placeholder(tf.float32, name='learning_rate_input')
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate_input).minimize(loss)

    global_step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver(tf.global_variables())

    # Merge all the summaries and write them out to tmp/retrain_logs (by default)
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    # validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    tf.global_variables_initializer().run()

    start_step = 1

    if FLAGS.start_checkpoint:
        models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
        start_step = global_step.eval(session=sess)

    tf.logging.info('Training from step: %d ', start_step)

    # Save graph.pbtxt.
    tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                         FLAGS.model_architecture + '.pbtxt')

    features_read_buffer = h5py.File(os.path.join(FLAGS.data_dir, 'features.hdf5'), 'r')
    p_tuples_read_buffer = open(os.path.join(FLAGS.data_dir, 'training_positive_tuples'), 'r')
    n_tuples_read_buffer = open(os.path.join(FLAGS.data_dir, 'training_negative_tuples'), 'r')
    all_p_tuples = p_tuples_read_buffer.readlines()
    all_n_tuples = n_tuples_read_buffer.readlines()

    with tf.name_scope('train_loss'):
        loss = models.tuple_loss(batch_size=FLAGS.batch_size,
                                 tuple_size=1 + FLAGS.num_utt_enrollment,
                                 speakers=outputs,
                                 labels=labels)

    # Training loop.
    training_steps_max = (int(len(all_p_tuples) / FLAGS.batch_size)) * 2
    training_steps_max = int(training_steps_max / 2)

    tf.logging.info('Total steps %d: ', training_steps_max)

    for training_step in range(training_steps_max):
        # Pull the audio samples we'll use for training.
        if training_step % 2 == 0:
            tuples = all_p_tuples[
                     int(training_step / 2) * FLAGS.batch_size:(int(training_step / 2) + 1) * FLAGS.batch_size]
            voiceprint, _ = audio_processor.get_data(tuples, features_read_buffer)
            label = 1
        else:
            tuples = all_n_tuples[int((training_step - 1) / 2) * FLAGS.batch_size:(int(
                (training_step - 1) / 2) + 1) * FLAGS.batch_size]
            voiceprint, _ = audio_processor.get_data(tuples, features_read_buffer)
            label = 0

        train_summary, train_loss, _ = sess.run(
            [
                merged_summaries,
                loss,
                train_step
            ],
            feed_dict={
                input_audio_data: voiceprint,
                labels: label,
                learning_rate_input: FLAGS.learning_rate,
                dropout_prob_input: FLAGS.dropout_prob
            })
        train_writer.add_summary(train_summary, training_step)
        tf.logging.info('Step #%d: loss %f' % (training_step, train_loss))

        # Save the model checkpoint periodically.
        if (training_step % FLAGS.save_step_interval == 0 or
                training_step == training_steps_max):
            checkpoint_path = os.path.join(FLAGS.train_dir,
                                           FLAGS.model_architecture + '.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
            saver.save(sess, checkpoint_path, global_step=training_step)

    features_read_buffer.close()
    p_tuples_read_buffer.close()
    n_tuples_read_buffer.close()


if __name__ == '__main__':
    pwd = os.getcwd()
    parser = argparse.ArgumentParser()
    command.prepare_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
