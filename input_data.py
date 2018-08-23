import os
import random

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops

import util


def prepare_audio_settings(sample_rate, clip_duration_ms, window_size_ms, window_stride_ms, num_coefficient):
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = desired_samples - window_size_samples

    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    return {
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'num_coefficient': num_coefficient,
        'sample_rate': sample_rate
    }


class AudioProcessor(object):
    def __init__(self, data_dir, audio_config, num_repeats, num_utt_enrollment, is_training=True):
        self.validation_percentage = 10
        self.testing_percentage = 10

        self.data_index = {'validation': [], 'testing': [], 'training': []}

        for wav in os.listdir(data_dir):
            if wav.endswith(".wav"):
                set_index = util.which_set(wav, self.validation_percentage, self.testing_percentage)
                self.data_index[set_index].append(os.path.join(data_dir, wav))

        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])

        self.data_dir = data_dir
        self.audio_settings = audio_config
        self.num_utt_enrollment = num_utt_enrollment

        if is_training:
            generate_tuples(data_dir, num_repeats)
            generate_features(data_dir, audio_config)

    def get_data(self, tuples, read_buffer):
        batch_size = len(tuples)
        tuple_size = self.num_utt_enrollment + 1
        desired_frames = self.audio_settings['spectrogram_length']
        feature_size = self.audio_settings['num_coefficient']
        data = np.zeros((batch_size, tuple_size, desired_frames, feature_size))
        label = np.zeros(batch_size)

        for i, t in enumerate(tuples):
            content = t.split()
            label[i] = int(content[-1])

            for j, utterance in enumerate(content[:-1]):
                if len(data[i]) <= j:
                    continue

                utterance_name = os.path.basename(utterance)
                mfcc_mat = read_buffer[utterance_name]
                mfcc_shape = mfcc_mat.shape
                num_frames_mfcc = mfcc_shape[0]
                manque_frames = num_frames_mfcc - desired_frames

                if manque_frames < 0:
                    if data[i, j].shape == mfcc_mat[0].shape:
                        data[i, j] = mfcc_mat[0]
                    else:
                        extra_frames = data[i, j].shape[0]-mfcc_mat[0].shape[0]
                        data[i, j] = np.lib.pad(mfcc_mat[0], [(extra_frames, 0), (0, 0)], 'constant', constant_values=0)
                else:
                    start_frame = random.randint(0, manque_frames)
                    data[i, j] = mfcc_mat[start_frame: start_frame + desired_frames]

        return data, label


def generate_features(data_dir, audio_settings):
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(data_dir + '/' + wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
        spectrogram = contrib_audio.audio_spectrogram(wav_decoder.audio,
                                                      window_size=audio_settings['window_size_samples'],
                                                      stride=audio_settings['window_stride_samples'],
                                                      magnitude_squared=True)
        mfcc = contrib_audio.mfcc(spectrogram, wav_decoder.sample_rate,
                                  dct_coefficient_count=audio_settings['num_coefficient'])

        hdf5_filename = os.path.join(data_dir, 'features.hdf5')
        write_buffer = h5py.File(hdf5_filename, 'w')

        for wav in os.listdir(data_dir):
            if not wav.endswith('.wav'):
                continue

            utterance_name = os.path.basename(wav)
            mfcc_mat = sess.run(mfcc,
                                feed_dict={
                                    wav_filename_placeholder: wav,
                                })
            write_buffer[utterance_name] = mfcc_mat
        write_buffer.close()


def generate_tuples(data_dir, num_repeats):
    # Generate a speaker to utterances dictionary
    speaker_to_utterances = {}
    for wav in os.listdir(data_dir):
        if not wav.endswith('.wav'):
            continue

        base_name = os.path.basename(wav)
        utterance_name = base_name[:base_name.index("_")]

        if utterance_name in speaker_to_utterances:
            utterances = speaker_to_utterances[utterance_name]
            utterances.add(wav)
        else:
            speaker_to_utterances[utterance_name] = {wav}

    for set_index in ['validation', 'testing', 'training']:
        generate_positive_tuples(data_dir, set_index, num_repeats, speaker_to_utterances)
        generate_negative_tuples(data_dir, set_index, num_repeats, speaker_to_utterances)

    # All Tuples
    all_tuples = {'validation': [], 'testing': [], 'training': []}
    for set_index in ['validation', 'testing', 'training']:
        all_tuples[set_index].append(os.path.join(data_dir, set_index + '_positive_tuples'))
        all_tuples[set_index].append(os.path.join(data_dir, set_index + '_negative_tuples'))

    for set_index in ['validation', 'testing', 'training']:
        with open(os.path.join(data_dir, 'all_' + set_index + '_tuples'), 'w') as outfile:
            for fname in all_tuples[set_index]:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)


def generate_positive_tuples(data_dir, mode, num_repeats, speaker_to_utterances):
    write_buffer = open(os.path.join(data_dir, mode + '_positive_tuples'), 'w')
    for i in range(num_repeats):
        for speaker in speaker_to_utterances.keys():
            utterances = speaker_to_utterances[speaker]
            samples = random.sample(utterances, len(utterances))
            for utterance in samples:
                write_buffer.write(utterance + ' ')
            write_buffer.write('1\n')
    write_buffer.close()


def generate_negative_tuples(data_dir, mode, num_repeats, speaker_to_utterances):
    write_buffer = open(os.path.join(data_dir, mode + '_negative_tuples'), 'w')
    for speaker in speaker_to_utterances.keys():
        all_speakers = speaker_to_utterances.keys()[:]
        all_speakers.remove(speaker)

        for i in range(num_repeats):
            enrollments = random.sample(all_speakers, 1)[0]
            evaluations = random.sample(speaker_to_utterances[speaker], 1)
            utterances = speaker_to_utterances[enrollments]
            samples = random.sample(utterances, len(utterances))
            write_buffer.write(evaluations[0] + ' ')
            for utt_enroll in samples:
                write_buffer.write(utt_enroll + ' ')
            write_buffer.write('0\n')
    write_buffer.close()
