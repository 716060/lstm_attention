def prepare_arguments(parser):
    add_data_arguments(parser)
    add_audio_arguments(parser)
    add_lstm_arguments(parser)
    add_training_arguments(parser)


def add_data_arguments(parser):
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/yes',
        help='Where to download the speech training data to.')
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')


def add_audio_arguments(parser):
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs')
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs')
    parser.add_argument(
        '--window_size_ms',
        type=int,
        default=25,
        help='How long each spectrogram timeslice is.')
    parser.add_argument(
        '--window_stride_ms',
        type=int,
        default=10,
        help='How far to move in time between spectogram timeslices.')
    parser.add_argument(
        '--num_coefficient',
        type=int,
        default=40,
        help='Number of coefficients of the wavs')
    parser.add_argument(
        '--num_utt_enrollment',
        type=int,
        default=5,
        help='Number of enrollment utts for each speaker')


def add_lstm_arguments(parser):
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='lstm')
    parser.add_argument(
        '--num_units',
        type=int,
        default=128,
        help='Numbers of units for each layer of lstm')
    parser.add_argument(
        '--dimension_projection',
        type=int,
        default=64,
        help='Dimension of projection layer of lstm')
    parser.add_argument(
        '--num_layers',
        type=int,
        default=3,
        help='Number of layers of multi-lstm')
    parser.add_argument(
        '--dimension_linear_layer',
        type=int,
        default=64,
        help='Dimension of linear layer on top of lstm')
    parser.add_argument(
        '--attention_layer',
        type=str,
        default=None,
        help='Attention layer of lstm')
    parser.add_argument(
        '--scoring_function',
        type=str,
        default='bias_only',
        help='Scoring function for attention layer of lstm')
    parser.add_argument(
        '--weights_pooling',
        type=str,
        default=None,
        help='Weights pooling for attention layer of lstm')


def add_training_arguments(parser):
    parser.add_argument(
        '--learning_rate',
        type=float,
        default='0.0001',
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--dropout_prob',
        type=float,
        default=0.5,
        help='Dropout probability')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=80,
        help='How many items to train with at once')
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=False,
        help='Whether to check for invalid numbers during processing')
    parser.add_argument(
        '--skip_feature_generation',
        type=bool,
        default=True,
        help='Whether to skip the phase of generating mfcc features')
    parser.add_argument(
        '--num_repeats',
        type=int,
        default=140,
        help='Number of repeat when we prepare the tuples')
    parser.add_argument(
        '--save_step_interval',
        type=int,
        default=100,
        help='Save model checkpoint every save_steps.')
    parser.add_argument(
        '--train_dir',
        type=str,
        default='tmp/train',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')
