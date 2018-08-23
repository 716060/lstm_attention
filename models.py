from __future__ import division

import tensorflow as tf


def create_model(input_audio_tuple, model_settings, model_architecture):
    if model_architecture == 'lstm':
        return create_lstm_model(input_audio_tuple, model_settings)
    else:
        raise Exception('model_architecture argument "' + model_architecture +
                        '" not recognized, should be one of "lstm"')


def create_lstm_model(input_audio_tuple, model_settings):
    input_data_shape = input_audio_tuple.shape
    batch_size = input_data_shape[0]
    tuple_size = input_data_shape[1]
    time_steps = input_data_shape[2]
    feature_size = input_data_shape[3]

    X = tf.reshape(input_audio_tuple, [batch_size * tuple_size, time_steps, feature_size])
    XT = tf.transpose(X, [1, 0, 2])
    XR = tf.reshape(XT, [-1, feature_size])
    X_split = tf.split(XR, time_steps, 0)

    num_units = model_settings['num_units']
    dimension_projection = model_settings['dimension_projection']
    num_layers = model_settings['num_layers']
    dropout_prob = model_settings['dropout_prob']

    cell = tf.nn.rnn_cell.MultiRNNCell(
        [make_lstm_cell(num_units, dimension_projection, dropout_prob) for _ in range(num_layers)],
        state_is_tuple=True)
    outputs, _states = tf.contrib.rnn.static_rnn(cell=cell, inputs=X_split, dtype=tf.float32)

    attention_layer = model_settings['attention_layer']
    scoring_function = model_settings['scoring_function']
    pooling = model_settings['weights_pooling']

    w = model_settings['weights']
    b = model_settings['bias']

    if not attention_layer:
        return tf.matmul(outputs[-1], w) + b
    else:
        w_att = tf.Variable(tf.random_normal([dimension_projection, 1], stddev=0.1), name='w_att')
        b_att = tf.Variable(tf.random_normal([outputs[0].shape[0].value, 1], stddev=0.1), name='b_att')
        W_att = tf.Variable(tf.random_normal([dimension_projection, 1], stddev=0.1), name='W_att')
        v_att = tf.Variable(tf.random_normal([1, outputs[0].shape[0].value], stddev=0.1), name='v_att')

        if scoring_function == 'bias_only':
            alpha = 1 / len(outputs)
            outputs = tf.convert_to_tensor(outputs)
            outputs = tf.map_fn(lambda t: tf.scalar_mul(alpha, t), outputs)
            return tf.reduce_sum(outputs, 0)
        elif scoring_function == 'linear':
            e_att = []
            for idx, val in enumerate(outputs):
                w_t = tf.Variable(tf.random_normal([dimension_projection, 1], stddev=0.1), name='w_t')
                e_t = tf.norm(tf.matmul(val, w_t)) + b
                e_att.append(e_t)

            exp_e_t = tf.reduce_sum(tf.exp(e_att))
            alphas = []
            for idx, val in enumerate(e_att):
                alphas.append(tf.norm(tf.exp(val) / exp_e_t))
            alphas = weights_pooling(alphas, pooling)

            for idx, val in enumerate(outputs):
                outputs[idx] = tf.scalar_mul(alphas[idx], val)
            return tf.reduce_sum(outputs, 0)
        elif scoring_function == 'shared_parameter_linear':
            e_att = []
            for idx, val in enumerate(outputs):
                e_t = tf.norm(tf.matmul(val, w_att)) + b
                e_att.append(e_t)

            exp_e_t = tf.reduce_sum(tf.exp(e_att))
            alphas = []
            for idx, val in enumerate(e_att):
                alphas.append(tf.norm(tf.exp(val) / exp_e_t))
            alphas = weights_pooling(alphas, pooling)

            for idx, val in enumerate(outputs):
                outputs[idx] = tf.scalar_mul(alphas[idx], val)
            return tf.reduce_sum(outputs, 0)
        elif scoring_function == 'non_linear':
            e_att = []
            for idx, val in enumerate(outputs):
                w_t = tf.Variable(tf.random_normal([dimension_projection, 1], stddev=0.1), name='w_t')
                e_t = tf.matmul(v_att, tf.tanh(tf.norm(tf.matmul(val, w_t)) + b_att))
                e_att.append(e_t)

            exp_e_t = tf.reduce_sum(tf.exp(e_att))
            alphas = []
            for idx, val in enumerate(e_att):
                alphas.append(tf.norm(tf.exp(val) / exp_e_t))
            alphas = weights_pooling(alphas, pooling)

            for idx, val in enumerate(outputs):
                outputs[idx] = tf.scalar_mul(alphas[idx], val)
            return tf.reduce_sum(outputs, 0)
        elif scoring_function == 'shared_parameter_non_linear':
            e_att = []
            for idx, val in enumerate(outputs):
                e_t = tf.matmul(v_att, tf.tanh(tf.matmul(val, W_att) + b_att))
                e_att.append(e_t)

            exp_e_t = tf.reduce_sum(tf.exp(e_att))
            alphas = []
            for idx, val in enumerate(e_att):
                alphas.append(tf.norm(tf.exp(val) / exp_e_t))
            alphas = weights_pooling(alphas, pooling)

            for idx, val in enumerate(outputs):
                outputs[idx] = tf.scalar_mul(alphas[idx], val)
            return tf.reduce_sum(outputs, 0)
        else:
            raise Exception('scoring function argument "' + scoring_function +
                            '" not recognized, should be one of "' +
                            'basic, bias_only, ' +
                            'linear, shared_parameter_linear, ' +
                            'non_linear, or shared_parameter_non_linear')


def make_lstm_cell(num_units, dimension_projection, dropout_prob):
    cell = tf.contrib.rnn.LSTMCell(num_units=num_units, num_proj=dimension_projection, state_is_tuple=True)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1 - dropout_prob)


def weights_pooling(weights, pooling):
    if pooling.startswith('sliding_window'):
        args = pooling.split()
        if len(args) > 1:
            size = int(args[1])
            step = int(args[2])
            return sliding_window_pooling(weights, size, step)
        else:
            return sliding_window_pooling(weights)
    elif pooling.startswith('top'):
        args = pooling.split()
        if len(args) > 1:
            k = int(args[1])
            return top_k_pooling(weights, k)
        else:
            return top_k_pooling(weights)
    else:
        return weights


def sliding_window_pooling(weights, size=10, step=5):
    dim = len(weights)
    weights = tf.convert_to_tensor(weights)
    weights = tf.reshape(weights, [dim, 1, 1])
    weights = tf.nn.pool(input=weights,
                         window_shape=[size],
                         pooling_type="MAX",
                         padding="SAME")
    return tf.reshape(weights, [dim])


def top_k_pooling(weights, k=5, batch_size=None):
    weights = tf.convert_to_tensor(weights)
    in_shape = tf.shape(weights)
    d = weights.get_shape().as_list()[-1]
    matrix_in = tf.reshape(weights, [-1, d])
    values, indices = tf.nn.top_k(matrix_in, k=k, sorted=False)
    out = []
    values = tf.unstack(values, axis=0, num=batch_size)
    indices = tf.unstack(indices, axis=0, num=batch_size)
    for i, idx in enumerate(indices):
        out.append(
            tf.sparse_tensor_to_dense(tf.SparseTensor(tf.reshape(tf.cast(idx, tf.int64), [-1, 1]), values[i], [d]),
                                      validate_indices=False))
    shaped_out = tf.reshape(tf.stack(out), in_shape)
    return shaped_out


def tuple_loss(batch_size, tuple_size, speakers, labels):
    feature_size = speakers.shape[1]
    w = tf.reshape(speakers, [batch_size, tuple_size, feature_size])

    def cos_sim(idx):
        wi_enroll = w[idx, 1:]
        wi_eval = w[idx, 0]
        normalize_wi_enroll = tf.nn.l2_normalize(wi_enroll, axis=1)
        c_k = tf.reduce_mean(normalize_wi_enroll, 0)
        normalize_ck = tf.nn.l2_normalize(c_k, axis=0)
        normalize_wi_eval = tf.nn.l2_normalize(wi_eval, axis=0)
        return tf.reduce_sum(tf.multiply(normalize_ck, normalize_wi_eval))

    def f1():
        loss = 0
        for idx in range(batch_size):
            loss += tf.sigmoid(cos_sim(idx))
        return -tf.log(loss / batch_size)

    def f2():
        loss = 0
        for idx in range(batch_size):
            loss += (1 - tf.sigmoid(cos_sim(idx)))
        return -tf.log(loss / batch_size)

    return tf.cond(tf.equal(labels, 1), f1, f2)


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.
    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)
