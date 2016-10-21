import tensorflow as tf
import tensorflow.python.util.nest as nest
import numpy as np
import logging
import cPickle as pkl
import pdb

class ScoreLayer(object):
    def __init__(self, sequence_length, input_size, num_labels, label_weights,
                 batch_size, batch_mask):
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.num_labels = num_labels
        self.label_weights = label_weights,
        self.batch_size = batch_size
        self.batch_mask = batch_mask

    def _create_variables(self):
        # Create matrix and bias variables for computing
        # s = tanh(W_cs*c_bar + w_s).
        self.w_p = tf.get_variable(name="w_p", shape=[self.num_labels],
                                   initializer= \
                                       tf.random_uniform_initializer(
                                           dtype=tf.float32))
        self.W_sp = tf.get_variable(name="W_sp", shape=[self.input_size,
                                                        self.num_labels],
                                    initializer= \
                                        tf.contrib.layers.xavier_initializer(
                                            uniform=True, dtype=tf.float32))

    def _score_predict_loss(self, score_input):
        """
        Predict a label for an input, compute the loss and return label and loss
        """
        [x, y] = score_input

        word_label_score = tf.matmul(tf.reshape(x,
                                                [self.batch_size,
                                                 self.input_size]),
                                     self.W_sp) + self.w_p

        word_label_probs = tf.nn.softmax(word_label_score)
        word_preds = tf.argmax(word_label_probs, 1)
        y_full = tf.one_hot(tf.squeeze(y),
                            depth=self.num_labels,
                            on_value=1.0,
                            off_value=0.0)
        cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits(word_label_score,
                                                    y_full)
        if self.label_weights is not None:
            label_weights = tf.reduce_sum(tf.mul(y_full, self.label_weights), 1)
            cross_entropy = tf.mul(cross_entropy, label_weights)
        return [word_preds, cross_entropy]

    def forward(self, input_sequence, y):
        self._create_variables()

        if self.label_weights is not None:
            label_weights = tf.constant(self.label_weights,
                                        name="label_weights")
        else:
            label_weigths = None

        f = lambda score_input : self._score_predict_loss(score_input)
        # elems are unpacked along dim 0 -> L
        scores_pred = tf.map_fn(f,
                                [tf.transpose(input_sequence, [1, 0, 2]),
                                 tf.transpose(y, [1, 0])],
                                dtype=[tf.int64, tf.float32])
        pred_labels = scores_pred[0]
        losses = scores_pred[1]

        # masked, batch_size x L
        pred_labels = self.batch_mask*tf.transpose(pred_labels, [1, 0])
        losses = tf.reduce_mean(tf.cast(self.batch_mask, tf.float32)*
                                tf.transpose(losses, [1, 0]),
                                1)  # masked, batch_size x 1
        return losses, pred_labels


class SketchLayer(object):
    def __init__(self, sequence_length, input_size, context_size, hidden_size,
                 batch_size, batch_mask, keep_prob, activation=tf.nn.tanh):
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.batch_mask = batch_mask
        self.keep_prob = keep_prob
        self.activation = activation

    def _create_variables(self):
        # Create matrix and bias variables for computing
        # s = tanh(W_cs*c_bar + w_s).
        window_size = 2*self.context_size+1
        state_size = (self.input_size + self.hidden_size) * window_size
        self.W_cs = tf.get_variable(name="W_cs",
                                    shape=[state_size, self.hidden_size],
                                    initializer= \
                                        tf.contrib.layers.xavier_initializer(
                                            uniform=True, dtype=tf.float32))
        self.w_s = tf.get_variable(name="w_s",
                                   shape=[self.hidden_size],
                                   initializer=\
                                      tf.random_uniform_initializer(
                                          dtype=tf.float32))

        # Create vector, matrix and bias variables for computing
        # z_i = v'*tanh(W_cz*c_i + w_z).
        self.W_cz = tf.get_variable(name="W_cz",
                                    shape=[state_size, self.hidden_size],
                                    initializer=\
                                    tf.contrib.layers.xavier_initializer(
                                        uniform=True, dtype=tf.float32))
        self.w_z = tf.get_variable(name="w_z",
                                   shape=[self.hidden_size],
                                   initializer=tf.random_uniform_initializer(
                                       dtype=tf.float32))
        self.v = tf.get_variable(name="v", shape=[self.hidden_size, 1],
                                 initializer=tf.random_uniform_initializer(
                                     dtype=tf.float32))

    def _convolute(self, input_matrix):
        """
        Extract r context columns around each column and concatenate
        :param padded_matrix: batch_size x L+(2*r) x 2*state_size
        :param r: context size
        :return:
        """
        state_size = tf.shape(input_matrix)[2]
        window_size = 2*self.context_size + 1
        padding_columns = tf.constant([[0, 0],
                                       [self.context_size, self.context_size],
                                       [0, 0]], name="padding_columns")

        # Add column on right and left.
        padded_matrix = tf.pad(input_matrix,
                               padding_columns,
                               "CONSTANT",
                               name="padded_matrix")

        # Gather indices of padded.
        # time-major  -> sequence_length x state_size x batch_size.
        time_major_matrix = tf.transpose(padded_matrix, [1, 2, 0])
        contexts = []
        for j in xrange(self.context_size,
                        self.sequence_length + self.context_size):
            # Extract 2r+1 rows around i for each batch.
            # 2*r+1 x state_size x batch_size.
            context = time_major_matrix[j-self.context_size:
                                        j+self.context_size+1, :, :]
            # Concatenate.
            # (2*r+1)*(state_size) x batch_size.
            context = tf.reshape(context, [window_size*state_size,
                                           self.batch_size])
            contexts.append(context)
        contexts = tf.pack(contexts)  # L x (2*r+1)*(state_size) x batch_size.
        # switch back: batch_size x L x (2*r+1)*2(state_size) (batch-major).
        output_matrix = tf.transpose(contexts, [2, 0, 1])
        return output_matrix

    def _softmax_with_mask(self, tensor, mask, tau=1.0):
        """
        compute the softmax including the mask
        the mask is multiplied with exp(x), before the normalization
        :param tensor: 2D
        :param mask: 2D, same shape as tensor
        :param tau: temperature, the cooler the more spiked is distribution
        :return:
        """
        row_max = tf.expand_dims(tf.reduce_max(tensor, 1), 1)
        t_shifted = tensor - row_max
        nom = tf.exp(t_shifted/tau)*tf.cast(mask, tf.float32)
        row_sum = tf.expand_dims(tf.reduce_sum(nom, 1), 1)
        softmax = nom / row_sum
        return softmax

    def _compute_attention(self, state_matrix, b,
                           discount_factor=0.0, temperature=1.0):
        """
        Compute attention weight for all words in sequence in batch
        :return:
        """
        state_size = tf.shape(state_matrix)[2]
        z = []
        for j in np.arange(self.sequence_length):
            state = tf.slice(state_matrix,
                             [0, j, 0],
                             [self.batch_size, 1, state_size])
            state = tf.reshape(state, [self.batch_size, state_size])
            activ = self.activation(tf.matmul(state, self.W_cz) + self.w_z)
            z_j = tf.matmul(activ, self.v)
            z.append(z_j)
        z = tf.pack(z)  # sequence_length x batch_size x 1.
        scores = tf.transpose(z, [1, 0, 2])  # batch-major.
        scores = tf.reshape(scores, [self.batch_size, self.sequence_length])
        # subtract cumulative attention
        d = discount_factor # 5.0  # discount factor
        tau = temperature # 0.2 # temperature.
        #attention = scores - discount_factor*b
        #attention = self._softmax_with_mask(attention, mask, tau=tau)
        attention = self._softmax_with_mask(scores, self.batch_mask, tau=1.0)
        return attention, scores

    def forward(self, input_sequence):
        self._create_variables()

        # Dropout within sketch.
        # see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
        # python/ops/nn_ops.py#L1078 (inverted dropout).
        W_cs_mask = \
            tf.to_float(tf.less_equal(tf.random_uniform(tf.shape(self.W_cs)),
                                      self.keep_prob)) * \
            tf.inv(self.keep_prob)

        # Input sequence (should be batch_size x sequence_length * input_size).
        H = input_sequence
        # Sketch matrix, initialized to zero.
        S = tf.zeros(shape=[self.batch_size,
                            self.sequence_length,
                            self.hidden_size],
                     dtype=tf.float32)

        # Cumulative attention; initialized to uniform.
        b = tf.ones(shape=[self.batch_size, self.sequence_length],
                    dtype=tf.float32) / self.sequence_length
        #b = tf.zeros(shape=[self.batch_size, self.sequence_length],
        #             dtype=tf.float32)
        b_n = b

        sketches = [] # For debug purposes.
        num_sketches = self.sequence_length
        sketch_counter = tf.constant(1, dtype=tf.int32, name="sketch_counter")
        if num_sketches > 0:
            for i in xrange(num_sketches):
                # Perform a sketch step.

                # Concatenation and convolution.
                HS = tf.concat(2, [H, S])
                C = self._convolute(HS)

                # Compute attention for where to focus next.
                a_n, _ = self._compute_attention(C, b)

                # Cumulative attention scores.
                #b_n = b + a_n
                b_n = (tf.cast(sketch_counter, tf.float32)-1)*b + a_n #rz
                b_n /= tf.cast(sketch_counter, tf.float32)

                state_size = tf.shape(C)[2]
                # batch_size x 1 x state_size.
                c_bar = tf.batch_matmul(tf.expand_dims(a_n, [1]), C)
                c_bar = tf.reshape(c_bar, [self.batch_size, state_size])

                # Same dropout for all steps:
                # (http://arxiv.org/pdf/1512.05287v3.pdf), mask is ones if
                # no dropout.
                a = tf.matmul(c_bar, tf.mul(self.W_cs, W_cs_mask))
                # batch_size x hidden_size
                s_n = self.activation(a + self.w_s)

                # batch_size x L x state_size.
                sketch_update = tf.batch_matmul(tf.expand_dims(a_n, [2]),
                                                tf.expand_dims(s_n, [1]))
                S += sketch_update
                sketch_counter += 1

                # For debug purposes?
                # append attention to sketch
                #sketch_attention_cumulative = tf.concat(2, [sketch_update, tf.expand_dims(a_n, 2), tf.expand_dims(b_n, 2)])
                sketch_attention_cumulative = tf.concat(2, [tf.expand_dims(a_n, 2), tf.expand_dims(b_n, 2)])
                sketches.append(sketch_attention_cumulative)

        sketches_tf = tf.pack(sketches)

        return S, sketches_tf


    
