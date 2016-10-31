# -*- coding: utf-8 -*-
'''This module implements several neural network layers.'''

import tensorflow as tf
import numpy as np

class EmbeddingLayer(object):
    '''A class for an embedding layer.'''
    def __init__(self, vocabulary_size, embedding_size, keep_prob,
                 embedding_index=0, num_features=1, embedding_table=None,
                 update_embeddings=True):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.keep_prob = keep_prob
        self.embedding_index = embedding_index
        self.num_features = num_features
        self.embedding_table = embedding_table
        self.update_embeddings = update_embeddings
        self.M = None

    def _create_variables(self):
        '''Creates the parameter variables.'''
        name = "M%d" % self.embedding_index
        if self.embedding_table is None:
            self.M = tf.get_variable(name=name,
                                     shape=[self.vocabulary_size,
                                            self.embedding_size],
                                     initializer=\
                                         tf.contrib.layers.xavier_initializer( \
                                             uniform=True, dtype=tf.float32))
        else:
            self.M = tf.get_variable(name=name,
                                     shape=[self.embedding_table.shape[0],
                                            self.embedding_table.shape[1]],
                                     initializer=\
                                     tf.constant_initializer( \
                                         self.embedding_table),
                                     trainable=self.update_embeddings)
            assert len(self.embedding_table[0] == self.embedding_size)

    def forward(self, input_sequence):
        '''Performs a forward step.
        input_sequence is batch_size x sequence_length x num_features.'''
        self._create_variables()

        batch_size = tf.shape(input_sequence)[0]
        sequence_length = tf.shape(input_sequence)[1]
        num_features = self.num_features # tf.shape(input_sequence)[2]

        # Dropout on embeddings.
        M = tf.nn.dropout(self.M, self.keep_prob)
        # batch_size x L x window_size x emb_size
        emb_orig = tf.nn.embedding_lookup(M, input_sequence, name="emb_orig")
        emb = tf.reshape(emb_orig,
                         [batch_size,
                          sequence_length,
                          num_features * self.embedding_size],
                         name="emb") # batch_size x L x window_size*emb_size
        return emb

class FeedforwardLayer(object):
    '''A class for a feedforward layer.'''
    def __init__(self, sequence_length, input_size, hidden_size, batch_size,
                 activation=tf.nn.tanh):
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.activation = activation
        self.W_xh = None
        self.b_h = None

    def _create_variables(self):
        '''Creates the parameter variables.'''
        self.W_xh = tf.get_variable(name="W_xh",
                                    shape=[self.input_size,
                                           self.hidden_size],
                                    initializer= \
                                        tf.contrib.layers.xavier_initializer(
                                            uniform=True, dtype=tf.float32))
        self.b_h = tf.get_variable(shape=[self.hidden_size],
                                   initializer=\
                                       tf.random_uniform_initializer(
                                           dtype=tf.float32), name="b_h")

    def forward(self, input_sequence):
        '''Performs a forward step.
        input_sequence is batch_size x sequence_length x input_size.'''
        self._create_variables()

        # fully-connected layer on top of embeddings to reduce size
        x = tf.reshape(input_sequence, [self.batch_size*self.sequence_length,
                                        self.input_size])
        H = tf.reshape(self.activation(tf.matmul(x, self.W_xh) + self.b_h), \
                       [self.batch_size,
                        self.sequence_length,
                        self.hidden_size])
        return H

class RNNLayer(object):
    '''A class for a vanilla recurrent layer.'''
    def __init__(self, sequence_lengths, hidden_size, batch_size, keep_prob):
        self.sequence_lengths = sequence_lengths
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.cell = None

    def _create_variables(self):
        '''Creates the parameter variables.'''
        self.cell = tf.nn.rnn_cell.RNNCell(num_units=self.hidden_size,
                                           state_is_tuple=True)

    def forward(self, input_sequence):
        '''Performs a forward step.
        input_sequence is batch_size x max_sequence_length x input_size.'''
        self._create_variables()

        # Dropout on RNN.
        cell = tf.nn.rnn_cell.DropoutWrapper( \
            self.cell,
            input_keep_prob=1.0, \
            output_keep_prob=self.keep_prob)

        outputs, _, = tf.nn.dynamic_rnn(
            cell=cell, inputs=input_sequence,
            sequence_length=self.sequence_lengths,
            dtype=tf.float32, time_major=False)

        H = outputs
        return H

class LSTMLayer(RNNLayer):
    '''A class for a LSTM recurrent layer. Derives from a vanilla RNN.'''
    def __init__(self, sequence_lengths, hidden_size, batch_size, keep_prob):
        RNNLayer.__init__(self, sequence_lengths, hidden_size, batch_size,
                          keep_prob)

    def _create_variables(self):
        '''Creates the parameter variables.'''
        self.cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size,
                                            state_is_tuple=True)

class BILSTMLayer(LSTMLayer):
    '''A class for a bidirectional LSTM recurrent layer. Derives from a LSTM.'''
    def __init__(self, sequence_lengths, hidden_size, batch_size, keep_prob):
        LSTMLayer.__init__(self, sequence_lengths, hidden_size, batch_size,
                           keep_prob)
        self.bw_cell = None

    def _create_variables(self):
        '''Creates the parameter variables.'''
        LSTMLayer._create_variables(self)
        with tf.name_scope("bw_cell"):
            self.bw_cell = tf.nn.rnn_cell.LSTMCell( \
                num_units=self.hidden_size, state_is_tuple=True)

    def forward(self, input_sequence):
        '''Performs a forward step.
        input_sequence is batch_size x max_sequence_length x input_size.'''
        self._create_variables()

        # Dropout on LSTM.
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(
            self.cell,
            input_keep_prob=1.0,
            output_keep_prob=self.keep_prob)
        with tf.name_scope("bw_cell"):
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                self.bw_cell,
                input_keep_prob=1.0,
                output_keep_prob=self.keep_prob)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, input_sequence,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32, time_major=False)
            outputs = tf.concat(2, outputs) # concat of fw and bw lstm output.

        H = outputs
        return H

class ScoreLayer(object):
    '''A class for a score (softmax) layer.'''
    def __init__(self, sequence_length, input_size, num_labels, label_weights,
                 batch_size, batch_mask):
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.num_labels = num_labels
        self.label_weights = label_weights,
        self.batch_size = batch_size
        self.batch_mask = batch_mask
        self.w_p = None
        self.W_sp = None

    def _create_variables(self):
        '''Creates the parameter variables.'''
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
        '''Predict a label for an input, compute the loss and return label and
        loss.'''
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
        '''Performs a forward step.'''
        # TODO: this is not using the label_weights!
        self._create_variables()
        if self.label_weights is not None:
            label_weights = tf.constant(self.label_weights,
                                        name="label_weights")
        else:
            label_weigths = None

        # elems are unpacked along dim 0 -> L
        scores_pred = tf.map_fn(self._score_predict_loss,
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
    '''A class for a sketch layer, which performs a sequence of sketch
    operations, as defined in the neural easy-first model.'''
    def __init__(self, num_sketches, sequence_length, input_size, context_size,
                 hidden_size, batch_size, batch_mask, keep_prob,
                 discount_factor=0.0, temperature=1.0, activation=tf.nn.tanh):
        self.num_sketches = num_sketches
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.batch_mask = batch_mask
        self.keep_prob = keep_prob
        self.discount_factor = discount_factor
        self.temperature = temperature
        self.activation = activation
        self.W_cs = None
        self.w_s = None
        self.W_cz = None
        self.w_z = None
        self.v = None

    def _create_variables(self):
        '''Creates the parameter variables.'''
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
        '''Extract self.context_size context columns around each column and
        concatenate.'''
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
        '''Compute the softmax including the mask
        the mask is multiplied with exp(x), before the normalization.
        :param tensor: 2D
        :param mask: 2D, same shape as tensor
        :param tau: temperature, the cooler the more spiked is distribution
        :return: the softmax distribution.
        '''
        row_max = tf.expand_dims(tf.reduce_max(tensor, 1), 1)
        t_shifted = tensor - row_max
        nom = tf.exp(t_shifted/tau)*tf.cast(mask, tf.float32)
        row_sum = tf.expand_dims(tf.reduce_sum(nom, 1), 1)
        softmax = nom / row_sum
        return softmax

    def _compute_attention(self, state_matrix, b,
                           discount_factor=0.0, temperature=1.0):
        '''Compute attention weight for all words in sequence in batch.'''
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
        attention = scores - discount_factor*b
        attention = self._softmax_with_mask(attention, self.batch_mask, tau=tau)
        #attention = self._softmax_with_mask(scores, self.batch_mask, tau=1.0)
        return attention, scores

    def forward(self, input_sequence):
        '''Performs a forward step.'''
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
        num_sketches = self.num_sketches
        sketch_counter = tf.constant(1, dtype=tf.int32, name="sketch_counter")
        if num_sketches > 0:
            for i in xrange(num_sketches):
                # Perform a sketch step.

                # Concatenation and convolution.
                HS = tf.concat(2, [H, S])
                C = self._convolute(HS)

                # Compute attention for where to focus next.
                a_n, _ = self._compute_attention(
                    C, b, discount_factor=self.discount_factor,
                    temperature=self.temperature)

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
                #sketch_attention_cumulative = \
                #tf.concat(2, [sketch_update, tf.expand_dims(a_n, 2),
                #                             tf.expand_dims(b_n, 2)])
                sketch_attention_cumulative = \
                    tf.concat(2, [tf.expand_dims(a_n, 2),
                                  tf.expand_dims(b_n, 2)])
                sketches.append(sketch_attention_cumulative)

        sketches_tf = tf.pack(sketches)

        return S, sketches_tf

