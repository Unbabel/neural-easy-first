# coding=utf-8

import tensorflow as tf
import numpy as np
import time
#from sklearn.utils import shuffle
import sys
from utils import *
import math

"""
Tensorflow implementation of the neural easy-first model
- Single-State Model


Tensorflow issues:
- no rmsprop support for sparse gradient updates in version 0.9
- no nested while loops supported in version 0.9
"""

# Flags
tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_string("optimizer", "sgd", "Optimizer [sgd, adam, adagrad, adadelta, "
                                                    "momentum]")
tf.app.flags.DEFINE_integer("batch_size", 10,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("vocab_size", 20000, "Vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "../data/WMT2016/WMT2016", "Data directory")
tf.app.flags.DEFINE_string("model_dir", "models/", "Model directory")
tf.app.flags.DEFINE_integer("max_train_data_size", 1000,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_float("max_gradient_norm", -1, "maximum gradient norm for clipping (-1: no clipping)")
tf.app.flags.DEFINE_integer("L", 50, "maximum length of sequences")
tf.app.flags.DEFINE_integer("buckets", 5, "number of buckets")
#tf.app.flags.DEFINE_string("src_embeddings", "../data/WMT2016/embeddings/polyglot-en.pkl", "path to source language embeddings")
#tf.app.flags.DEFINE_string("tgt_embeddings", "../data/WMT2016/embeddings/polyglot-de.pkl", "path to target language embeddings")
tf.app.flags.DEFINE_string("src_embeddings", "", "path to source language embeddings")
tf.app.flags.DEFINE_string("tgt_embeddings", "", "path to target language embeddings")
tf.app.flags.DEFINE_integer("K", 2, "number of labels")
tf.app.flags.DEFINE_integer("D", 64, "dimensionality of embeddings")
tf.app.flags.DEFINE_integer("N", 50, "number of sketches")
tf.app.flags.DEFINE_integer("J", 100, "dimensionality of hidden layer")
tf.app.flags.DEFINE_integer("r", 2, "context size")
tf.app.flags.DEFINE_integer("bad_weight", 0.9, "weight for BAD instances" )
tf.app.flags.DEFINE_boolean("concat", False, "concatenating s_i and h_i for prediction")
tf.app.flags.DEFINE_boolean("train", True, "training model")
tf.app.flags.DEFINE_integer("epochs", 100, "training epochs")
tf.app.flags.DEFINE_boolean("shuffle", False, "shuffling training data before each epoch")
tf.app.flags.DEFINE_integer("checkpoint_freq", 1, "save model every x epochs")
tf.app.flags.DEFINE_integer("lstm_units", 100, "number of LSTM-RNN encoder units")
tf.app.flags.DEFINE_float("l2_scale", 0.0001, "L2 regularization constant")
tf.app.flags.DEFINE_float("keep_prob", 1, "keep probability for dropout during training (1: no dropout)")  # TODO fix
tf.app.flags.DEFINE_boolean("interactive", False, "interactive mode")
tf.app.flags.DEFINE_boolean("restore", False, "restoring last session from checkpoint")
tf.app.flags.DEFINE_integer("threads", 1, "number of threads")
FLAGS = tf.app.flags.FLAGS


def ef_single_state(inputs, labels, mask, seq_lens, vocab_size, K, D, N, J, L, r,
                    lstm_units, concat, window_size, keep_prob,
                    src_embeddings=None, tgt_embeddings=None, class_weights=None):
    """
    Single-state easy-first model with embeddings and optional LSTM-RNN encoder
    :param inputs:
    :param labels:
    :param vocab_size:
    :param K:
    :param D:
    :param N:
    :param J:
    :param L:
    :param r:
    :param lstm_units:
    :return:
    """

    def forward(x, y, mask, seq_lens, class_weights=None):
        """
        Compute a forward step for the easy first model and return loss and predictions for a batch
        :param x:
        :param y:
        :return:
        """
        batch_size = tf.shape(x)[0]
        with tf.name_scope("ef_fw"):
            with tf.name_scope("embedding"):
                if src_embeddings.table is None:
                    #print "Random src embeddings of dimensionality %d" % D
                    M_src = tf.get_variable(name="M_src", shape=[vocab_size, D],  # TODO vocab size
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
                    emb_size = D
                else:
                    M_src = tf.Variable(src_embeddings.table)
                    D_loaded = len(tgt_embeddings.table[0])
                    #print "Loading existing src embeddings of dimensionality %d" % D_loaded
                    emb_size = D_loaded


                if tgt_embeddings.table is None:
                    #print "Random tgt embeddings of dimensionality %d" % D
                    M_tgt = tf.get_variable(name="M_tgt", shape=[vocab_size, D],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
                    emb_size += D
                else:
                    M_tgt = tf.Variable(tgt_embeddings.table)  # TODO parameter if update embeddings or not
                    D_loaded = len(tgt_embeddings.table[0])
                    #print "Loading existing tgt embeddings of dimensionality %d" % D_loaded
                    emb_size += D_loaded

                if keep_prob < 1:  # dropout for word embeddings ("pervasive dropout")
                    #print "Dropping out word embeddings"
                    M_tgt = tf.nn.dropout(M_tgt, keep_prob)  # TODO make param
                    M_src = tf.nn.dropout(M_src, keep_prob)

                #print "embedding size", emb_size
                x_src, x_tgt = tf.split(2, 2, x)  # split src and tgt part of input
                emb_tgt = tf.nn.embedding_lookup(M_tgt, x_src, name="emg_tgt")  # batch_size x L x window_size x emb_size
                emb_src = tf.nn.embedding_lookup(M_src, x_tgt, name="emb_src")  # batch_size x L x window_size x emb_size
                emb_comb = tf.concat(2, [emb_src, emb_tgt], name="emb_comb") # batch_size x L x 2*window_size x emb_size
                emb = tf.reshape(emb_comb, [batch_size, -1, window_size*emb_size], name="emb") # batch_size x L x window_size*emb_size

            if lstm_units > 0:
                with tf.name_scope("lstm"):
                    # alternatively use LSTMCell supporting peep-holes and output projection
                    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_units, state_is_tuple=True)
                    if keep_prob < 1:
                        #print "Dropping out LSTM input and output"
                        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1, output_keep_prob=keep_prob)  # TODO make params, input is already dropped out
                    # see: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py
                    # Permuting batch_size and n_steps
                    remb = tf.transpose(emb, [1, 0, 2])  # L x batch_size x window_size*emb_size
                    # Reshaping to (n_steps*batch_size, n_input)
                    remb = tf.reshape(remb, [-1, window_size*emb_size])  # L*batch_size x window_size*emb_size
                    # Split to get a list of 'n_steps=L' tensors of shape (batch_size, window_size*emb_size)
                    remb = tf.split(0, L, remb)

                    rnn_outputs, rnn_states = tf.nn.rnn(cell, remb,
                                    sequence_length=seq_lens, dtype=tf.float32)
                    # 'outputs' is a list of output at every timestep (until L)
                    rnn_outputs = tf.pack(rnn_outputs)
                    rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])  # batch_size x L x lstm_units
                    H = rnn_outputs
                    state_size = lstm_units
            else:
                H = emb
                state_size = 2*window_size*emb_size

            with tf.name_scope("alpha"):
                w_z = tf.get_variable(name="w_z", shape=[J],
                                      initializer=tf.random_uniform_initializer(dtype=tf.float32))
                #V = tf.get_variable(name="V", shape=[J, L],
                #                    initializer=tf.random_uniform_initializer(dtype=tf.float32))
                V = tf.Variable(tf.random_uniform(shape=[J, L]), name="V", dtype=tf.float32)  # not shared, because L is different for each bucket
                W_hz = tf.get_variable(name="W_hz", shape=[state_size, J],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
                W_sz = tf.get_variable(name="W_sz", shape=[state_size*(2*r+1), J],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
                W_bz = tf.get_variable(name="W_bz", shape=[2*r+1, J],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))

            with tf.name_scope("beta"):
                W_hs = tf.get_variable(name="W_hs", shape=[state_size, state_size],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
                W_ss = tf.get_variable(name="W_ss", shape=[state_size*(2*r+1), state_size],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
                w_s = tf.get_variable(name="w_s", shape=[state_size],
                                      initializer=tf.random_uniform_initializer(dtype=tf.float32))

            with tf.name_scope("prediction"):
                w_p = tf.get_variable(name="w_p", shape=[K],
                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))

                wsp_size = state_size
                if concat:  # h_i and s_i are concatenated before prediction
                    wsp_size = state_size*2
                W_sp = tf.get_variable(name="W_sp", shape=[wsp_size, K],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))

            with tf.name_scope("paddings"):
                padding_s_col = tf.constant([[0, 0], [0, 0], [r, r]], name="padding_s_col")
                padding_b = tf.constant([[0, 0], [r, r]], name="padding_b")

            with tf.name_scope("sketches"):
                n = tf.constant(1, dtype=tf.int32, name="n")
                S = tf.zeros(shape=[batch_size, state_size, L], dtype=tf.float32)
                b = tf.ones(shape=[batch_size, L], dtype=tf.float32)
                b = b/L

            if keep_prob < 1:
                with tf.name_scope("dropout"):
                    #print "Dropping out in sketches"
                    W_hs_dropped = tf.nn.dropout(tf.ones_like(W_hs), keep_prob, name="W_hs_dropout")  # TODO make param
                    W_hs_mask = tf.get_default_graph().get_tensor_by_name("ef_model/dropout/W_hs_dropout/Floor:0")
                    W_ss_dropped = tf.nn.dropout(tf.ones_like(W_ss), keep_prob, name="W_ss_dropout")
                    W_ss_mask = tf.get_default_graph().get_tensor_by_name("ef_model/dropout/W_ss_dropout/Floor:0")
                    S_n_dropped = tf.nn.dropout(tf.ones_like(S), keep_prob, name="S_n_dropout")
                    S_n_mask = tf.get_default_graph().get_tensor_by_name("ef_model/dropout/S_n_dropout/Floor:0")


        def z_i(i):
            """
            Compute attention weight
            :param i:
            :return:
            """
            h_i = tf.slice(H, [0, i, 0], [batch_size, 1, state_size], name="h_i")
            h_i = tf.reshape(h_i, [batch_size, state_size])
            v_i = tf.slice(V, [0, i], [J, 1], name="v_i")

            S_padded = tf.pad(S, padding_s_col, "CONSTANT", name="S_padded")
            S_sliced = tf.slice(S_padded, [0, 0, i], [batch_size, state_size, 2*r+1])  # slice columns around i
            s_context = tf.reshape(S_sliced, [batch_size, state_size*(2*r+1)], name="s_context")  # batch_size x state_size*(2*r+1)

            b_padded = tf.pad(b, padding_b, "CONSTANT", name="padded")
            b_context = tf.slice(b_padded, [0, i-r+r], [batch_size, 2*r+1], name="b_sliced")

            _1 = tf.matmul(s_context, W_sz)
            _2 = tf.matmul(b_context, W_bz)
            _3 = tf.batch_matmul(h_i, W_hz)
            tanh = tf.tanh(_1 + _2 + _3 + w_z)
            z_i = tf.matmul(tanh, v_i)
            return z_i

        def alpha():
            """
            Compute attention weight for all words in sequence in batch
            :return:
            """
            z = []
            for i in np.arange(L):
                z.append(z_i(i))
            z_packed = tf.pack(z)
            rz = tf.reshape(z_packed, [batch_size, L])
            a_n = tf.nn.softmax(rz)
            return a_n

        def conv_r(S):
            """
            Extract r context columns around each column and concatenate
            :param S:
            :param r:
            :return:
            """
            # pad
            S_padded = tf.pad(S, padding_s_col, name="padded")
            # now gather indices of padded
            transposed_S = tf.transpose(S_padded, [2,1,0])  # batch is last dimension -> L x state_size x batch_size
            contexts = []
            for i in np.arange(r, L+r):
                # extract 2r+1 rows around i for each batch
                context_i = transposed_S[i-r:i+r+1, :, :]  # 2*r+1 x state_size x batch_size
                # concatenate
                context_i = tf.reshape(context_i, [(2*r+1)*state_size, batch_size])  # (2*r+1)*(state_size) x batch_size
                contexts.append(context_i)
            contexts = tf.pack(contexts)  # L x (2*r+1)*(state_size) x batch_size
            contexts = tf.transpose(contexts, [2, 0, 1]) # switch back: batch_size x L x (2*r+1)*(state_size)
            return contexts

        def sketch_step(n, b, S):
            """
            Compute the sketch vector and update the sketch according to attention over words
            :param n:
            :param b:
            :param S:
            :param H:
            :return:
            """
            # beta function
            a_n = alpha()  # batch_size x L
            b_n = tf.add(tf.mul(tf.sub(tf.cast(n, tf.float32), -1.), b), a_n)
            b_n = tf.div(b_n, tf.cast(n, tf.float32))
            h_avg = tf.batch_matmul(tf.expand_dims(a_n, [1]), H)  # batch_size x 1 x state_size
            h_avg = tf.reshape(h_avg, [batch_size, state_size])  # batch_size x state_size
            conv = conv_r(S)  # batch_size x L x state_size*(2*r+1)
            s_avg = tf.batch_matmul(tf.expand_dims(a_n, [1]), conv)  # batch_size x 1 x state_size*(2*r+1)
            s_avg = tf.reshape(s_avg, [batch_size, state_size*(2*r+1)])

            if keep_prob < 1:  # same dropout for all steps (http://arxiv.org/pdf/1512.05287v3.pdf)
                _1 = tf.matmul(h_avg, tf.mul(W_hs, W_hs_mask))
                _2 = tf.matmul(s_avg, tf.mul(W_ss, W_ss_mask))
            else:
                _1 = tf.matmul(h_avg, W_hs)
                _2 = tf.matmul(s_avg, W_ss)
            s_n = tf.nn.tanh(_1 + _2 + w_s)  # batch_size x state_size
            S_update = tf.batch_matmul(tf.expand_dims(s_n, [2]), tf.expand_dims(a_n, [1]))
            S_n = S + S_update
            if keep_prob < 1:
                S_n = tf.mul(S_n, S_n_mask)
            return n+1, b_n, S_n

        with tf.name_scope("sketching"):

            (final_n, final_b, final_S) = tf.while_loop(
                cond=lambda n, _1, _2: n <= N,
                body=sketch_step,
                loop_vars=(n, b, S)
            )

            def score(i):
                """
                Score the word at index i
                """
                # state vector for this word (column) across batch
                s_i = tf.slice(final_S, [0, 0, i], [batch_size, state_size, 1])
                # hidden representation for this word across batch
                h_i = tf.slice(H, [0, i, 0], [batch_size, 1, state_size], name="h_i")
                if concat:
                    # 1st dimension is batch size
                    s_i = tf.concat(1, [s_i, tf.transpose(h_i, [0,2,1])])
                l = tf.matmul(tf.reshape(s_i, [batch_size, -1]), W_sp) + w_p
                return l  # batch_size x K

        with tf.name_scope("scoring"):

            # avg word-level xent
            pred_labels = []
            losses = []

            # need weight matrix to multiply losses with:  weights = class_weights[y_words]
            # for whole batch
            # tf.equal
            # tf.select
            # tf.where
            if class_weights is not None:
                class_weights = tf.constant(class_weights, name="class_weights")

            for i in np.arange(L):  # compute score, probs and losses per word for whole batch
                word_label_score = score(i)
                word_label_probs = tf.nn.softmax(word_label_score)
                word_preds = tf.argmax(word_label_probs, 1)
                pred_labels.append(word_preds)
                y_words = tf.reshape(tf.slice(y, [0, i], [batch_size, 1]), [batch_size])
                y_words_full = tf.one_hot(y_words, depth=K, on_value=1.0, off_value=0.0)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(word_label_score,
                                                                                y_words_full)
                if class_weights is not None:
                    # see: https://github.com/lopuhin/skflow/blob/5c978498d24472bac44235964b6ab528ca952918/skflow/ops/losses_ops.py
                    label_weights = tf.reduce_mean(tf.mul(y_words_full, class_weights), 1)
                    cross_entropy = tf.mul(cross_entropy, label_weights)

                # TODO regularizer
                l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                loss = tf.add(cross_entropy, l2_loss)

                losses.append(loss)
            pred_labels = mask*tf.transpose(tf.pack(pred_labels), [1, 0])  # masked, batch_size x L
            losses = tf.reduce_mean(tf.cast(mask, tf.float32)*tf.transpose(tf.pack(losses), [1, 0]),
                                    1)  # masked, batch_size x 1

        return losses, pred_labels

    losses, predictions = forward(inputs, labels, mask, seq_lens, class_weights)
    return losses, predictions



# TODO quetch: outsource emb, b, init, deeper, regularizer...
def quetch(x, y, masks, seq_lens, vocab_size, K, D, N, J, L, r, lstm_units, concat, window_size, src_embeddings, tgt_embeddings, class_weights, l2_scale, keep_prob):
    batch_size = tf.shape(x)[0]
    with tf.name_scope("embedding"):
        if src_embeddings.table is None:
            #print "Random src embeddings of dimensionality %d" % D
            M_src = tf.get_variable(name="M_src", shape=[vocab_size, D],  # TODO vocab size
                            initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
            emb_size = D
        else:
            M_src = tf.Variable(src_embeddings.table)
            D_loaded = len(tgt_embeddings.table[0])
            #print "Loading existing src embeddings of dimensionality %d" % D_loaded
            emb_size = D_loaded


        if tgt_embeddings.table is None:
            #print "Random tgt embeddings of dimensionality %d" % D
            M_tgt = tf.get_variable(name="M_tgt", shape=[vocab_size, D],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
            emb_size += D
        else:
            M_tgt = tf.Variable(tgt_embeddings.table)  # TODO parameter if update embeddings or not
            D_loaded = len(tgt_embeddings.table[0])
            #print "Loading existing tgt embeddings of dimensionality %d" % D_loaded
            emb_size += D_loaded

        if keep_prob < 1:  # dropout for word embeddings ("pervasive dropout")
            #print "Dropping out word embeddings"
            M_tgt = tf.nn.dropout(M_tgt, keep_prob)  # TODO make param
            M_src = tf.nn.dropout(M_src, keep_prob)

        #print "embedding size", emb_size
        x_src, x_tgt = tf.split(2, 2, x)  # split src and tgt part of input
        emb_tgt = tf.nn.embedding_lookup(M_tgt, x_src, name="emg_tgt")  # batch_size x L x window_size x emb_size
        emb_src = tf.nn.embedding_lookup(M_src, x_tgt, name="emb_src")  # batch_size x L x window_size x emb_size
        emb_comb = tf.concat(2, [emb_src, emb_tgt], name="emb_comb") # batch_size x L x 2*window_size x emb_size
        emb = tf.reshape(emb_comb, [batch_size, -1, window_size*emb_size], name="emb") # batch_size x L x window_size*emb_size

    with tf.name_scope("mlp"):
        x = tf.reshape(emb, [-1, window_size*emb_size])
        W = tf.Variable(tf.random_uniform(shape=[window_size*emb_size, K]), name="W")
        a = tf.tanh(tf.matmul(x, W))  # batch_size*L, K
        logits = tf.reshape(a, [batch_size, L, K])
        pred_labels = masks*tf.argmax(logits, 2)  # batch_size x L
        y_full = tf.one_hot(y, depth=K, on_value=1.0, off_value=0.0)
        cross_entropy = tf.cast(masks, dtype=tf.float32)*tf.reduce_mean(tf.log(logits*y_full), 2)  # batch_size x L
        losses = cross_entropy
    return losses, pred_labels


class EasyFirstModel():
    """
    Neural easy-first model
    """
    def __init__(self, K, D, N, J, r, vocab_size, batch_size, optimizer, learning_rate,
                 max_gradient_norm, lstm_units, concat, buckets, window_size, src_embeddings,
                 tgt_embeddings, forward_only=False, class_weights=None, l2_scale=0.1,
                 keep_prob=0.5, model_dir="models/"):
        """
        Initialize the model
        :param K:
        :param D:
        :param N:
        :param J:
        :param r:
        :param lstm_units:
        :param vocab_size:
        :param batch_size:
        :param optimizer:
        :param learning_rate:
        :param max_gradient_norm:
        :param forward_only:
        :param buckets:
        :param src_embeddings
        :param tgt_embeddings
        :param l2_scale
        :param keep_prob
        :return:
        """
        self.K = K
        self.D = D
        self.N = N
        self.J = J
        self.r = r
        self.lstm_units = lstm_units
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.concat = concat
        self.window_size = window_size
        self.global_step = tf.Variable(0, trainable=False)
        optimizer_map = {"sgd": tf.train.GradientDescentOptimizer, "adam": tf.train.AdamOptimizer,
                        "adagrad": tf.train.AdagradOptimizer, "adadelta": tf.train.AdadeltaOptimizer,
                        "rmsprop": tf.train.RMSPropOptimizer, "momemtum": tf.train.MomentumOptimizer}
        self.optimizer = optimizer_map.get(optimizer,
                                           tf.train.GradientDescentOptimizer)(self.learning_rate)
        self.src_embeddings = src_embeddings
        self.tgt_embeddings = tgt_embeddings
        self.l2_scale = l2_scale
        self.keep_prob = keep_prob

        self.class_weights = class_weights if class_weights is not None else [1./K]*K

        if self.lstm_units > 0:
            print "Model with LSTM RNN encoder of %d units" % self.lstm_units
        else:
            if self.src_embeddings.table is None and self.tgt_embeddings.table is None:
                print "Model with simple embeddings of size %d" % self.D
            else:
                print "Model with simple embeddings of size %d (src) & %d (tgt)" % \
                      (self.src_embeddings.table.shape[0], self.tgt_embeddings.table.shape[0])


        if self.N > 0:
            print "Model with %d sketches" % self.N
        else:
            print "No sketches"

        if self.concat or self.N == 0:
            print "Concatenating H and S for predictions"

        if self.l2_scale > 0:
            print "L2 regularizer with weight %f" % self.l2_scale

        if self.keep_prob < 1:
            print "Dropout with p=%f" % self.keep_prob

        self.buckets = buckets
        print "Buckets:", self.buckets

        # seq2seq model with buckets: (for in AND output)
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/seq2seq_model.py#L31
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py#L970

        # prepare input feeds
        self.inputs = []
        self.labels = []
        self.masks = []
        self.seq_lens = []
        self.losses = []
        self.predictions = []
        for j, max_len in enumerate(self.buckets):
            self.inputs.append(tf.placeholder(tf.int32, shape=[None, max_len, 2*self.window_size], name="inputs{0}".format(j)))
            self.labels.append(tf.placeholder(tf.int32, shape=[None, max_len], name="labels{0}".format(j)))
            self.masks.append(tf.placeholder(tf.int64, shape=[None, max_len], name="masks{0}".format(j)))
            self.seq_lens.append(tf.placeholder(tf.int32, shape=[None], name="seq_lens{0}".format(j)))
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 else None): #, regularizer=tf.contrib.layers.l2_regularizer(self.l2_scale)):
                print "Initializing parameters for bucket with max len", max_len
                print self.inputs[j]
                bucket_losses, bucket_predictions = ef_single_state(self.inputs[j], self.labels[j],
                                                                     self.masks[j], self.seq_lens[j],
                                                                     vocab_size=vocab_size, K=self.K,
                                                                     D=self.D, N=max_len,  # TODO number of sketches is fixed to sequence length
                                                                     J=self.J, L=max_len, r=self.r, lstm_units=lstm_units,
                                                                     concat=self.concat, window_size=window_size,
                                                                     src_embeddings=src_embeddings, tgt_embeddings=tgt_embeddings,
                                                                     class_weights=class_weights,
                                                                     keep_prob=keep_prob)
                self.losses.append(bucket_losses)
                self.predictions.append(bucket_predictions)
        print "losses", self.losses  # list of tensors, one for each bucket
        print "predictions", self.predictions  # list of tensors, one for each bucket  # TODO maybe return?

        # gradients and update operation for training the model
        if not forward_only:
            params = tf.trainable_variables()
            print "params", [p.name for p in params]
            self.gradient_norms = []
            self.updates = []
            for j in xrange(len(buckets)):
                gradients = tf.gradients(tf.reduce_mean(self.losses[j], 0), params)  # batch normalization
                if max_gradient_norm > -1:
                    print "Clipping gradient to %f" % max_gradient_norm
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    self.gradient_norms.append(norm)
                    update = self.optimizer.apply_gradients(zip(clipped_gradients, params))
                    self.updates.append(update)

                else:
                    print "No gradient clipping"
                    self.gradient_norms.append(tf.global_norm(gradients))
                    update = self.optimizer.apply_gradients(zip(gradients, params))
                    self.updates.append(update)
            print "updates", self.updates
            print "grad norms", self.gradient_norms

        self.saver = tf.train.Saver(tf.all_variables())

        self.path = "%s/ef_single_state_K%d_D%d_N%d_J%d_r%d_vocab%d_batch%d_opt%s_lr%f_gradnorm%f" \
                    "_lstm%d_concat%r_window%d_weights%s_l2r%f_dropout%f.model" % \
                    (model_dir, K, D, N, J, r, vocab_size, batch_size, optimizer,
                     learning_rate, max_gradient_norm, lstm_units, concat, window_size,
                     "-".join([str(c) for c in class_weights]), l2_scale, keep_prob)
        print "Model path:", self.path

    def batch_update(self, session, bucket_id, inputs, labels, masks, seq_lens, forward_only=False):
        """
        Training step
        :param session:
        :param bucket_id:
        :param inputs:
        :param labels:
        :param forward_only:
        :return:
        """
        print "getting updates for bucket", bucket_id
        # get input feed for bucket
        input_feed = {}
        input_feed[self.inputs[bucket_id].name] = inputs
        input_feed[self.labels[bucket_id].name] = labels
        input_feed[self.masks[bucket_id].name] = masks
        input_feed[self.seq_lens[bucket_id].name] = seq_lens
        print "input_feed", input_feed.keys()

        if not forward_only:
            output_feed = [self.losses[bucket_id],
                           self.predictions[bucket_id]]
                          # self.updates[bucket_id],
                          # self.gradient_norms[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id], self.predictions[bucket_id]]
        print "output_feed", output_feed

        outputs = session.run(output_feed, input_feed)
        print "outputs", outputs

        predictions = []
        for seq_len, pred in zip(seq_lens, outputs[1]):
                predictions.append(pred[:seq_len].tolist())

        return outputs[0], predictions  # loss, predictions


def create_model(session, buckets, forward_only=False, src_embeddings=None, tgt_embeddings=None,
                 class_weights=None):
    """
    Create a model
    :param session:
    :param forward_only:
    :return:
    """
    model = EasyFirstModel(K=FLAGS.K, D=FLAGS.D, N=FLAGS.N, J=FLAGS.J, r=FLAGS.r,
                           vocab_size=FLAGS.vocab_size, batch_size=FLAGS.batch_size,
                           optimizer=FLAGS.optimizer, learning_rate=FLAGS.learning_rate,
                           max_gradient_norm=FLAGS.max_gradient_norm, lstm_units=FLAGS.lstm_units,
                           concat=FLAGS.concat, forward_only=forward_only, buckets=buckets,
                           src_embeddings=src_embeddings, tgt_embeddings=tgt_embeddings,
                           window_size=3, class_weights=class_weights, l2_scale=FLAGS.l2_scale,
                           keep_prob=1 if forward_only else FLAGS.keep_prob, model_dir=FLAGS.model_dir)

    checkpoint = tf.train.get_checkpoint_state(model.path)
    if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path) and FLAGS.restore:
        print "Reading model parameters from %s" % checkpoint.model_checkpoint_path
        model.saver.restore(session, checkpoint.model_checkpoint_path)
    else:
        print "Creating model with fresh parameters"
        session.run(tf.initialize_all_variables())
    return model


def train():
    """
    Train a model
    :return:
    """
    print "Training on %d thread(s)" % FLAGS.threads

    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=FLAGS.threads)) as sess:

        # load data and embeddings
        src_embeddings = load_embedding(FLAGS.src_embeddings) if FLAGS.src_embeddings != "" \
            else None
        tgt_embeddings = load_embedding(FLAGS.tgt_embeddings) if FLAGS.tgt_embeddings != "" \
            else None

        train_dir = FLAGS.data_dir+"/task2_en-de_training/train.basic_features_with_tags"  # TODO language as parameter
        dev_dir = FLAGS.data_dir+"/task2_en-de_dev/dev.basic_features_with_tags"

        train_feature_vectors, train_tgt_sentences, train_labels, train_label_dict, train_src_embeddings, train_tgt_embeddings = \
            load_data(train_dir, src_embeddings, tgt_embeddings, max_sent = FLAGS.max_train_data_size, train=True, labeled=True)
        dev_feature_vectors, dev_tgt_sentences, dev_labels, dev_label_dict = \
            load_data(dev_dir, train_src_embeddings, train_tgt_embeddings, train=False, labeled=True)  # use training vocab for dev

        if FLAGS.src_embeddings == "":
            src_embeddings = embedding.embedding(None, train_src_embeddings.word2id, train_src_embeddings.id2word, train_src_embeddings.UNK_id, train_src_embeddings.PAD_id, train_src_embeddings.end_id, train_src_embeddings.start_id)
        if FLAGS.tgt_embeddings == "":
            tgt_embeddings = embedding.embedding(None, train_tgt_embeddings.word2id, train_tgt_embeddings.id2word, train_tgt_embeddings.UNK_id, train_tgt_embeddings.PAD_id, train_tgt_embeddings.end_id, train_tgt_embeddings.start_id)

        X_train = train_feature_vectors
        Y_train = train_labels

        X_dev = dev_feature_vectors
        Y_dev = dev_labels

        src_vocab_size = len(train_src_embeddings.word2id)
        tgt_vocab_size = len(train_tgt_embeddings.word2id)
        print "src vocab size", src_vocab_size
        print "tgt vocab size", tgt_vocab_size

        print "Training on %d instances" % len(X_train)
        print "Validating on %d instances" % len(X_dev)

        print "Maximum sentence length (train):", max([len(y) for y in Y_train])
        print "Maximum sentence length (dev):", max([len(y) for y in Y_dev])


        print "Training data samples:", X_train[:3], Y_train[:3]
        class_weights = [1-FLAGS.bad_weight, FLAGS.bad_weight]  # TODO QE specific
        print "Weights for classes:", class_weights

        train_buckets, reordering_indexes, bucket_edges = buckets_by_length(np.asarray(train_feature_vectors),
                                                          np.asarray(train_labels), buckets=FLAGS.buckets, max_len=FLAGS.L, mode="pad")
        model = create_model(sess, bucket_edges[1:], False, src_embeddings, tgt_embeddings, class_weights)
        # train_buckets is dictionary, each train bucket contains X_train_padded, Y_train_padded, train_masks, train_seq_lens
        train_buckets_sizes = {i : len(indx) for i, indx in reordering_indexes.items()}
        for i in train_buckets.keys():
            X_train_padded, Y_train_padded, train_masks, train_seq_lens = train_buckets[i]
            total_number_of_pads = sum([bucket_edges[i+1]-l for l in train_seq_lens])
            print "Bucket no %d with max length %d: %d instances, avg length %f,  %f number of PADS in total" % (i, bucket_edges[i+1], train_buckets_sizes[i], np.average(train_seq_lens), total_number_of_pads)

        # training in epochs
        for epoch in xrange(FLAGS.epochs):
            current_sample = 0
            loss = 0.0
            train_predictions = []
            train_true = []
            start_time_epoch = time.time()

            # random bucket order
            bucket_ids = np.random.permutation(train_buckets.keys())
            print "Bucket training order", bucket_ids

            for bucket_id in train_buckets.keys():
                print "bucket", bucket_id
                bucket_xs, bucket_ys, bucket_masks, bucket_seq_lens = train_buckets[bucket_id]
                # random order of samples in batch
                order = np.random.permutation(len(bucket_xs))
                # split data in bucket into random batches of at most batch_size
                samples_per_batch = train_buckets_sizes[bucket_id]/float(FLAGS.batch_size)
                batch_ids = np.array_split(order, samples_per_batch)
                # make update on each batch
                for i, batch_samples in enumerate(batch_ids):
                    print "batch", i
                    x_batch = bucket_xs[batch_samples]
                    #print "x_batch", x_batch
                    y_batch = bucket_ys[batch_samples]
                    mask_batch = bucket_masks[batch_samples]
                    seq_lens_batch = bucket_seq_lens[batch_samples]

                    step_loss, predictions = model.batch_update(sess, bucket_id, x_batch, y_batch, mask_batch, seq_lens_batch, False)  # FIXME feed one batch for each bucket?

                    loss += np.sum(step_loss)  # sum over batch
                    #print i, x_i, _, y_i, predictions, step_loss, embeddings
                    train_predictions.extend(predictions)
                    train_true.extend(y_batch)  # needs to be stored because of random order
                    current_sample += len(x_batch)

            time_epoch = time.time() - start_time_epoch
            train_accuracy = accuracy(train_true, train_predictions)
            train_f1_1, train_f1_2 = f1s_binary(train_true, train_predictions)

            print "EPOCH %d: epoch time %fs, loss %f, train acc. %f, f1 prod %f (%f/%f) " % \
                  (epoch+1, time_epoch, loss, train_accuracy,  train_f1_1*train_f1_2, train_f1_1, train_f1_2)

            # eval on dev  # TODO only ever x timesteps
            """
            # put dev data in according bucket
            #for bucket_id, (source_size, target_size) in enumerate(_buckets):
                #if len(source_ids) < source_size and len(target_ids) < target_size:
                    #data_set[bucket_id].append([source_ids, target_ids])
                    #break


            step_loss, dev_predictions = model.batch_update(sess, X_dev, Y_dev, True)  # TODO switch off dropout (like for testing)

            train_accuracy = accuracy(Y_train, train_predictions)
            dev_accuracy = accuracy(Y_dev, dev_predictions)
            train_f1_1, train_f1_2 = f1s_binary(Y_train, train_predictions)
            dev_f1_1, dev_f1_2 = f1s_binary(Y_dev, dev_predictions)

            print "EPOCH %d: epoch time %fs, loss %f, train acc. %f, f1 prod %f (%f/%f), " \
                  "dev acc. %f, f1 prod %f (%f/%f)" % \
                (epoch+1, time_epoch, loss,
                 train_accuracy,  train_f1_1*train_f1_2, train_f1_1, train_f1_2, dev_accuracy,
                 dev_f1_1*dev_f1_2, dev_f1_1, dev_f1_2)
            """
            if epoch % FLAGS.checkpoint_freq == 0:
                model.saver.save(sess, model.path, global_step=model.global_step)



def test():
    """
    Test a model
    :return:
    """
    print "Testing"
    FLAGS.restore = True  # has to be loaded
    with tf.Session() as sess:
         # load data and embeddings

        if FLAGS.src_embeddings != "":
            src_embeddings = load_embedding(FLAGS.src_embeddings)
        else:
            src_train_vocab_file = FLAGS.data_dir+"/task2_en-de_training/train.basic_features_with_tags.vocab.src.pkl"
            print "Reading src vocabulary from %s" % src_train_vocab_file
            src_train_vocab = pkl.load(open(src_train_vocab_file, "rb"))
            src_word2id = {w: i for i, w in enumerate(src_train_vocab)}
            src_id2word = {i: w for w, i in src_word2id.items()}
            src_embeddings = embedding.embedding(None, src_word2id, src_id2word, 0, 1, 2, 3)

        if FLAGS.tgt_embeddings != "":
            tgt_embeddings = load_embedding(FLAGS.tgt_embeddings)
        else:
            tgt_train_vocab_file = FLAGS.data_dir+"/task2_en-de_training/train.basic_features_with_tags.vocab.tgt.pkl"
            print "Reading tgt vocabulary from %s" % tgt_train_vocab_file
            tgt_train_vocab = pkl.load(open(tgt_train_vocab_file, "rb"))
            tgt_word2id = {w: i for i, w in enumerate(tgt_train_vocab)}
            tgt_id2word = {i: w for w, i in tgt_word2id.items()}
            tgt_embeddings = embedding.embedding(None, tgt_word2id, tgt_id2word, 0, 1, 2, 3)

        test_dir = FLAGS.data_dir+"/task2_en-de_test/test.features"
        test_feature_vectors, test_tgt_sentences, test_labels, test_label_dict = \
            load_data(test_dir, src_embeddings, tgt_embeddings, train=False, labeled=False)

        # load model
        class_weights = [1./FLAGS.K for i in xrange(FLAGS.K)]  # TODO QE specific
        model = create_model(sess, buckets, True, src_embeddings, tgt_embeddings, class_weights)

        X_test = test_feature_vectors
        Y_test = test_labels

        #no_test_instances = 40
        #xs, ys = random_interdependent_data_with_len([no_test_instances], FLAGS.L, FLAGS.vocab_size,
        #                                             FLAGS.K)
        #X_test, Y_test = xs[0], ys[0]

        print "Testing on %d instances" % len(X_test)

        # eval
        loss = 0

        step_loss, test_predictions = model.batch_update(sess, X_test, Y_test, True)
        loss += np.sum(step_loss)

        test_accuracy = accuracy(Y_test, test_predictions)
        test_f1_1, test_f1_2 = f1s_binary(Y_test, test_predictions)

        print "Test loss %f, accuracy %f, f1 prod %f (%f / %f)" % (loss/len(X_test),
                                                                   test_accuracy,
                                                                   test_f1_1*test_f1_2,
                                                                   test_f1_1, test_f1_2)


def demo():
    """
    Test a model dynamically by reading input from stdin
    :return:
    """
    FLAGS.restore = True
    with tf.Session() as sess:
        # load embeddings
        if FLAGS.src_embeddings != "":
            src_embeddings = load_embedding(FLAGS.src_embeddings)
        else:
            src_train_vocab_file = FLAGS.data_dir+"/task2_en-de_training/train.basic_features_with_tags.vocab.src.pkl"
            print "Reading src vocabulary from %s" % src_train_vocab_file
            src_train_vocab = pkl.load(open(src_train_vocab_file, "rb"))
            src_word2id = {w: i for i, w in enumerate(src_train_vocab)}
            src_id2word = {i: w for w, i in src_word2id.items()}
            src_embeddings = embedding.embedding(None, src_word2id, src_id2word, 0, 1, 2, 3)

        if FLAGS.tgt_embeddings != "":
            tgt_embeddings = load_embedding(FLAGS.tgt_embeddings)
        else:
            tgt_train_vocab_file = FLAGS.data_dir+"/task2_en-de_training/train.basic_features_with_tags.vocab.tgt.pkl"
            print "Reading tgt vocabulary from %s" % tgt_train_vocab_file
            tgt_train_vocab = pkl.load(open(tgt_train_vocab_file, "rb"))
            tgt_word2id = {w: i for i, w in enumerate(tgt_train_vocab)}
            tgt_id2word = {i: w for w, i in tgt_word2id.items()}
            tgt_embeddings = embedding.embedding(None, tgt_word2id, tgt_id2word, 0, 1, 2, 3)

        # load model
        class_weights = [1./FLAGS.K for i in xrange(FLAGS.K)]  # TODO QE specific
        model = create_model(sess, True, src_embeddings, tgt_embeddings, class_weights)

        sys.stdout.write("Enter source and target sentence separated by '|'.\n")
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            inputs = sentence.split("|")
            src = inputs[0].strip()
            tgt = inputs[1].strip()

            # get word ids
            words_src = [src_embeddings.get_id(src_word) for src_word in src.split()]
            words_tgt = [tgt_embeddings.get_id(tgt_word) for tgt_word in tgt.split()]
            if len(words_tgt) > FLAGS.L:
                print "WARNING: target input too long, last %d words will not be classified" % \
                      (len(words_tgt)-FLAGS.L)

            window_size = 3
            # features for each word: context word on src and tgt
            X = []
            for i, w in enumerate(words_tgt):
                x = []
                context_size = int(math.floor(window_size/2.))
                for j in range(i-context_size, i+context_size+1):  # source context words
                    if j < 0 or j >= len(words_src):
                        x.append(src_embeddings.PAD_id)
                    else:
                        x.append(words_src[j])

                for j in range(i-context_size, i+context_size+1):  # target context words
                    if j < 0 or j >= len(words_tgt):
                        x.append(tgt_embeddings.PAD_id)
                    else:
                        x.append(words_tgt[j])
                X.append(x)

            Y = [0 for x in X]  # dummy labels

            step_loss, predictions = model.batch_update(sess, [X], [Y], True)
            outputs = predictions[0]

            while len(outputs) < len(words_tgt):  # can happen because of sentence length limit
                outputs.append(0)
            print "prediction: ",  zip(tgt.split(), outputs)
            sys.stdout.flush()
            sys.stdout.write("Enter source and target sentence separated by '|'.\n")
            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


def main(_):

    if not FLAGS.interactive:
        training = FLAGS.train

        if training:
            train()
        else:
            test()
    else:
        demo()

if __name__ == "__main__":
    tf.app.run()


# TODO
# - demo, dev
# - sent. F1 as loss?
# - QUETCH
# - modularization