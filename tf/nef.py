# coding=utf-8

import tensorflow as tf
import numpy as np
import time
import sys
from utils import *
import math
from embedding import *
import logging
import datetime

"""
Tensorflow implementation of the neural easy-first model
- Single-State Model

Baseline model
- QUETCH
"""

# Flags
tf.app.flags.DEFINE_string("model", "ef_single_state", "Model for training: quetch or ef_single_state")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_string("optimizer", "adam", "Optimizer [sgd, adam, adagrad, adadelta, "
                                                    "momentum]")
tf.app.flags.DEFINE_integer("batch_size", 200,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("src_vocab_size", 10000, "Vocabulary size.")
tf.app.flags.DEFINE_integer("tgt_vocab_size", 10000, "Vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "../data/WMT2016/WMT2016", "Data directory")
tf.app.flags.DEFINE_string("model_dir", "models/", "Model directory")
tf.app.flags.DEFINE_string("sketch_dir", "sketches/", "Directory where sketch dumps are stored")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_float("max_gradient_norm", -1, "maximum gradient norm for clipping (-1: no clipping)")
tf.app.flags.DEFINE_integer("L", 58, "maximum length of sequences")
tf.app.flags.DEFINE_integer("buckets", 10, "number of buckets")
tf.app.flags.DEFINE_string("src_embeddings", "../data/WMT2016/embeddings/polyglot-en.train.basic_features_with_tags.7000.extended.pkl", "path to source language embeddings")
tf.app.flags.DEFINE_string("tgt_embeddings", "../data/WMT2016/embeddings/polyglot-de.train.basic_features_with_tags.7000.extended.pkl", "path to target language embeddings")
#tf.app.flags.DEFINE_string("src_embeddings", "../data/WMT2016/embeddings/polyglot-en.pkl", "path to source language embeddings")
#tf.app.flags.DEFINE_string("tgt_embeddings", "../data/WMT2016/embeddings/polyglot-de.pkl", "path to target language embeddings")
#tf.app.flags.DEFINE_string("src_embeddings", "", "path to source language embeddings")
#tf.app.flags.DEFINE_string("tgt_embeddings", "", "path to target language embeddings")
tf.app.flags.DEFINE_boolean("update_emb", True, "update the embeddings")
tf.app.flags.DEFINE_string("activation", "tanh", "activation function")
tf.app.flags.DEFINE_integer("K", 2, "number of labels")
tf.app.flags.DEFINE_integer("D", 64, "dimensionality of embeddings")
tf.app.flags.DEFINE_integer("N", 50, "number of sketches")
tf.app.flags.DEFINE_integer("J", 20, "dimensionality of hidden layer")
tf.app.flags.DEFINE_integer("r", 2, "context size")
tf.app.flags.DEFINE_float("bad_weight", 3.0, "weight for BAD instances" )
tf.app.flags.DEFINE_boolean("concat", True, "concatenating s_i and h_i for prediction")
tf.app.flags.DEFINE_boolean("train", True, "training model")
tf.app.flags.DEFINE_integer("epochs", 500, "training epochs")
tf.app.flags.DEFINE_integer("checkpoint_freq", 100, "save model every x epochs")
tf.app.flags.DEFINE_integer("lstm_units", 20, "number of LSTM-RNN encoder units")
tf.app.flags.DEFINE_boolean("bilstm", False, "bi-directional LSTM-RNN encoder")
tf.app.flags.DEFINE_float("l2_scale", 0, "L2 regularization constant")
tf.app.flags.DEFINE_float("l1_scale", 0, "L1 regularization constant")
tf.app.flags.DEFINE_float("keep_prob", 1 , "keep probability for dropout during training (1: no dropout)")
tf.app.flags.DEFINE_float("keep_prob_sketch", 1, "keep probability for dropout during sketching (1: no dropout)")
tf.app.flags.DEFINE_boolean("interactive", False, "interactive mode")
tf.app.flags.DEFINE_boolean("restore", False, "restoring last session from checkpoint")
tf.app.flags.DEFINE_integer("threads", 8, "number of threads")
FLAGS = tf.app.flags.FLAGS

logging.basicConfig(filename=FLAGS.model_dir+str(datetime.datetime.now()).replace(" ", "-")+".training.log")
logger = logging.getLogger("NEF")
logger.setLevel(logging.INFO)


def ef_single_state(inputs, labels, masks, seq_lens, src_vocab_size, tgt_vocab_size, K, D, N, J, L, r,
                    lstm_units, concat, window_size, keep_prob, keep_prob_sketch,
                    l2_scale, l1_scale, src_embeddings=None, tgt_embeddings=None,
                    class_weights=None, bilstm=True, activation=tf.nn.tanh,
                    update_emb=True):
    """
    Single-state easy-first model with embeddings and optional LSTM-RNN encoder
    :param inputs:
    :param labels:
    :param src_vocab_size:
    :param tgt_vocab_size:
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
        with tf.name_scope("embedding"):
            if src_embeddings.table is None:
                #print "Random src embeddings of dimensionality %d" % D
                M_src = tf.get_variable(name="M_src", shape=[src_vocab_size, D],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
                emb_size = D
            else:
                M_src = tf.get_variable(name="M_src",
                                shape=[src_embeddings.table.shape[0],
                                       src_embeddings.table.shape[1]],
                                initializer=tf.constant_initializer(src_embeddings.table),
                                trainable=update_emb)
                D_loaded = len(tgt_embeddings.table[0])
                #print "Loading existing src embeddings of dimensionality %d" % D_loaded
                emb_size = D_loaded


            if tgt_embeddings.table is None:
                #print "Random tgt embeddings of dimensionality %d" % D
                M_tgt = tf.get_variable(name="M_tgt", shape=[tgt_vocab_size, D],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
                emb_size += D
            else:
                M_tgt = tf.get_variable(name="M_tgt",
                    shape=[tgt_embeddings.table.shape[0],
                           tgt_embeddings.table.shape[1]],
                    initializer=tf.constant_initializer(tgt_embeddings.table),
                    trainable=update_emb)
                D_loaded = len(tgt_embeddings.table[0])
                #print "Loading existing tgt embeddings of dimensionality %d" % D_loaded
                emb_size += D_loaded

            if keep_prob < 1:  # dropout for word embeddings ("pervasive dropout")
                #print "Dropping out word embeddings"
                M_tgt = tf.nn.dropout(M_tgt, keep_prob)  # TODO make param
                M_src = tf.nn.dropout(M_src, keep_prob)

            #print "embedding size", emb_size
            x_tgt, x_src = tf.split(2, 2, x)  # split src and tgt part of input
            emb_tgt = tf.nn.embedding_lookup(M_tgt, x_tgt, name="emg_tgt")  # batch_size x L x window_size x emb_size
            emb_src = tf.nn.embedding_lookup(M_src, x_src, name="emb_src")  # batch_size x L x window_size x emb_size
            emb_comb = tf.concat(2, [emb_src, emb_tgt], name="emb_comb") # batch_size x L x 2*window_size x emb_size
            emb = tf.reshape(emb_comb, [batch_size, L, window_size*emb_size],
                             name="emb") # batch_size x L x window_size*emb_size

        with tf.name_scope("hidden"):
            if lstm_units > 0:

                fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, state_is_tuple=True)

                if bilstm:
                    with tf.name_scope("bi-lstm"):
                        bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, state_is_tuple=True)

                        if keep_prob < 1:
                            #print "Dropping out LSTM output"
                            fw_cell = \
                                tf.nn.rnn_cell.DropoutWrapper(
                                    fw_cell, input_keep_prob=1, output_keep_prob=keep_prob)  # TODO make params, input is already dropped out
                            bw_cell = \
                                tf.nn.rnn_cell.DropoutWrapper(
                                    bw_cell, input_keep_prob=1, output_keep_prob=keep_prob)

                        outputs, _ = \
                            tf.nn.bidirectional_dynamic_rnn(
                                fw_cell, bw_cell, emb, sequence_length=seq_lens,
                                dtype=tf.float32, time_major=False)
                        outputs = tf.concat(2, outputs)
                        state_size = 2*lstm_units  # concat of fw and bw lstm output
                else:
                    with tf.name_scope("lstm"):

                        if keep_prob < 1:
                            #print "Dropping out LSTM output"
                            fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                                fw_cell, input_keep_prob=1, output_keep_prob=keep_prob)  # TODO make params, input is already dropped out

                        outputs, _, = tf.nn.dynamic_rnn(
                            cell=fw_cell, inputs=emb, sequence_length=seq_lens,
                            dtype=tf.float32, time_major=False)
                        state_size = lstm_units

                H = outputs

            else:
                # fully-connected layer on top of embeddings to reduce size
                remb = tf.reshape(emb, [batch_size*L, window_size*emb_size])
                W_fc = tf.get_variable(name="W_fc", shape=[window_size*emb_size, J],  # TODO another param?
                              initializer=tf.contrib.layers.xavier_initializer(
                                  uniform=True, dtype=tf.float32))
                b_fc = tf.get_variable(shape=[J], initializer=tf.random_uniform_initializer(
                    dtype=tf.float32), name="b_fc")
                H = tf.reshape(activation(tf.matmul(remb, W_fc)+b_fc), [batch_size, L, J])
                state_size = J

        with tf.name_scope("sketching"):
            W_hss = tf.get_variable(name="W_hss", shape=[2*state_size*(2*r+1), state_size],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
            w_s = tf.get_variable(name="w_s", shape=[state_size],
                                      initializer=tf.random_uniform_initializer(dtype=tf.float32))
            w_z = tf.get_variable(name="w_z", shape=[J],
                                      initializer=tf.random_uniform_initializer(dtype=tf.float32))
            v = tf.get_variable(name="v", shape=[J, 1],
                                initializer=tf.random_uniform_initializer(dtype=tf.float32))
            W_hsz = tf.get_variable(name="W_hsz", shape=[2*state_size*(2*r+1), J],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))

            if keep_prob_sketch < 1:
                # create mask
                keep_prob_tensor = tf.convert_to_tensor(keep_prob_sketch, name="keep_prob_sketch")
                # see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_ops.py#L1078 (inverted dropout)
                #W_hss_mask = tf.to_float(tf.less(tf.random_uniform(tf.shape(W_hss)), keep_prob_tensor)) * tf.inv(keep_prob_tensor)
                W_hss_mask = tf.get_variable("W_hss_mask", shape=tf.shape(W_hss), initializer=tf.random_uniform_initializer())
                W_hss_mask = tf.to_float(tf.less(W_hss_mask, keep_prob_tensor)) * tf.inv(keep_prob_tensor)

            def z_j(j, padded_matrix):
                """
                Compute attention weight
                :param j:
                :return:
                """
                matrix_sliced = tf.slice(padded_matrix, [0, j, 0], [batch_size, 2*r+1, 2*state_size])
                matrix_context = tf.reshape(matrix_sliced, [batch_size, 2*state_size*(2*r+1)], name="s_context")  # batch_size x 2*state_size*(2*r+1)
                activ = activation(tf.matmul(matrix_context, W_hsz) + w_z)
                z_i = tf.matmul(activ, v)
                return z_i

            def alpha(sequence_len, padded_matrix):
                """
                Compute attention weight for all words in sequence in batch
                :return:
                """
                z = []
                for j in np.arange(sequence_len):
                    z.append(z_j(j, padded_matrix))
                z_packed = tf.pack(z)  # seq_len, batch_size, 1
                rz = tf.transpose(z_packed, [1, 0, 2])  # batch-major
                rz = tf.reshape(rz, [batch_size, sequence_len])
                a_n = tf.nn.softmax(rz)
                return a_n

            def conv_r(padded_matrix, r):
                """
                Extract r context columns around each column and concatenate
                :param padded_matrix: batch_size x L+(2*r) x 2*state_size
                :param r: context size
                :return:
                """
                # gather indices of padded
                time_major_matrix = tf.transpose(padded_matrix, [1, 2, 0])  # time-major  -> L x 2*state_size x batch_size
                contexts = []
                for j in np.arange(r, L+r):
                    # extract 2r+1 rows around i for each batch
                    context_j = time_major_matrix[j-r:j+r+1, :, :]  # 2*r+1 x 2*state_size x batch_size
                    # concatenate
                    context_j = tf.reshape(context_j, [(2*r+1)*2*state_size, batch_size])  # (2*r+1)*(state_size) x batch_size
                    contexts.append(context_j)
                contexts = tf.pack(contexts)  # L x (2*r+1)* 2*(state_size) x batch_size
                batch_major_contexts = tf.transpose(contexts, [2, 0, 1]) # switch back: batch_size x L x (2*r+1)*2(state_size) (batch-major)
                return batch_major_contexts

            def sketch_step(sketch_embedding_matrix):
                """
                Compute the sketch vector and update the sketch according to attention over words
                :param sketch_embedding_matrix: batch_size x L x 2*state_size (concatenation of H and S)
                :return:
                """
                sketch_embedding_matrix_padded = tf.pad(sketch_embedding_matrix, padding_hs_col, "CONSTANT", name="HS_padded")  # add column on right and left

                # beta function
                a_n = alpha(L, sketch_embedding_matrix_padded)  # batch_size x L
                conv = conv_r(sketch_embedding_matrix_padded, r)  # batch_size x L x 2*state_size*(2*r+1)
                hs_avg = tf.batch_matmul(tf.expand_dims(a_n, [1]), conv)  # batch_size x 1 x 2*state_size*(2*r+1)
                hs_avg = tf.reshape(hs_avg, [batch_size, 2*state_size*(2*r+1)])

                if keep_prob_sketch < 1:  # same dropout for all steps (http://arxiv.org/pdf/1512.05287v3.pdf)
                    a = tf.matmul(hs_avg, tf.mul(W_hss, W_hss_mask))
                else:
                    a = tf.matmul(hs_avg, W_hss)
                hs_n = activation(a + w_s)  # batch_size x state_size

                sketch_update = tf.batch_matmul(tf.expand_dims(a_n, [2]), tf.expand_dims(hs_n, [1]))  # batch_size x L x state_size
                return sketch_update

            S = tf.zeros(shape=[batch_size, L, state_size], dtype=tf.float32)
            HS = tf.concat(2, [H, S])
            sketches = []

            padding_hs_col = tf.constant([[0, 0], [r, r], [0, 0]], name="padding_hs_col")
            embedding_update = tf.zeros(shape=[batch_size, L, state_size])  # batch_size x L x state_size

            if N > 0:
                for i in np.arange(N):
                    sketch_update = sketch_step(HS)
                    HS += tf.concat(2, [embedding_update, sketch_update])
                    sketches.append(tf.split(2, 2, HS)[1])
            sketches_tf = tf.pack(sketches)

        with tf.name_scope("scoring"):

            w_p = tf.get_variable(name="w_p", shape=[K],
                                   initializer=tf.random_uniform_initializer(dtype=tf.float32))
            wsp_size = 2*state_size
            W_sp = tf.get_variable(name="W_sp", shape=[wsp_size, K],
                                   initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))

            def score(j, sketch_embedding_matrix):
                """
                Score the word at index j
                """
                # state vector for this word (column) across batch
                hs_j = tf.slice(sketch_embedding_matrix, [0, j, 0], [batch_size, 1, 2*state_size])
                l = tf.matmul(tf.reshape(hs_j, [batch_size, 2*state_size]), W_sp) + w_p
                return l  # batch_size x K

            # avg word-level xent
            pred_labels = []
            losses = []

            if class_weights is not None:
                class_weights = tf.constant(class_weights, name="class_weights")

            for i in np.arange(L):  # compute score, probs and losses per word for whole batch
                word_label_score = score(i, HS)
                word_label_probs = tf.nn.softmax(word_label_score)
                word_preds = tf.argmax(word_label_probs, 1)
                pred_labels.append(word_preds)
                y_words = tf.reshape(tf.slice(y, [0, i], [batch_size, 1]), [batch_size])
                y_words_full = tf.one_hot(y_words, depth=K, on_value=1.0, off_value=0.0)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(word_label_score,
                                                                                y_words_full)
                if class_weights is not None:
                    label_weights = tf.reduce_sum(tf.mul(y_words_full, class_weights), 1)
                    cross_entropy = tf.mul(cross_entropy, label_weights)

                loss = cross_entropy
                losses.append(loss)

            pred_labels = mask*tf.transpose(tf.pack(pred_labels), [1, 0])  # masked, batch_size x L
            losses = tf.reduce_mean(tf.cast(mask, tf.float32)*tf.transpose(tf.pack(losses), [1, 0]),
                                    1)  # masked, batch_size x 1
            losses_reg = losses
            if l2_scale > 0:
                weights_list = [W_hss, W_sp]  # M_src, M_tgt word embeddings not included
                l2_loss = tf.contrib.layers.apply_regularization(
                    tf.contrib.layers.l2_regularizer(l2_scale), weights_list=weights_list)
                losses_reg += l2_loss
            if l1_scale > 0:
                weights_list = [W_hss, W_sp]
                l1_loss = tf.contrib.layers.apply_regularization(
                    tf.contrib.layers.l1_regularizer(l1_scale), weights_list=weights_list)
                losses_reg += l1_loss

        return losses, losses_reg, pred_labels, M_src, M_tgt, sketches_tf

    losses, losses_reg, predictions, M_src, M_tgt, sketches_tf = forward(inputs, labels, masks, seq_lens, class_weights)
    return losses, losses_reg, predictions, M_src, M_tgt, sketches_tf


def quetch(inputs, labels, masks, src_vocab_size, tgt_vocab_size, K, D, J, L, window_size,
           src_embeddings, tgt_embeddings, class_weights, l2_scale, keep_prob, l1_scale,
           keep_prob_sketch=1,
           lstm_units=0, bilstm=False, concat=False, r=0, N=0, seq_lens=None,
           activation=tf.nn.tanh, update_emb=True):
    """
    QUETCH model for word-level QE predictions  (MLP based on embeddings)
    :param inputs:
    :param labels:
    :param masks:
    :param seq_lens:
    :param src_vocab_size:
    :param tgt_vocab_size:
    :param K:
    :param D:
    :param N:
    :param J:
    :param L:
    :param r:
    :param lstm_units:
    :param bilstm:
    :param concat:
    :param window_size:
    :param src_embeddings:
    :param tgt_embeddings:
    :param class_weights:
    :param l2_scale:
    :param l1_scale:
    :param keep_prob:
    :return:
    """
    batch_size = tf.shape(inputs)[0]
    with tf.name_scope("embedding"):
        if src_embeddings.table is None:
            M_src = tf.get_variable(name="M_src", shape=[src_vocab_size, D],
                            initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, dtype=tf.float32))
            emb_size = D
        else:
            M_src = tf.get_variable(name="M_src",
                                    shape=[src_embeddings.table.shape[0],
                                           src_embeddings.table.shape[1]],
                                    initializer=tf.constant_initializer(src_embeddings.table),
                                    trainable=update_emb)
            D_loaded = len(tgt_embeddings.table[0])
            emb_size = D_loaded


        if tgt_embeddings.table is None:
            M_tgt = tf.get_variable(name="M_tgt", shape=[tgt_vocab_size, D],
                            initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, dtype=tf.float32))
            emb_size += D
        else:
            M_tgt = tf.get_variable(name="M_tgt",
                                    shape=[tgt_embeddings.table.shape[0],
                                           tgt_embeddings.table.shape[1]],
                                    initializer=tf.constant_initializer(tgt_embeddings.table),
                                    trainable=update_emb)
            D_loaded = len(tgt_embeddings.table[0])
            emb_size += D_loaded

        #if keep_prob < 1:  # dropout for word embeddings ("pervasive dropout")
        #    M_tgt = tf.nn.dropout(M_tgt, keep_prob)  # TODO make param
        #    M_src = tf.nn.dropout(M_src, keep_prob)

        x_tgt, x_src = tf.split(2, 2, inputs)  # split src and tgt part of input
        emb_tgt = tf.nn.embedding_lookup(M_tgt, x_tgt, name="emg_tgt")  # batch_size x L x window_size x emb_size
        emb_src = tf.nn.embedding_lookup(M_src, x_src, name="emb_src")  # batch_size x L x window_size x emb_size
        emb_comb = tf.concat(2, [emb_src, emb_tgt], name="emb_comb") # batch_size x L x 2*window_size x emb_size
        emb = tf.reshape(emb_comb, [batch_size, -1, window_size*emb_size], name="emb") # batch_size x L x window_size*emb_size

        if keep_prob < 1:
            emb = tf.nn.dropout(emb, keep_prob=keep_prob, name="drop_emb")

    with tf.name_scope("mlp"):
        inputs = tf.reshape(emb, [-1, window_size*emb_size])
        W_1 = tf.get_variable(shape=[window_size*emb_size, J],
                              initializer=tf.contrib.layers.xavier_initializer(
                                  uniform=True, dtype=tf.float32, ), name="W_1")
        W_2 = tf.get_variable(shape=[J, K],
                              initializer=tf.contrib.layers.xavier_initializer(
                                  uniform=True, dtype=tf.float32), name="W_2")
        b_1 = tf.get_variable(shape=[J], initializer=tf.random_uniform_initializer(
            dtype=tf.float32), name="b_1")
        b_2 = tf.get_variable(shape=[K], initializer=tf.random_uniform_initializer(
            dtype=tf.float32), name="b_2")
        hidden = activation(tf.matmul(inputs, W_1) + b_1)  # batch_size*emb_size, J
        if keep_prob < 1:
            hidden = tf.nn.dropout(hidden, keep_prob)
        out = tf.matmul(hidden, W_2) + b_2  # batch_size*emb_size, K
        logits = tf.reshape(out, [batch_size, L, K])
        softmax = tf.nn.softmax(out)  # batch_size*L, K
        pred_labels = masks*tf.argmax(logits, 2)  # batch_size x L
        y_full = tf.one_hot(labels, depth=K, on_value=1.0, off_value=0.0)  # batch_size x L x K
        xent = -y_full*tf.log(tf.reshape(softmax, [batch_size, L, K]) + 1e-10)  # batch_size x L x K

        if class_weights is not None:
            class_weights = tf.constant(class_weights, name="class_weights")
            label_weights = tf.mul(y_full, class_weights)  # batch_size x L x K
            xent = tf.mul(xent, label_weights)

        cross_entropy = tf.cast(masks, dtype=tf.float32)*tf.reduce_mean(xent, 2)  # batch_size x L

        losses = cross_entropy
        losses_reg = losses

        if l2_scale > 0:
            weights_list = [W_1, W_2]  # M_src, M_tgt,
            l2_loss = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l2_regularizer(l2_scale), weights_list=weights_list)
            losses_reg += l2_loss
        if l1_scale > 0:
            weights_list = [W_1, W_2]  # M_src, M_tgt,
            l1_loss = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l1_regularizer(l1_scale), weights_list=weights_list)
            losses_reg += l1_loss

    return losses, losses_reg, pred_labels, M_src, M_tgt


class EasyFirstModel():
    """
    Neural easy-first model
    """
    def __init__(self, K, D, N, J, r, src_vocab_size, tgt_vocab_size, batch_size, optimizer, learning_rate,
                 max_gradient_norm, lstm_units, concat, buckets, window_size, src_embeddings,
                 tgt_embeddings, forward_only=False, class_weights=None, l2_scale=0,
                 keep_prob=1, keep_prob_sketch=1, model_dir="models/",
                 bilstm=True, model="ef_single_state", activation="tanh", l1_scale=0,
                 update_emb=True):
        """
        Initialize the model
        :param K:
        :param D:
        :param N:
        :param J:
        :param r:
        :param lstm_units:
        :param src_vocab_size:
        :param tgt_vocab_size:
        :param batch_size:
        :param optimizer:
        :param learning_rate:
        :param max_gradient_norm:
        :param forward_only:
        :param buckets:
        :param src_embeddings
        :param tgt_embeddings
        :param l2_scale
        :param l1_scale
        :param keep_prob
        :param keep_prob_sketch
        :param bilstm
        :param model
        :param update_emb
        :return:
        """
        self.K = K
        self.D = D
        self.N = N
        self.J = J
        self.r = r
        self.lstm_units = lstm_units
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
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
        self.l1_scale = l1_scale
        self.keep_prob = keep_prob
        self.keep_prob_sketch = keep_prob_sketch
        self.bilstm = bilstm
        self.max_gradient_norm = max_gradient_norm
        self.update_emb = update_emb
        self.model_dir = model_dir

        self.class_weights = class_weights if class_weights is not None else [1./K]*K

        self.path = "%s/%s_K%d_D%d_N%d_J%d_r%d_batch%d_opt%s_lr%0.4f_gradnorm%0.2f" \
                    "_lstm%d_concat%r_window%d_weights%s_l2r%0.4f_l1r%0.4f_dropout%0.2f_sketchdrop%0.2f_updateemb%s_srcvoc%d_tgtvoc%d.model" % \
                    (self.model_dir, model, self.K, self.D, self.N, self.J,
                     self.r, self.batch_size, optimizer,
                     self.learning_rate, self.max_gradient_norm, self.lstm_units,
                     self.concat, self.window_size,
                     "-".join([str(c) for c in class_weights]), self.l2_scale,
                     self.l1_scale, self.keep_prob, self.keep_prob_sketch,
                     self.update_emb, self.src_vocab_size, self.tgt_vocab_size)
        logger.info("Model path: %s"  % self.path)


        if self.lstm_units > 0:
            if self.bilstm:
                logger.info("Model with bi-directional LSTM RNN encoder of %d units" % self.lstm_units)
            else:
                logger.info("Model with uni-directional LSTM RNN encoder of %d units" % self.lstm_units)
        else:
            if self.src_embeddings.table is None and self.tgt_embeddings.table is None:
                logger.info("Model with simple embeddings of size %d" % self.D)
            else:
                logger.info("Model with simple embeddings of size %d (src) & %d (tgt)" % \
                      (self.src_embeddings.table.shape[0], self.tgt_embeddings.table.shape[0]))

        if update_emb:
            logger.info("Updating the embeddings during training")
        else:
            logger.info("Keeping the embeddings fixed")

        if self.N > 0:
            logger.info("Model with %d sketches" % self.N)
        else:
            logger.info("No sketches")
            self.concat = True

        if self.concat or self.N == 0:
            logger.info("Concatenating H and S for predictions")

        if self.l2_scale > 0:
            logger.info("L2 regularizer with weight %f" % self.l2_scale)

        if self.l1_scale > 0:
            logger.info("L1 regularizer with weight %f" % self.l1_scale)

        if forward_only:
            self.keep_prob = 1
            self.keep_prob_sketch = 1
        if self.keep_prob < 1:
            logger.info("Dropout with p=%f" % self.keep_prob)
        if self.keep_prob_sketch < 1:
            logger.info("Dropout during sketching with p=%f" % self.keep_prob_sketch)

        self.buckets = buckets

        buckets_path = self.path.split(".model", 2)[0]+".buckets.pkl"
        if self.buckets is not None:  # store bucket edges
            logger.info("Dumping bucket edges in %s" % buckets_path)
            pkl.dump(self.buckets, open(buckets_path, "wb"))
        else:  # load bucket edges
            logger.info("Loading bucket edges from %s" % buckets_path)
            self.buckets = pkl.load(open(buckets_path, "rb"))
        logger.info("Buckets: %s" % str(self.buckets))

        if model == "quetch":
            model_func = quetch
            logger.info("Using QUETCH model")
        else:
            model_func = ef_single_state
            logger.info("Using neural easy first single state model")

        activation_func = tf.nn.tanh
        if activation == "relu":
            activation_func = tf.nn.relu
        elif activation == "sigmoid":
            activation_func = tf.nn.sigmoid
        logger.info("Activation function %s" % activation_func.__name__)

        # prepare input feeds
        self.inputs = []
        self.labels = []
        self.masks = []
        self.seq_lens = []
        self.losses = []
        self.losses_reg = []
        self.predictions = []
        self.sketches_tfs = []
        for j, max_len in enumerate(self.buckets):
            self.inputs.append(tf.placeholder(tf.int32,
                                              shape=[None, max_len, 2*self.window_size],
                                              name="inputs{0}".format(j)))
            self.labels.append(tf.placeholder(tf.int32,
                                              shape=[None, max_len], name="labels{0}".format(j)))
            self.masks.append(tf.placeholder(tf.int64,
                                             shape=[None, max_len], name="masks{0}".format(j)))
            self.seq_lens.append(tf.placeholder(tf.int64,
                                                shape=[None], name="seq_lens{0}".format(j)))
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 else None):
                logger.info("Initializing parameters for bucket with max len %d" % max_len)
                bucket_losses, bucket_losses_reg, bucket_predictions, src_table, tgt_table, sketches = model_func(
                    inputs=self.inputs[j], labels=self.labels[j], masks=self.masks[j],
                    seq_lens=self.seq_lens[j], src_vocab_size=self.src_vocab_size,
                    tgt_vocab_size=self.tgt_vocab_size, K=self.K,
                    D=self.D, N=max_len,  # as many sketches as words in sequence
                    J=self.J, L=max_len, r=self.r, lstm_units=self.lstm_units,
                    concat=self.concat, window_size=self.window_size,
                    src_embeddings=self.src_embeddings, tgt_embeddings=self.tgt_embeddings,
                    class_weights=self.class_weights, update_emb=update_emb,
                    keep_prob=self.keep_prob, keep_prob_sketch=self.keep_prob_sketch,
                    l2_scale=self.l2_scale, l1_scale=self.l1_scale,
                    bilstm=self.bilstm, activation=activation_func)

                self.losses_reg.append(bucket_losses_reg)
                self.losses.append(bucket_losses) # list of tensors, one for each bucket
                self.predictions.append(bucket_predictions)  # list of tensors, one for each bucket
                self.src_table = src_table  # shared for all buckets
                self.tgt_table = tgt_table
                self.sketches_tfs.append(sketches)

        # gradients and update operation for training the model
        if not forward_only:
            params = tf.trainable_variables()
            self.gradient_norms = []
            self.updates = []
            for j in xrange(len(buckets)):
                gradients = tf.gradients(tf.reduce_mean(self.losses_reg[j], 0), params)  # batch normalization
                if max_gradient_norm > -1:
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    self.gradient_norms.append(norm)
                    update = self.optimizer.apply_gradients(zip(clipped_gradients, params))
                    self.updates.append(update)

                else:
                    self.gradient_norms.append(tf.global_norm(gradients))
                    update = self.optimizer.apply_gradients(zip(gradients, params))
                    self.updates.append(update)

        self.saver = tf.train.Saver(tf.all_variables())

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
        # get input feed for bucket
        input_feed = {}
        input_feed[self.inputs[bucket_id].name] = inputs
        input_feed[self.labels[bucket_id].name] = labels
        input_feed[self.masks[bucket_id].name] = masks
        input_feed[self.seq_lens[bucket_id].name] = seq_lens
        #print "input_feed", input_feed.keys()

        if not forward_only:
            output_feed = [self.losses[bucket_id],
                           self.predictions[bucket_id],
                           self.losses_reg[bucket_id],
                           self.updates[bucket_id],
                           self.gradient_norms[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id], self.predictions[bucket_id], self.losses_reg[bucket_id]]
        #print "output_feed", output_feed

        outputs = session.run(output_feed, input_feed)
        #print "outputs", outputs

        predictions = []
        for seq_len, pred in zip(seq_lens, outputs[1]):
                predictions.append(pred[:seq_len].tolist())

        return outputs[0], predictions, outputs[2]  # loss, predictions, regularized loss

    def get_sketches_for_single_sample(self, session, bucket_id, input, label, mask, seq_len):
        """
        fetch the sketches for a single sample from the graph
        """
        input_feed = {}
        input_feed[self.inputs[bucket_id].name] = np.expand_dims(input, 0)  # batch_size = 1
        input_feed[self.labels[bucket_id].name] = np.expand_dims(label, 0)
        input_feed[self.masks[bucket_id].name] = np.expand_dims(mask, 0)
        input_feed[self.seq_lens[bucket_id].name] = np.expand_dims(seq_len, 0)

        output_feed = [self.sketches_tfs[bucket_id]]
        outputs = session.run(output_feed, input_feed)

        return outputs[0]


def create_model(session, buckets, src_vocab_size, tgt_vocab_size,
                 forward_only=False, src_embeddings=None, tgt_embeddings=None,
                 class_weights=None):
    """
    Create a model
    :param session:
    :param forward_only:
    :return:
    """
    model = EasyFirstModel(K=FLAGS.K, D=FLAGS.D, N=FLAGS.N, J=FLAGS.J, r=FLAGS.r,
                           src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
                           batch_size=FLAGS.batch_size,
                           optimizer=FLAGS.optimizer, learning_rate=FLAGS.learning_rate,
                           max_gradient_norm=FLAGS.max_gradient_norm, lstm_units=FLAGS.lstm_units,
                           concat=FLAGS.concat, forward_only=forward_only, buckets=buckets,
                           src_embeddings=src_embeddings, tgt_embeddings=tgt_embeddings,
                           window_size=3, class_weights=class_weights, l2_scale=FLAGS.l2_scale,
                           keep_prob=FLAGS.keep_prob, keep_prob_sketch=FLAGS.keep_prob_sketch,
                           model_dir=FLAGS.model_dir, bilstm=FLAGS.bilstm, update_emb=FLAGS.update_emb,
                           model=FLAGS.model, activation=FLAGS.activation, l1_scale=FLAGS.l1_scale)
    checkpoint = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path) and FLAGS.restore:
        logger.info("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
        model.saver.restore(session, checkpoint.model_checkpoint_path)
    else:
        logger.info("Creating model with fresh parameters")
        session.run(tf.initialize_all_variables())
    return model


def print_config():
    logger.info("Configuration: %s" % str(FLAGS.__dict__["__flags"]))


def train():
    """
    Train a model
    :return:
    """
    print_config()
    logger.info("Training on %d thread(s)" % FLAGS.threads)

    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=FLAGS.threads)) as sess:

        # load data and embeddings
        train_dir = FLAGS.data_dir+"/task2_en-de_training/train.basic_features_with_tags"
        dev_dir = FLAGS.data_dir+"/task2_en-de_dev/dev.basic_features_with_tags"

        if FLAGS.src_embeddings == "":
            src_embeddings = None
        else:
            src_embeddings = load_embedding(FLAGS.src_embeddings)

        if FLAGS.tgt_embeddings == "":
            tgt_embeddings = None
        else:
            tgt_embeddings = load_embedding(FLAGS.tgt_embeddings)

        train_feature_vectors, train_tgt_sentences, train_labels, train_label_dict, \
        train_src_embeddings, train_tgt_embeddings = load_data(train_dir, src_embeddings,
                                                               tgt_embeddings,
                                                               max_sent=FLAGS.max_train_data_size,
                                                               train=True, labeled=True)

        dev_feature_vectors, dev_tgt_sentences, dev_labels, dev_label_dict = \
            load_data(dev_dir, train_src_embeddings, train_tgt_embeddings, train=False,
                      labeled=True)  # use training vocab for dev


        if FLAGS.src_embeddings == "":
            src_embeddings = embedding.Embedding(None, train_src_embeddings.word2id,
                                                 train_src_embeddings.id2word,
                                                 train_src_embeddings.UNK_id,
                                                 train_src_embeddings.PAD_id,
                                                 train_src_embeddings.end_id,
                                                 train_src_embeddings.start_id)
            src_vocab_size = FLAGS.src_vocab_size
        else:
            src_vocab_size = len(train_src_embeddings.word2id)

        if FLAGS.tgt_embeddings == "":
            tgt_embeddings = embedding.Embedding(None, train_tgt_embeddings.word2id,
                                                 train_tgt_embeddings.id2word,
                                                 train_tgt_embeddings.UNK_id,
                                                 train_tgt_embeddings.PAD_id,
                                                 train_tgt_embeddings.end_id,
                                                 train_tgt_embeddings.start_id)
            tgt_vocab_size = FLAGS.tgt_vocab_size
        else:
            tgt_vocab_size = len(train_tgt_embeddings.word2id)

        logger.info("src vocab size: %d" % src_vocab_size)
        logger.info("tgt vocab size: %d" % tgt_vocab_size)

        logger.info("Training on %d instances" % len(train_labels))
        logger.info("Validating on %d instances" % len(dev_labels))

        logger.info("Maximum sentence length (train): %d" % max([len(y) for y in train_labels]))
        logger.info("Maximum sentence length (dev): %d" % max([len(y) for y in dev_labels]))

        class_weights = [1, FLAGS.bad_weight]  # TODO QE specific
        logger.info("Weights for classes: %s" % str(class_weights))

        # bucketing training and dev data

        # equal bucket sizes
        data_buckets, reordering_indexes, bucket_edges = buckets_by_length(
            np.asarray(train_feature_vectors),
            np.asarray(train_labels), buckets=FLAGS.buckets,
            max_len=FLAGS.L, mode="pad")
        train_buckets = data_buckets
        train_reordering_indexes = reordering_indexes

        # bucketing dev data
        dev_buckets, dev_reordering_indexes = put_in_buckets(np.asarray(dev_feature_vectors),
                                                             np.asarray(dev_labels),
                                                             buckets=bucket_edges)

        # create the model
        model = create_model(sess, bucket_edges, src_vocab_size, tgt_vocab_size,
                             False, src_embeddings, tgt_embeddings, class_weights)

        train_buckets_sizes = {i: len(indx) for i, indx in train_reordering_indexes.items()}
        dev_buckets_sizes = {i: len(indx) for i, indx in dev_reordering_indexes.items()}


        logger.info("Creating buckets for training data:")
        for i in train_buckets.keys():
            X_train_padded, Y_train_padded, train_masks, train_seq_lens = train_buckets[i]
            total_number_of_pads = sum([bucket_edges[i]-l for l in train_seq_lens])
            logger.info("Bucket no %d with max length %d: %d instances, avg length %f,  " \
                  "%d number of PADS in total" % (i, bucket_edges[i], train_buckets_sizes[i],
                                                  np.average(train_seq_lens), total_number_of_pads))

        logger.info("Creating buckets for dev data:")
        for i in dev_buckets.keys():
            X_dev_padded, Y_dev_padded, dev_masks, dev_seq_lens = dev_buckets[i]
            total_number_of_pads = sum([bucket_edges[i]-l for l in dev_seq_lens])
            logger.info("Bucket no %d with max length %d: %d instances, avg length %f,  " \
                  "%d number of PADS in total" % (i, bucket_edges[i], dev_buckets_sizes[i],
                                                  np.average(dev_seq_lens), total_number_of_pads))

        # choose a training sample to analyse during sketching
        train_corpus_id = 38
        sample_bucket_id = np.nonzero([train_corpus_id in train_reordering_indexes[b] for b in train_reordering_indexes.keys()])[0][0]
        sample_in_bucket_index = np.nonzero([i == train_corpus_id for i in train_reordering_indexes[sample_bucket_id]])[0]  # position of sample within bucket
        print "Chose sketch sample: corpus id %d, bucket %d, index in bucket %d" % (train_corpus_id, sample_bucket_id, sample_in_bucket_index)

        # training in epochs
        best_valid = 0
        best_valid_epoch = 0
        for epoch in xrange(FLAGS.epochs):
            current_sample = 0
            loss = 0.0
            loss_reg = 0.0
            train_predictions = []
            train_true = []
            start_time_epoch = time.time()

            # random bucket order
            bucket_ids = np.random.permutation(train_buckets.keys())

            for bucket_id in bucket_ids:
                bucket_xs, bucket_ys, bucket_masks, bucket_seq_lens = train_buckets[bucket_id]
                # random order of samples in batch
                order = np.random.permutation(len(bucket_xs))
                # split data in bucket into random batches of at most batch_size
                if train_buckets_sizes[bucket_id] > FLAGS.batch_size:
                    number_of_batches = np.ceil(train_buckets_sizes[bucket_id]/float(FLAGS.batch_size))
                    batch_ids = np.array_split(order, number_of_batches)
                else:
                    batch_ids = [order]  # only one batch
                bucket_loss = 0
                bucket_loss_reg = 0
                # make update on each batch
                for i, batch_samples in enumerate(batch_ids):
                    x_batch = bucket_xs[batch_samples]
                    y_batch = bucket_ys[batch_samples]
                    mask_batch = bucket_masks[batch_samples]
                    seq_lens_batch = bucket_seq_lens[batch_samples]
                    step_loss, predictions, step_loss_reg = model.batch_update(sess, bucket_id,
                                                                               x_batch, y_batch,
                                                                               mask_batch,
                                                                               seq_lens_batch,
                                                                               False)  # loss for each instance in batch
                    loss_reg += np.sum(step_loss_reg)
                    loss += np.sum(step_loss)  # sum over batch
                    bucket_loss += np.sum(step_loss)
                    bucket_loss_reg += np.sum(step_loss_reg)
                    train_predictions.extend(predictions)
                    train_true.extend(y_batch)  # needs to be stored because of random order
                    current_sample += len(x_batch)

                    if FLAGS.model == "ef_single_state":
                        if bucket_id == sample_bucket_id and sample_in_bucket_index in batch_samples:
                            sample_in_batch_index = np.nonzero([i == sample_in_bucket_index for i in batch_samples])[0][0]
                            all_sketches = model.get_sketches_for_single_sample(
                                sess, bucket_id, x_batch[sample_in_batch_index],
                                y_batch[sample_in_batch_index],
                                mask_batch[sample_in_batch_index],
                                seq_lens_batch[sample_in_batch_index])
                            sample_sketch = np.squeeze(all_sketches, 1)
                            pkl.dump(sample_sketch, open("%s/sketches_sent%d_epoch%d.pkl" % (FLAGS.sketch_dir, train_corpus_id, epoch), "wb"))


                logger.info("bucket %d - loss %0.2f - loss+reg %0.2f" % (bucket_id,
                                                                   bucket_loss/len(bucket_xs),
                                                                   bucket_loss_reg/len(bucket_xs)))

            train_accuracy = accuracy(train_true, train_predictions)
            train_f1_1, train_f1_2 = f1s_binary(train_true, train_predictions)
            time_epoch = time.time() - start_time_epoch

            logger.info("EPOCH %d: epoch time %fs, loss %f, train acc. %f, f1 prod %f (%f/%f) " % \
                  (epoch+1, time_epoch, loss/len(train_labels), train_accuracy,
                   train_f1_1*train_f1_2, train_f1_1, train_f1_2))

            # eval on dev (every epoch)
            start_time_valid = time.time()
            dev_loss = 0.0
            dev_predictions = []
            dev_true = []
            for bucket_id in dev_buckets.keys():
                bucket_xs, bucket_ys, bucket_masks, bucket_seq_lens = dev_buckets[bucket_id]
                step_loss, predictions, step_loss_reg = model.batch_update(sess, bucket_id,
                                                                           bucket_xs, bucket_ys,
                                                                           bucket_masks,
                                                                           bucket_seq_lens,
                                                                           True)  # loss for whole bucket
                dev_predictions.extend(predictions)
                dev_true.extend(bucket_ys)
                dev_loss += np.sum(step_loss)
            time_valid = time.time() - start_time_valid
            dev_accuracy = accuracy(dev_true, dev_predictions)
            dev_f1_1, dev_f1_2 = f1s_binary(dev_true, dev_predictions)
            logger.info("EPOCH %d: validation time %fs, loss %f, dev acc. %f, f1 prod %f (%f/%f) " % \
                  (epoch+1, time_valid, dev_loss/len(dev_labels), dev_accuracy,
                   dev_f1_1*dev_f1_2, dev_f1_1, dev_f1_2))
            if dev_f1_1*dev_f1_2 > best_valid:
                logger.info("NEW BEST!")
                best_valid = dev_f1_1*dev_f1_2
                best_valid_epoch = epoch+1
                # save checkpoint
                model.saver.save(sess, model.path, global_step=model.global_step, write_meta_graph=False)
            else:
                logger.info("current best: %f at epoch %d" % (best_valid, best_valid_epoch))


        logger.info("Training finished after %d epochs. Best validation result: %f at epoch %d." \
              % (epoch+1, best_valid, best_valid_epoch))

        # dump final embeddings
        src_lookup, tgt_lookup = sess.run([model.src_table, model.tgt_table])
        src_embeddings.set_table(src_lookup)
        tgt_embeddings.set_table(tgt_lookup)
        src_embeddings.store("%s.%d.src.emb.pkl" % (model.path.split(".model")[0], epoch+1))
        tgt_embeddings.store("%s.%d.tgt.emb.pkl" % (model.path.split(".model")[0], epoch+1))


def test():
    """
    Test a model
    :return:
    """
    logger.info("Testing")
    FLAGS.restore = True  # has to be loaded
    with tf.Session() as sess:

        # load the embeddings
        src_embeddings = load_embedding(FLAGS.src_embeddings)
        tgt_embeddings = load_embedding(FLAGS.tgt_embeddings)
        src_vocab_size = src_embeddings.table.shape[0]
        tgt_vocab_size = tgt_embeddings.table.shape[0]

        test_dir = FLAGS.data_dir+"/task2_en-de_test/test.corrected_full_parsed_features_with_tags"
        test_feature_vectors, test_tgt_sentences, test_labels, test_label_dict = \
            load_data(test_dir, src_embeddings, tgt_embeddings, train=False,
                      labeled=True)

        # load model
        class_weights = [1, FLAGS.bad_weight]  #QE-specific
        model = create_model(sess, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, buckets=None, forward_only=True, src_embeddings=src_embeddings,
                             tgt_embeddings=tgt_embeddings, class_weights=class_weights)

        # bucketing test data
        bucket_edges = model.buckets
        test_buckets, test_reordering_indexes = put_in_buckets(np.asarray(test_feature_vectors),
                                                               np.asarray(test_labels),
                                                               buckets=bucket_edges)
        test_buckets_sizes = {i: len(indx) for i, indx in test_reordering_indexes.items()}

        logger.info("Creating buckets for test data:")
        for i in test_buckets.keys():
            X_test_padded, Y_test_padded, test_masks, test_seq_lens = test_buckets[i]
            total_number_of_pads = sum([bucket_edges[i]-l for l in test_seq_lens])
            logger.info("Bucket no %d with max length %d: %d instances, avg length %f,  " \
                  "%d number of PADS in total" % (i, bucket_edges[i], test_buckets_sizes[i],
                                                  np.average(test_seq_lens), total_number_of_pads))

        # eval on test
        start_time_valid = time.time()
        test_loss = 0.0
        test_predictions = []
        test_true = []
        for bucket_id in test_buckets.keys():
            bucket_xs, bucket_ys, bucket_masks, bucket_seq_lens = test_buckets[bucket_id]
            step_loss, predictions, step_loss_reg = model.batch_update(sess, bucket_id,
                                                                       bucket_xs, bucket_ys,
                                                                       bucket_masks,
                                                                       bucket_seq_lens,
                                                                       True)  # loss for whole bucket
            test_predictions.extend(predictions)
            test_true.extend(bucket_ys)
            test_loss += np.sum(step_loss)
        time_valid = time.time() - start_time_valid
        test_accuracy = accuracy(test_true, test_predictions)
        test_f1_1, test_f1_2 = f1s_binary(test_true, test_predictions)
        logger.info("Test time %fs, loss %f, dev acc. %f, f1 prod %f (%f/%f) " % \
              (time_valid, test_loss/len(test_labels), test_accuracy,
               test_f1_1*test_f1_2, test_f1_1, test_f1_2))


def demo():
    """
    Test a model dynamically by reading input from stdin
    :return:
    """
    FLAGS.restore = True
    with tf.Session() as sess:
        # load the embeddings
        src_embeddings = load_embedding(FLAGS.src_embeddings)
        tgt_embeddings = load_embedding(FLAGS.tgt_embeddings)
        src_vocab_size = src_embeddings.table.shape[0]
        tgt_vocab_size = tgt_embeddings.table.shape[0]


        # load model
        class_weights = [1, FLAGS.bad_weight]  # QE-specific
        model = create_model(sess, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
                             buckets=None, forward_only=True, src_embeddings=src_embeddings,
                             tgt_embeddings=tgt_embeddings, class_weights=class_weights)

        # bucketing test data
        bucket_edges = model.buckets

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

            bucket_id = 0
            for bucket_edge in bucket_edges:
                if len(X) > bucket_edge:
                    bucket_id += 1

            # pad sentence and create mask
            seq_len = len(X)
            mask = np.zeros((1,bucket_edges[bucket_id]))
            X_padded = np.zeros((1, bucket_edges[bucket_id], 2*window_size))
            for i, x in enumerate(X):
                mask[0][i] = 1
                X_padded[0][i] = np.asarray(x).T
            Y_padded = np.zeros_like(mask)  # dummy labels

            # get predictions
            step_loss, predictions, step_loss_reg = model.batch_update(sess, bucket_id, X_padded,
                                                                       Y_padded, mask, [seq_len],
                                                                       True)
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
# - language als parameter
# - modularization
# - replace numbers by <NUM>?
