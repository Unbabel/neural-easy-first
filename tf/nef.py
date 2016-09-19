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
from models import *

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
tf.app.flags.DEFINE_string("data_dir", "/mnt/data/datasets/tacl/data/WMT2016/roundtrip_translations/", "Data directory")
tf.app.flags.DEFINE_string("model_dir", "models/", "Model directory")
tf.app.flags.DEFINE_string("sketch_dir", "sketches/", "Directory where sketch dumps are stored")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_float("max_gradient_norm", -1, "maximum gradient norm for clipping (-1: no clipping)")
tf.app.flags.DEFINE_integer("L", 58, "maximum length of sequences")
tf.app.flags.DEFINE_integer("buckets", 10, "number of buckets")
tf.app.flags.DEFINE_string("src_embeddings", "../embeddings/polyglot-en.train_full.features.0.min20.extended.pkl", "path to source language embeddings")
tf.app.flags.DEFINE_string("tgt_embeddings", "../embeddings/polyglot-de.train_full.features.0.min20.extended.pkl", "path to target language embeddings")
#tf.app.flags.DEFINE_string("src_embeddings", "../data/WMT2016/embeddings/polyglot-en.train.basic_features_with_tags.7000.extended.pkl", "path to source language embeddings")
#tf.app.flags.DEFINE_string("tgt_embeddings", "../data/WMT2016/embeddings/polyglot-de.train.basic_features_with_tags.7000.extended.pkl", "path to target language embeddings")
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
tf.app.flags.DEFINE_boolean("track_sketches", False, "keep track of the sketches during learning")
FLAGS = tf.app.flags.FLAGS

logging.basicConfig(filename=FLAGS.model_dir+str(datetime.datetime.now()).replace(" ", "-")+".training.log")
logger = logging.getLogger("NEF")
logger.setLevel(logging.INFO)

class EasyFirstModel():
    """
    Neural easy-first model
    """
    def __init__(self, K, D, N, J, r, src_vocab_size, tgt_vocab_size, batch_size, optimizer, learning_rate,
                 max_gradient_norm, lstm_units, concat, buckets, window_size, src_embeddings,
                 tgt_embeddings, forward_only=False, class_weights=None, l2_scale=0,
                 keep_prob=1, keep_prob_sketch=1, model_dir="models/",
                 bilstm=True, model="ef_single_state", activation="tanh", l1_scale=0,
                 update_emb=True, track_sketches=False, is_train=False):
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
        if model == "seq2seq":
            model_func = seq2seq
            logger.info("Using seq2seq model")
        else:
            model_func = ef_single_state
            logger.info("Using neural easy first single state model")

        activation_func = tf.nn.tanh
        if activation == "relu":
            activation_func = tf.nn.relu
        elif activation == "sigmoid":
            activation_func = tf.nn.sigmoid
        logger.info("Activation function %s" % activation_func.__name__)

        self.track_sketches = track_sketches
        if track_sketches:
            logger.info("Tracking sketches")

        # prepare input feeds
        self.inputs = []
        self.labels = []
        self.masks = []
        self.seq_lens = []
        self.losses = []
        self.losses_reg = []
        self.predictions = []
        self.sketches_tfs = []
        self.keep_probs = []
        self.keep_prob_sketches = []
        self.is_trains = []
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
            self.keep_prob_sketches.append(tf.placeholder(tf.float32, name="keep_prob_sketch{0}".format(j)))
            self.keep_probs.append(tf.placeholder(tf.float32, name="keep_prob{0}".format(j)))
            self.is_trains.append(tf.placeholder(tf.bool, name="is_train{0}".format(j)))
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
                    keep_prob=self.keep_probs[j], keep_prob_sketch=self.keep_prob_sketches[j],
                    l2_scale=self.l2_scale, l1_scale=self.l1_scale,
                    bilstm=self.bilstm, activation=activation_func,
                    track_sketches=self.track_sketches, is_train=self.is_trains[j])

                self.losses_reg.append(bucket_losses_reg)
                self.losses.append(bucket_losses) # list of tensors, one for each bucket
                self.predictions.append(bucket_predictions)  # list of tensors, one for each bucket
                self.src_table = src_table  # shared for all buckets
                self.tgt_table = tgt_table
                if self.track_sketches:  # else sketches are just empty
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
        input_feed[self.keep_probs[bucket_id].name] = 1 if forward_only else self.keep_prob
        input_feed[self.keep_prob_sketches[bucket_id].name] = 1 if forward_only else self.keep_prob_sketch
        input_feed[self.is_trains[bucket_id].name] = not forward_only
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
        fetch the sketches and the attention for a single sample from the graph
        """
        input_feed = {}
        input_feed[self.inputs[bucket_id].name] = np.expand_dims(input, 0)  # batch_size = 1
        input_feed[self.labels[bucket_id].name] = np.expand_dims(label, 0)
        input_feed[self.masks[bucket_id].name] = np.expand_dims(mask, 0)
        input_feed[self.seq_lens[bucket_id].name] = np.expand_dims(seq_len, 0)
        input_feed[self.keep_probs[bucket_id].name] = 1.0
        input_feed[self.keep_prob_sketches[bucket_id].name] = 1.0
        input_feed[self.is_trains[bucket_id].name] = False

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
    np.random.seed(123)
    tf.set_random_seed(123)
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
                           model=FLAGS.model, activation=FLAGS.activation, l1_scale=FLAGS.l1_scale,
                           track_sketches=FLAGS.track_sketches)
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
        train_dir = FLAGS.data_dir+"/train_full.features"
        dev_dir = FLAGS.data_dir+"/dev.features"

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
        if FLAGS.track_sketches:
            train_corpus_id = 37
            sample_bucket_id = np.nonzero([train_corpus_id in train_reordering_indexes[b] for b in train_reordering_indexes.keys()])[0][0]
            sample_in_bucket_index = np.nonzero([i == train_corpus_id for i in train_reordering_indexes[sample_bucket_id]])[0]  # position of sample within bucket
            logger.info("Chosen sketch sample: corpus id %d, bucket %d, index in bucket %d" % (train_corpus_id, sample_bucket_id, sample_in_bucket_index))

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

                    if FLAGS.model == "ef_single_state" and FLAGS.track_sketches:
                        if bucket_id == sample_bucket_id and sample_in_bucket_index in batch_samples:
                            sample_in_batch_index = np.nonzero([i == sample_in_bucket_index for i in batch_samples])[0][0]
                            all_sketches = model.get_sketches_for_single_sample(
                                sess, bucket_id, x_batch[sample_in_batch_index],
                                y_batch[sample_in_batch_index],
                                mask_batch[sample_in_batch_index],
                                seq_lens_batch[sample_in_batch_index])
                            sample_sketch = np.squeeze(all_sketches, 1)
                            logger.info("true: %s", str(y_batch[sample_in_batch_index]))
                            logger.info("predicted: %s", str(predictions[sample_in_batch_index]))
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

        test_dir = FLAGS.data_dir+"/test.features"
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
        message = "Test time %fs, loss %f, test acc. %f, f1 prod %f (%f/%f) " % \
                  (time_valid, test_loss/len(test_labels), test_accuracy,
                   test_f1_1*test_f1_2, test_f1_1, test_f1_2)
        logger.info(message)
        print message


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
