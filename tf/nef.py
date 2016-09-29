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
from easy_first_model import *

"""
Tensorflow implementation of the neural easy-first model
- Single-State Model

Baseline model
- QUETCH
"""

# Flags
tf.app.flags.DEFINE_string("model", "ef_single_state",
                           "Model for training: rnn or ef_single_state")

tf.app.flags.DEFINE_boolean("train", True, "training model")
tf.app.flags.DEFINE_integer("epochs", 500, "training epochs")
tf.app.flags.DEFINE_integer("checkpoint_frequency", 100, "save model every x epochs")
tf.app.flags.DEFINE_float("l2_scale", 0, "L2 regularization constant")
tf.app.flags.DEFINE_float("l1_scale", 0, "L1 regularization constant")

tf.app.flags.DEFINE_string("optimizer", "adam",
                           "Optimizer [sgd, adam, adagrad, adadelta, momentum]")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 200,
                            "Batch size to use during training.")

tf.app.flags.DEFINE_integer("word_cutoff", 1, "Word cutoff.")

tf.app.flags.DEFINE_string("data_dir", "pos_tagging/data", "Data directory")
tf.app.flags.DEFINE_string("model_dir", "pos_tagging/models", "Model directory")
tf.app.flags.DEFINE_string("sketch_dir", "pos_tagging/sketches",
                           "Directory where sketch dumps are stored")

tf.app.flags.DEFINE_float("max_gradient_norm", -1,
                          "max gradient norm for clipping (-1: no clipping)")

tf.app.flags.DEFINE_integer("buckets", 10, "number of buckets")
tf.app.flags.DEFINE_string("embeddings",
                           "../data/WMT2016/embeddings/polyglot-en.train.basic_features_with_tags.7000.extended.pkl",
                           "path to word embeddings")

tf.app.flags.DEFINE_boolean("update_embeddings", False, "update the embeddings")
tf.app.flags.DEFINE_string("activation", "tanh", "activation function")
tf.app.flags.DEFINE_integer("embedding_size", 64,
                            "dimensionality of embeddings")
tf.app.flags.DEFINE_integer("hidden_size", 20, "dimensionality of hidden layer")
tf.app.flags.DEFINE_integer("lstm_size", 20, "number of LSTM-RNN encoder units")
tf.app.flags.DEFINE_boolean("bilstm", False, "bi-directional LSTM-RNN encoder")
tf.app.flags.DEFINE_integer("context_size", 2, "context size")

tf.app.flags.DEFINE_boolean("concatenate", True,
                            "concatenating s_i and h_i for prediction")
tf.app.flags.DEFINE_float("attention_discount_factor", 0.0,
                          "Attention discount factor")
tf.app.flags.DEFINE_float("attention_temperature", 1.0,
                          "Attention temperature")
tf.app.flags.DEFINE_float("keep_prob", 1.0,
                          "keep probability for dropout during training "
                          "(1: no dropout)")
tf.app.flags.DEFINE_float("keep_prob_sketch", 1.0,
                          "keep probability for dropout during sketching "
                          "(1: no dropout)")

tf.app.flags.DEFINE_boolean("interactive", False, "interactive mode")
tf.app.flags.DEFINE_boolean("restore", False,
                            "restoring last session from checkpoint")
tf.app.flags.DEFINE_integer("threads", 8, "number of threads")
tf.app.flags.DEFINE_boolean("track_sketches", False,
                            "keep track of the sketches during learning")
FLAGS = tf.app.flags.FLAGS

log_file_path = FLAGS.model_dir + \
                str(datetime.datetime.now()).replace(" ", "-") + ".training.log"
logging.basicConfig(filename=log_file_path)
logger = logging.getLogger("NEF")
logger.setLevel(logging.INFO)



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

    with tf.Session(config=tf.ConfigProto( \
        intra_op_parallelism_threads=FLAGS.threads)) as sess:

        # Load data and embeddings.
        train_dir = FLAGS.data_dir+"/task2_en-de_training/train.features"
        dev_dir = FLAGS.data_dir+"/task2_en-de_dev/dev.features"

        if FLAGS.embeddings == "":
            embeddings = None
        else:
            embeddings = load_embeddings(FLAGS.embeddings)

        train_feature_vectors, train_tgt_sentences, train_labels, train_label_dict, \
        train_src_embeddings, train_tgt_embeddings = load_data(train_dir, src_embeddings,
                                                               tgt_embeddings,
                                                               max_sent=FLAGS.max_train_data_size,
                                                               train=True, labeled=True)

        dev_feature_vectors, dev_tgt_sentences, dev_labels, dev_label_dict = \
            load_data(dev_dir, train_src_embeddings, train_tgt_embeddings, train=False,
                      labeled=True)  # use training vocab for dev

        if FLAGS.embeddings == "":
            embeddings = embedding.Embedding(None, train_embeddings.word2id,
                                             train_embeddings.id2word,
                                             train_embeddings.UNK_id,
                                             train_embeddings.PAD_id,
                                             train_embeddings.end_id,
                                             train_embeddings.start_id)
            vocab_size = FLAGS.vocab_size # Do we need this?

        logger.info("vocab size: %d" % vocab_size)

        logger.info("Training on %d instances" % len(train_labels))
        logger.info("Validating on %d instances" % len(dev_labels))

        logger.info("Maximum sentence length (train): %d" % \
                    max([len(y) for y in train_labels]))
        logger.info("Maximum sentence length (dev): %d" % \
                    max([len(y) for y in dev_labels]))

        maximum_sentence_length = max([len(y) for y in train_labels])

        # bucketing training and dev data

        # equal bucket sizes
        data_buckets, reordering_indexes, bucket_edges = buckets_by_length(
            np.asarray(train_feature_vectors),
            np.asarray(train_labels), buckets=FLAGS.buckets,
            max_len=maximum_sentence_length, mode="pad")
        train_buckets = data_buckets
        train_reordering_indexes = reordering_indexes

        # bucketing dev data
        dev_buckets, dev_reordering_indexes = put_in_buckets(np.asarray(dev_feature_vectors),
                                                             np.asarray(dev_labels),
                                                             buckets=bucket_edges)

        # create the model
        model = create_model(sess, bucket_edges, vocab_size, False, embeddings)

        train_buckets_sizes = {i: len(indx) \
                               for i, indx in train_reordering_indexes.items()}
        dev_buckets_sizes = {i: len(indx) \
                             for i, indx in dev_reordering_indexes.items()}


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
        message = "Test time %fs, loss %f, test acc. %f, f1 prod %f (%f/%f) " % \
                  (time_valid, test_loss/len(test_labels), test_accuracy,
                   test_f1_1*test_f1_2, test_f1_1, test_f1_2)
        logger.info(message)
        print message


def main(_):

    if not FLAGS.interactive:
        training = FLAGS.train

        if training:
            train()
        else:
            test()
    else:
        raise NonImplementedError

if __name__ == "__main__":
    tf.app.run()
