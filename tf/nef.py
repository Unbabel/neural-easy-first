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
import os
from easy_first_model import *
from buckets import BucketFactory
import pdb

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
tf.app.flags.DEFINE_integer("batch_size", 5, #200,
                            "Batch size to use during training.")

tf.app.flags.DEFINE_integer("word_cutoff", 1, "Word cutoff.")

tf.app.flags.DEFINE_string("training_file",
                           "pos_tagging/data/en-ud-normalized_train_sample.conll.tagging",
                           #"pos_tagging/data/en-ud-normalized_train.conll.tagging",
                           "Training file.")
tf.app.flags.DEFINE_string("dev_file",
                           #"pos_tagging/data/en-ud-normalized_dev.conll.tagging",
                           "pos_tagging/data/en-ud-normalized_train_sample.conll.tagging",
                           "Dev file.")
tf.app.flags.DEFINE_string("model_dir", "pos_tagging/models", "Model directory")
tf.app.flags.DEFINE_string("sketch_dir", "pos_tagging/sketches",
                           "Directory where sketch dumps are stored")

tf.app.flags.DEFINE_float("max_gradient_norm", -1,
                          "max gradient norm for clipping (-1: no clipping)")

tf.app.flags.DEFINE_integer("buckets", 3, "number of buckets")
#tf.app.flags.DEFINE_integer("buckets", 10, "number of buckets")
tf.app.flags.DEFINE_string("embeddings",
                           "pos_tagging/data/embeddings/polyglot-en.pkl",
                           "path to word embeddings")

tf.app.flags.DEFINE_boolean("update_embeddings", False, "update the embeddings")
tf.app.flags.DEFINE_string("activation", "tanh", "activation function")
tf.app.flags.DEFINE_integer("embedding_size", 64,
                            "dimensionality of embeddings")
tf.app.flags.DEFINE_integer("hidden_size", 20, "dimensionality of hidden layers")
tf.app.flags.DEFINE_string("encoder", "lstm", "Encoder type: bilstm, lstm, or "
                           "feedforward")
tf.app.flags.DEFINE_integer("context_size", 2, "context size")

tf.app.flags.DEFINE_boolean("concatenate_last_layer", True,
                            "concatenating sketches and encoder states for prediction")
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

log_file_path = FLAGS.model_dir + os.sep + \
                str(datetime.datetime.now()).replace(" ", "-") + ".training.log"
logging.basicConfig(filename=log_file_path)
logger = logging.getLogger("NEF")
logger.setLevel(logging.INFO)



def create_model(session, buckets, vocabulary_size, num_labels,
                 is_train=True, embeddings=None,
                 label_weights=None):
    """
    Create a model
    :param session:
    :param forward_only:
    :return:
    """
    np.random.seed(123)
    tf.set_random_seed(123)
    model = EasyFirstModel(num_labels=num_labels,
                           embedding_size=FLAGS.embedding_size,
                           hidden_size=FLAGS.hidden_size,
                           context_size=FLAGS.context_size,
                           vocabulary_size=vocabulary_size,
                           encoder=FLAGS.encoder,
                           concatenate_last_layer=FLAGS.concatenate_last_layer,
                           batch_size=FLAGS.batch_size,
                           optimizer=FLAGS.optimizer,
                           learning_rate=FLAGS.learning_rate,
                           max_gradient_norm=FLAGS.max_gradient_norm,
                           keep_prob=FLAGS.keep_prob,
                           keep_prob_sketch=FLAGS.keep_prob_sketch,
                           label_weights=label_weights,
                           l2_scale=FLAGS.l2_scale,
                           l1_scale=FLAGS.l1_scale,
                           embeddings=embeddings,
                           update_embeddings=FLAGS.update_embeddings,
                           activation=FLAGS.activation,
                           buckets=buckets,
                           track_sketches=FLAGS.track_sketches,
                           model_dir=FLAGS.model_dir,
                           is_train=is_train)
    checkpoint = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path) \
       and FLAGS.restore:
        logger.info("Reading model parameters from %s" %
                    checkpoint.model_checkpoint_path)
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
        filepath_train = FLAGS.training_file
        filepath_dev = FLAGS.dev_file

        if FLAGS.embeddings == "":
            embeddings = None
        else:
            embeddings = load_embedding(FLAGS.embeddings)

        train_sentences, train_labels, train_label_dict, train_embeddings = \
            load_pos_data(filepath_train, embeddings, label_dict={}, train=True)

        # Use training vocab/labels for dev.
        dev_sentences, dev_labels, dev_label_dict = \
            load_pos_data(filepath_dev, train_embeddings,
                          label_dict=train_label_dict, train=False)

        vocab_size = train_embeddings.vocab_size()
        num_labels = len(train_label_dict)

        if FLAGS.embeddings == "":
            embeddings = embedding.Embedding(None, train_embeddings.word2id,
                                             train_embeddings.id2word,
                                             train_embeddings.UNK_id,
                                             train_embeddings.PAD_id,
                                             train_embeddings.end_id,
                                             train_embeddings.start_id)

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
        factory = BucketFactory()
        train_buckets = factory.build_buckets(np.asarray(train_sentences),
                                              np.asarray(train_labels),
                                              num_buckets=FLAGS.buckets)

        dev_buckets = factory.put_in_buckets(np.asarray(dev_sentences),
                                             np.asarray(dev_labels),
                                             reference_buckets=train_buckets)

        # create the model
        bucket_lengths = [bucket.data.max_length() for bucket in train_buckets]
        model = create_model(sess, bucket_lengths, vocab_size, num_labels,
                             is_train=True, embeddings=embeddings)

        # Training in epochs.
        best_valid = 0
        best_valid_epoch = 0
        for epoch in xrange(FLAGS.epochs):
            current_sample = 0
            loss = 0.0
            loss_reg = 0.0
            train_predictions = []
            train_true = []
            start_time_epoch = time.time()

            # Random bucket order.
            bucket_ids = np.random.permutation(range(len(train_buckets)))

            for bucket_id in bucket_ids:
                bucket = train_buckets[bucket_id]
                # Random order of samples in batch.
                order = np.random.permutation(bucket.data.num_sequences())
                # Split data in bucket into random batches of at most
                # batch_size.
                # If the bucket has fewer data than the batch size, there will
                # be only one batch.
                num_batches = np.ceil(bucket.data.num_sequences() /
                                      float(FLAGS.batch_size))
                batch_ids = np.array_split(order, num_batches)
                bucket_loss = 0
                bucket_loss_reg = 0
                # Make update on each batch.
                for i, batch_samples in enumerate(batch_ids):
                    batch_data = bucket.data.select(batch_samples)
                    # Loss for each instance in batch.
                    step_loss, predictions, step_loss_reg = \
                        model.batch_update(sess, bucket_id, batch_data, False)
                    loss_reg += np.sum(step_loss_reg)
                    loss += np.sum(step_loss)  # sum over batch
                    bucket_loss += np.sum(step_loss)
                    bucket_loss_reg += np.sum(step_loss_reg)
                    train_predictions.extend(predictions)
                    # needs to be stored because of random order
                    train_true.extend(batch_data.labels)

                logger.info("bucket %d - loss %0.2f - loss+reg %0.2f" %
                            (bucket_id,
                             bucket_loss / bucket.data.num_sequences(),
                             bucket_loss_reg / bucket.data.num_sequences()))

            train_accuracy = accuracy(train_true, train_predictions)
            time_epoch = time.time() - start_time_epoch

            logger.info("EPOCH %d: epoch time %fs, loss %f, train acc. %f" % \
                        (epoch+1, time_epoch, loss/len(train_labels),
                         train_accuracy))

            # Eval on dev (every epoch).
            start_time_valid = time.time()
            dev_loss = 0.0
            dev_predictions = []
            dev_true = []
            for bucket_id in range(len(dev_buckets)):
                bucket = dev_buckets[bucket_id]
                if bucket == None:
                    continue
                # Loss for whole bucket.
                step_loss, predictions, step_loss_reg = \
                    model.batch_update(sess, bucket_id,
                                       bucket.data,
                                       True)
                dev_predictions.extend(predictions)
                dev_true.extend(bucket.data.labels)
                dev_loss += np.sum(step_loss)
            time_valid = time.time() - start_time_valid
            dev_accuracy = accuracy(dev_true, dev_predictions)
            dev_f1_1, dev_f1_2 = f1s_binary(dev_true, dev_predictions)
            logger.info("EPOCH %d: validation time %fs, loss %f, dev acc. %f" %
                        (epoch+1, time_valid, dev_loss/len(dev_labels),
                         dev_accuracy))
            if dev_accuracy > best_valid:
                logger.info("NEW BEST!")
                best_valid = dev_accuracy
                best_valid_epoch = epoch+1
                # Save checkpoint.
                model.saver.save(sess, model.path,
                                 global_step=model.global_step,
                                 write_meta_graph=False)
            else:
                logger.info("current best: %f at epoch %d" %
                            (best_valid, best_valid_epoch))


        logger.info("Training finished after %d epochs. "
                    "Best validation result: %f at epoch %d." \
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
