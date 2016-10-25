# coding=utf-8
'''This module allows training and testing a Tensorflow implementation of the
neural easy-first model.'''

import tensorflow as tf
import numpy as np
import time
import logging
import datetime
import os
from easy_first_model import EasyFirstModel
from embedding import Embedding
from dataset import DatasetReader
from buckets import BucketFactory
from evaluator import Evaluator

# Flags.
tf.app.flags.DEFINE_boolean("train", True, "True if training, False if "
                            "testing.")

tf.app.flags.DEFINE_integer("epochs", 50, "Number of training epochs.")
tf.app.flags.DEFINE_integer("checkpoint_frequency", 10, "Save model every N "
                            "epochs.")
tf.app.flags.DEFINE_string("optimizer", "adam",
                           "Optimizer [sgd, adam, adagrad, adadelta, "
                           "momentum].")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 200,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_float("max_gradient_norm", -1,
                          "Max gradient norm for clipping (-1: no clipping).")
tf.app.flags.DEFINE_integer("buckets", 10, "Number of buckets.")
tf.app.flags.DEFINE_float("l2_scale", 0, "L2 regularization constant")
tf.app.flags.DEFINE_float("l1_scale", 0, "L1 regularization constant")
tf.app.flags.DEFINE_float("keep_prob", 1.0,
                          "Keep probability for dropout during training "
                          "(1: no dropout).")
tf.app.flags.DEFINE_float("keep_prob_sketch", 1.0,
                          "Keep probability for dropout during sketching "
                          "(1: no dropout).")

tf.app.flags.DEFINE_string("training_file",
                           "pos_tagging/data/en-ud-normalized_train.conll."
                           "tagging",
                           "Path to training file.")
tf.app.flags.DEFINE_string("dev_file",
                           "pos_tagging/data/en-ud-normalized_dev.conll."
                           "tagging",
                           "Path to dev file.")
tf.app.flags.DEFINE_string("test_file",
                           "pos_tagging/data/en-ud-normalized_test.conll."
                           "tagging",
                           "Path to test file.")
tf.app.flags.DEFINE_string("embeddings",
                           "pos_tagging/data/embeddings/polyglot-en.pkl",
                           "Path to word embeddings.")
tf.app.flags.DEFINE_string("model_dir", "pos_tagging/models", "Path to folder "
                           "where the model is stored.")

tf.app.flags.DEFINE_integer("word_cutoff", 1, "Word cutoff.")
tf.app.flags.DEFINE_integer("max_sentence_length", 50,
                            "Discard sentences longer than this length (both "
                            "for training and dev files. Set this to -1 to "
                            "keep all the sentences.")
tf.app.flags.DEFINE_boolean("update_embeddings", True,
                            "True to update the embeddings; False otherwise.")
tf.app.flags.DEFINE_string("activation", "tanh", "Activation function.")
tf.app.flags.DEFINE_integer("embedding_size", 64,
                            "Dimensionality of embeddings.")
tf.app.flags.DEFINE_integer("hidden_size", 20,
                            "Dimensionality of hidden layers.")
tf.app.flags.DEFINE_string("encoder", "bilstm",
                           "Encoder type: bilstm, lstm, or feedforward")
tf.app.flags.DEFINE_integer("context_size", 2, "context size")
tf.app.flags.DEFINE_integer("num_sketches", -1, "Number of sketches. Set to "
                            "-1 to let the number of sketches to equal the "
                            "number of words.")
tf.app.flags.DEFINE_boolean("concatenate_last_layer", True,
                            "Concatenate sketches and encoder states for "
                            "prediction")
tf.app.flags.DEFINE_float("attention_discount_factor", 0.0,
                          "Attention discount factor.")
tf.app.flags.DEFINE_float("attention_temperature", 1.0,
                          "Attention temperature.")

tf.app.flags.DEFINE_boolean("interactive", False, "Set interactive mode.")
tf.app.flags.DEFINE_boolean("restore", False,
                            "Restore last session from checkpoint.")
tf.app.flags.DEFINE_integer("threads", 8, "Number of threads.")
tf.app.flags.DEFINE_boolean("track_sketches", False,
                            "Keep track of the sketches during learning.")
tf.app.flags.DEFINE_string("sketch_dir", "pos_tagging/sketches",
                           "Directory where sketch dumps are stored")

FLAGS = tf.app.flags.FLAGS

# Set logging file.
LOG_FILE = FLAGS.model_dir + os.sep + \
           str(datetime.datetime.now()).replace(" ", "-") + ".training.log"
logging.basicConfig(filename=LOG_FILE)
LOGGER = logging.getLogger("NEF")
LOGGER.setLevel(logging.INFO)

def print_config():
    '''Print all the flags.'''
    LOGGER.info("Configuration: %s", str(FLAGS.__dict__["__flags"]))

def create_model(session, buckets, vocabulary_size, num_labels, is_train=True,
                 embeddings=None, label_weights=None):
    '''Create a new easy-first model.'''
    np.random.seed(123)
    tf.set_random_seed(123)
    model = EasyFirstModel(num_labels=num_labels,
                           embedding_size=FLAGS.embedding_size,
                           hidden_size=FLAGS.hidden_size,
                           context_size=FLAGS.context_size,
                           vocabulary_size=vocabulary_size,
                           num_sketches=FLAGS.num_sketches,
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
        LOGGER.info("Reading model parameters from %s",
                    checkpoint.model_checkpoint_path)
        model.saver.restore(session, checkpoint.model_checkpoint_path)
    else:
        LOGGER.info("Creating model with fresh parameters")
        session.run(tf.initialize_all_variables())
    return model

def train():
    '''Train a model.'''
    print_config()
    LOGGER.info("Training on %d thread(s)", FLAGS.threads)

    with tf.Session(config=tf.ConfigProto( \
        intra_op_parallelism_threads=FLAGS.threads)) as sess:

        # Load data and embeddings.
        filepath_train = FLAGS.training_file
        filepath_dev = FLAGS.dev_file

        reader = DatasetReader()
        if FLAGS.embeddings == "":
            embeddings = None
        else:
            embeddings = reader.load_embeddings(FLAGS.embeddings)

        max_sentence_length = FLAGS.max_sentence_length
        train_sentences, train_labels, train_label_dict, train_embeddings = \
            reader.load_data(filepath_train,
                             embeddings=embeddings,
                             max_length=max_sentence_length,
                             label_dict=None,
                             train=True)

        # Use training vocab/labels for dev.
        dev_sentences, dev_labels, _ = \
            reader.load_data(filepath_dev,
                             embeddings=train_embeddings,
                             max_length=max_sentence_length,
                             label_dict=train_label_dict,
                             train=False)

        vocabulary_size = train_embeddings.vocabulary_size()
        num_labels = len(train_label_dict)

        if FLAGS.embeddings == "":
            embeddings = Embedding(None,
                                   train_embeddings.word2id,
                                   train_embeddings.id2word,
                                   train_embeddings.unk_id,
                                   train_embeddings.pad_id,
                                   train_embeddings.end_id,
                                   train_embeddings.start_id)

        LOGGER.info("Vocabulary size: %d", vocabulary_size)
        LOGGER.info("Training on %d instances", len(train_labels))
        LOGGER.info("Validating on %d instances", len(dev_labels))
        LOGGER.info("Maximum sentence length (train): %d",
                    max([len(y) for y in train_labels]))
        LOGGER.info("Maximum sentence length (dev): %d",
                    max([len(y) for y in dev_labels]))

        # Bucket training and dev data (equal bucket sizes).
        factory = BucketFactory()
        train_buckets = factory.build_buckets(np.asarray(train_sentences),
                                              np.asarray(train_labels),
                                              num_buckets=FLAGS.buckets)

        dev_buckets = factory.put_in_buckets(np.asarray(dev_sentences),
                                             np.asarray(dev_labels),
                                             reference_buckets=train_buckets)

        # Create the model.
        bucket_lengths = [bucket.data.max_length() for bucket in train_buckets]
        model = create_model(sess, bucket_lengths, vocabulary_size, num_labels,
                             is_train=True, embeddings=embeddings)

        # Training in epochs.
        evaluator = Evaluator()
        best_valid = 0
        best_valid_epoch = 0
        for epoch in xrange(FLAGS.epochs):
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
                for batch_samples in batch_ids:
                    batch_data = bucket.data.select(batch_samples)
                    # Loss for each instance in batch.
                    step_loss, predictions, step_loss_reg = \
                        model.batch_update(sess, bucket_id, batch_data, False)
                    loss_reg += np.sum(step_loss_reg)
                    loss += np.sum(step_loss)  # Sum over batch.
                    bucket_loss += np.sum(step_loss)
                    bucket_loss_reg += np.sum(step_loss_reg)
                    train_predictions.extend(predictions)
                    # Needs to be stored because of random order.
                    train_true.extend(batch_data.labels)

                LOGGER.info("bucket %d - loss %0.5f - loss+reg %0.5f",
                            bucket_id,
                            bucket_loss / bucket.data.num_sequences(),
                            bucket_loss_reg / bucket.data.num_sequences())

            train_accuracy = evaluator.accuracy(train_true, train_predictions)
            time_epoch = time.time() - start_time_epoch

            LOGGER.info("EPOCH %d: epoch time %fs, loss %f, loss+reg %f, "
                        "train acc. %f",
                        epoch+1, time_epoch, loss/len(train_labels),
                        loss_reg/len(train_labels), train_accuracy)

            # Eval on dev (every epoch).
            start_time_valid = time.time()
            dev_loss = 0.0
            dev_predictions = []
            dev_true = []
            for bucket_id in xrange(len(dev_buckets)):
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
            dev_accuracy = evaluator.accuracy(dev_true, dev_predictions)
            LOGGER.info("EPOCH %d: validation time %fs, loss %f, dev acc. %f",
                        epoch+1, time_valid, dev_loss/len(dev_labels),
                        dev_accuracy)
            if dev_accuracy > best_valid:
                LOGGER.info("NEW BEST!")
                best_valid = dev_accuracy
                best_valid_epoch = epoch+1
                # Save checkpoint.
                model.saver.save(sess, model.path,
                                 global_step=model.global_step,
                                 write_meta_graph=False)
            else:
                LOGGER.info("current best: %f at epoch %d",
                            best_valid, best_valid_epoch)

        LOGGER.info("Training finished after %d epochs. "
                    "Best validation result: %f at epoch %d.",
                    FLAGS.epochs, best_valid, best_valid_epoch)

def test():
    '''Test the model.'''
    LOGGER.info("Testing")
    FLAGS.restore = True  # The model has to be loaded.
    with tf.Session() as sess:

        # Load the embeddings.
        reader = DatasetReader()
        embeddings = reader.load_embeddings(FLAGS.embeddings)

        filepath_test = FLAGS.test_file
        max_sentence_length = FLAGS.max_sentence_length
        # Fow now, we need this step to make sure we load the labels and
        # embeddings consistent with the training data.
        # TODO: Need to save the label dictionary from training.
        # And maybe also the vocabulary?
        filepath_train = FLAGS.training_file
        _, _, train_label_dict, train_embeddings = \
            reader.load_data(filepath_train,
                             embeddings=embeddings,
                             max_length=max_sentence_length,
                             label_dict={},
                             train=True)

        test_sentences, test_labels, test_label_dict = \
            reader.load_data(filepath_test,
                             embeddings=train_embeddings,
                             max_length=max_sentence_length,
                             label_dict=train_label_dict,
                             train=False)

        assert test_label_dict == train_label_dict
        vocabulary_size = train_embeddings.vocabulary_size()
        num_labels = len(test_label_dict)

        # Bucket test data.
        LOGGER.info("Creating buckets for test data.")
        factory = BucketFactory()
        test_buckets = factory.build_buckets(np.asarray(test_sentences),
                                              np.asarray(test_labels),
                                              num_buckets=FLAGS.buckets)
        bucket_lengths = [bucket.data.max_length() for bucket in test_buckets]

        # Load model.
        model = create_model(sess,
                             buckets=bucket_lengths,
                             vocabulary_size=vocabulary_size,
                             num_labels=num_labels,
                             is_train=False,
                             embeddings=embeddings)

        # Evaluate on test.
        evaluator = Evaluator()
        start_time_valid = time.time()
        test_loss = 0.0
        test_predictions = []
        test_true = []
        for bucket_id in xrange(len(test_buckets)):
            bucket = test_buckets[bucket_id]
            if bucket == None:
                continue
            # Loss for whole bucket.
            step_loss, predictions, _ = model.batch_update(sess, bucket_id,
                                                           bucket.data,
                                                           True)
            test_predictions.extend(predictions)
            test_true.extend(bucket.data.labels)
            test_loss += np.sum(step_loss)
        time_valid = time.time() - start_time_valid
        test_accuracy = evaluator.accuracy(test_true, test_predictions)
        message = "Test time %fs, loss %f, test acc. %f" % \
                  (time_valid, test_loss/len(test_labels), test_accuracy)
        LOGGER.info(message)
        print message

def main(_):
    '''Main function.'''
    if not FLAGS.interactive:
        training = FLAGS.train
        if training:
            train()
        else:
            test()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    tf.app.run()
