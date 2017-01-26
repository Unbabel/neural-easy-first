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
from dataset import DatasetReader, QualityDatasetReader
from buckets import BucketFactory
from evaluator import Evaluator

# Flags.
tf.app.flags.DEFINE_string("task", "sequence_tagging",
                           "Can be either 'sequence_tagging' (which subsumes "
                           "POS tagging) or 'quality_estimation'.")

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
tf.app.flags.DEFINE_float("sketch_cost", 0, "L2 regularization on sketch")
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
                           "Paths to embeddings (separated by comma).")
tf.app.flags.DEFINE_string("model_dir", "pos_tagging/models", "Path to folder "
                           "where the model is stored.")

tf.app.flags.DEFINE_integer("word_cutoff", 1, "Word cutoff.")
tf.app.flags.DEFINE_integer("max_sentence_length", 50,
                            "Discard sentences longer than this length (both "
                            "for training and dev files. Set this to -1 to "
                            "keep all the sentences.")
tf.app.flags.DEFINE_float("bad_weight", 3.0, "weight for BAD labels (applies "
                          "only to the quality estimation task.")
tf.app.flags.DEFINE_boolean("update_embeddings", True,
                            "True to update the embeddings; False otherwise.")
tf.app.flags.DEFINE_string("activation", "tanh", "Activation function.")
tf.app.flags.DEFINE_string("embedding_sizes", "64",
                           "Dimensionality of embeddings (separated by comma).")
tf.app.flags.DEFINE_integer("hidden_size", 20,
                            "Dimensionality of hidden layers.")
tf.app.flags.DEFINE_integer("hidden_size_2", 20,
                            "Dimensionality of layer for attention scoring.")
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
tf.app.flags.DEFINE_integer("data_limit", "-1", "limit of training "
                           "sentences")

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

def create_model(session, buckets, num_embedding_features, num_labels,
                 vocabulary_sizes, embedding_sizes, is_train=True,
                 embeddings=None, label_weights=None):
    '''Create a new easy-first model.'''
    np.random.seed(123)
    tf.set_random_seed(123)
    model = EasyFirstModel(num_embedding_features=num_embedding_features,
                           num_labels=num_labels,
                           embedding_sizes=embedding_sizes,
                           hidden_size=FLAGS.hidden_size,
                           hidden_size_2=FLAGS.hidden_size_2,
                           context_size=FLAGS.context_size,
                           vocabulary_sizes=vocabulary_sizes,
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
                           sketch_scale=FLAGS.sketch_cost,
                           discount_factor=FLAGS.attention_discount_factor,
                           temperature=FLAGS.attention_temperature,
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

        if FLAGS.task == 'sequence_tagging':
            reader = DatasetReader()
            label_dictionary = None
            label_weights = None
        elif FLAGS.task == 'quality_estimation':
            reader = QualityDatasetReader()
            label_dictionary = {'OK': 0, 'BAD': 1}
            label_weights = [1.0, FLAGS.bad_weight]
            LOGGER.info("Weights for labels: %s", str(label_weights))
        else:
            raise NotImplementedError

        if FLAGS.embeddings == "":
            embeddings = None
        else:
            embeddings = []
            embedding_paths = FLAGS.embeddings.split(',')
            for embedding_path in embedding_paths:
                embeddings.append(reader.load_embeddings(embedding_path))

        max_sentence_length = FLAGS.max_sentence_length
        data_limit = FLAGS.data_limit
        train_sentences, train_labels, train_label_dict, train_embeddings = \
            reader.load_data(filepath_train,
                             embeddings=embeddings,
                             max_length=max_sentence_length,
                             label_dict=label_dictionary,
                             limit=data_limit,
                             train=True)

        # Use training vocab/labels for dev.
        dev_sentences, dev_labels, _ = \
            reader.load_data(filepath_dev,
                             embeddings=train_embeddings,
                             max_length=max_sentence_length,
                             label_dict=train_label_dict,
                             train=False)

        vocabulary_sizes = [embedding.vocabulary_size()
                            for embedding in train_embeddings]
        num_labels = len(train_label_dict)

        if FLAGS.embeddings == "":
            embeddings = []
            for embedding in train_embeddings:
                embeddings.append(Embedding(None,
                                            embedding.word2id,
                                            embedding.id2word,
                                            embedding.unk_id,
                                            embedding.pad_id,
                                            embedding.end_id,
                                            embedding.start_id))

        LOGGER.info("Vocabulary sizes: %s",
                    ' '.join([str(v) for v in vocabulary_sizes]))
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

        # Number of features using each embedding.
        num_embedding_features = reader.num_embedding_features()

        # Embedding sizes.
        embedding_sizes = [int(e) for e in FLAGS.embedding_sizes.split(',')]

        model = create_model(sess, bucket_lengths, num_embedding_features,
                             num_labels,
                             vocabulary_sizes, embedding_sizes,
                             is_train=True, embeddings=embeddings,
                             label_weights=label_weights)

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
                    step_loss, predictions, step_loss_reg, sketches = \
                        model.batch_update(sess, bucket_id, batch_data, False)
                    #import pdb
                    #pdb.set_trace()
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

            time_epoch = time.time() - start_time_epoch
            if FLAGS.task == 'sequence_tagging':
                train_accuracy = evaluator.accuracy(train_true,
                                                    train_predictions)
                LOGGER.info("EPOCH %d: epoch time %fs, loss %f, loss+reg %f, "
                            "train acc. %f",
                            epoch+1, time_epoch, loss/len(train_labels),
                            loss_reg/len(train_labels), train_accuracy)
            elif FLAGS.task == 'quality_estimation':
                train_accuracy = evaluator.accuracy(train_true,
                                                    train_predictions)
                f1_ok, f1_bad = evaluator.f1s_binary(train_true,
                                                     train_predictions)
                LOGGER.info("EPOCH %d: epoch time %fs, loss %f, loss+reg %f, "
                            "train acc. %f, train f1_bad %f, train f1_mult %f",
                            epoch+1, time_epoch, loss/len(train_labels),
                            loss_reg/len(train_labels), train_accuracy,
                            f1_bad, f1_ok*f1_bad)
            else:
                LOGGER.info("Unknown task: %s", FLAGS.task)
                raise NotImplementedError

            # Eval on dev (every epoch).
            start_time_valid = time.time()
            dev_loss = 0.0
            dev_predictions = []
            dev_true = []
            if FLAGS.track_sketches:
                sequence_sketches = [None for _ in xrange(len(dev_sentences))]
                sequence_predictions = [None for _ in xrange(len(dev_sentences))]
            for bucket_id in xrange(len(dev_buckets)):
                bucket = dev_buckets[bucket_id]
                if bucket == None:
                    continue
                # Loss for whole bucket.
                step_loss, predictions, step_loss_reg, sketches = \
                    model.batch_update(sess, bucket_id,
                                       bucket.data,
                                       True)
                # Save the sketches for each sentence in the bucket.
                if FLAGS.track_sketches:
                    for i, sequence_index in enumerate(bucket.indices):
                        sequence_sketches[sequence_index] = sketches[:, i, :, 0]
                        sequence_predictions[sequence_index] = predictions[i]
                dev_predictions.extend(predictions)
                dev_true.extend(bucket.data.labels)
                dev_loss += np.sum(step_loss)
            time_valid = time.time() - start_time_valid
            if FLAGS.task == 'sequence_tagging':
                dev_accuracy = evaluator.accuracy(dev_true, dev_predictions)
                LOGGER.info("EPOCH %d: validation time %fs, loss %f, "
                            "dev acc. %f",
                            epoch+1, time_valid, dev_loss/len(dev_labels),
                            dev_accuracy)
                metric = dev_accuracy
            elif FLAGS.task == 'quality_estimation':
                dev_accuracy = evaluator.accuracy(dev_true, dev_predictions)
                f1_ok, f1_bad = evaluator.f1s_binary(dev_true,
                                                     dev_predictions)
                LOGGER.info("EPOCH %d: validation time %fs, loss %f, "
                            "dev acc. %f, dev f1_bad %f, dev f1_mult %f",
                            epoch+1, time_valid, dev_loss/len(dev_labels),
                            dev_accuracy, f1_bad, f1_ok*f1_bad)
                metric = f1_ok*f1_bad
            else:
                LOGGER.info("Unknown task: %s", FLAGS.task)
                raise NotImplementedError

            # TODO: for QE, don't select best model based accuracy!!!
            if metric > best_valid:
                LOGGER.info("NEW BEST!")
                best_valid = metric
                best_valid_epoch = epoch+1
                # Save checkpoint.
                model.saver.save(sess, model.path,
                                 global_step=model.global_step,
                                 write_meta_graph=False)
                # Write sketch information.
                if FLAGS.track_sketches:
                    LOGGER.info("Writing sketch information.")
                    filepath_sketches = FLAGS.sketch_dir + os.sep + \
                                        'sketches.txt'
                    f = open(filepath_sketches, 'w')
                    labels = ['' for key in train_label_dict]
                    for key, value in train_label_dict.iteritems():
                        labels[value] = key
                        [labels[lid] for lid in dev_labels[0]]

                    #import pdb
                    #pdb.set_trace()
                    for sequence_index in xrange(len(sequence_sketches)):
                        sentence = dev_sentences[sequence_index]
                        sentence_labels = dev_labels[sequence_index]
                        sentence_predictions = \
                            sequence_predictions[sequence_index]
                        words = [train_embeddings[0].id2word[fids[0]] \
                                 for fids in sentence]
                        gold_tags =  [labels[lid] for lid in sentence_labels]
                        predicted_tags = [labels[lid]
                                          for lid in sentence_predictions]
                        f.write('# ' + ' '.join(words) + '\n')
                        f.write('# ' + ' '.join(gold_tags) + '\n')
                        f.write('# ' + ' '.join(predicted_tags) + '\n')
                        sketches = sequence_sketches[sequence_index]
                        for step in xrange(sketches.shape[0]):
                            f.write(
                                'Step %d: %s' % (
                                    step+1,
                                    ' '.join(['{:.3f}'.format(p)
                                              for p in sketches[step, :]]))
                                + '\n')
                        f.write('\n')
                    f.close()
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
        if FLAGS.task == 'sequence_tagging':
            reader = DatasetReader()
            label_dictionary = None
            label_weights = None
        elif FLAGS.task == 'quality_estimation':
            reader = QualityDatasetReader()
            label_dictionary = {'OK': 0, 'BAD': 1}
            label_weights = [1.0, FLAGS.bad_weight]
            LOGGER.info("Weights for labels: %s", str(label_weights))
        else:
            raise NotImplementedError

        embeddings = []
        embedding_paths = FLAGS.embeddings.split(',')
        for embedding_path in embedding_paths:
            embeddings.append(reader.load_embeddings(embedding_path))

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
                             label_dict=label_dictionary,
                             train=True)

        test_sentences, test_labels, test_label_dict = \
            reader.load_data(filepath_test,
                             embeddings=train_embeddings,
                             max_length=max_sentence_length,
                             label_dict=train_label_dict,
                             train=False)

        assert test_label_dict == train_label_dict
        vocabulary_sizes = [embedding.vocabulary_size()
                            for embedding in train_embeddings]
        num_labels = len(test_label_dict)

        # Bucket test data.
        LOGGER.info("Creating buckets for test data.")
        factory = BucketFactory()
        test_buckets = factory.build_buckets(np.asarray(test_sentences),
                                              np.asarray(test_labels),
                                              num_buckets=FLAGS.buckets)
        bucket_lengths = [bucket.data.max_length() for bucket in test_buckets]

        # Number of features using each embedding.
        num_embedding_features = reader.num_embedding_features()

        # Embedding sizes.
        embedding_sizes = [int(e) for e in FLAGS.embedding_sizes.split(',')]

        # Load model.
        model = create_model(sess,
                             buckets=bucket_lengths,
                             num_embedding_features=num_embedding_features,
                             num_labels=num_labels,
                             vocabulary_sizes=vocabulary_sizes,
                             embedding_sizes=embedding_sizes,
                             is_train=False,
                             embeddings=embeddings,
                             label_weights=label_weights)

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
        if FLAGS.task == 'sequence_tagging':
            test_accuracy = evaluator.accuracy(test_true, test_predictions)
            message = "Test time %fs, loss %f, test acc. %f" % \
                      (time_valid, test_loss/len(test_labels), test_accuracy)
            LOGGER.info(message)
            print message
        elif FLAGS.task == 'quality_estimation':
            test_accuracy = evaluator.accuracy(test_true, test_predictions)
            f1_ok, f1_bad = evaluator.f1s_binary(test_true,
                                                 test_predictions)
            message = "Test time %fs, loss %f, test acc. %f, " \
                      "test f1_ok %f, test f1_bad %f" % \
                      (time_valid, test_loss/len(test_labels), test_accuracy,
                       f1_ok, f1_bad)
            LOGGER.info(message)
            print message
        else:
            LOGGER.info("Unknown task: %s", FLAGS.task)

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
