# coding=utf-8
import tensorflow as tf
import numpy as np
import time
from sklearn.utils import shuffle
import sys
from data_generator import *


"""
Tensorflow implementation of the neural easy-first model
- Single-State Model


Tensorflow issues:
- no rmsprop support for sparse gradient updates in version 0.9
- no nested while loops supported in version 0.9
"""

# Flags
tf.app.flags.DEFINE_float("learning_rate", 0.2, "Learning rate.")
tf.app.flags.DEFINE_string("optimizer", "sgd", "Optimizer [sgd, adam, adagrad, adadelta, "
                                                    "momentum]")
tf.app.flags.DEFINE_integer("batch_size", 2,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("vocab_size", 10, "Vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "models/", "Model directory")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_float("max_gradient_norm", -1, "maximum gradient norm for clipping")
tf.app.flags.DEFINE_integer("L", 7, "length of sequences")
tf.app.flags.DEFINE_integer("K", 2, "number of labels")
tf.app.flags.DEFINE_integer("D", 10, "dimensionality of embeddings")
tf.app.flags.DEFINE_integer("N", 7, "number of sketches")
tf.app.flags.DEFINE_integer("J", 100, "dimensionality of hidden layer")
tf.app.flags.DEFINE_integer("r", 1, "context size")
tf.app.flags.DEFINE_boolean("concat", False, "concatenating s_i and h_i for prediction")
tf.app.flags.DEFINE_boolean("train", True, "training model")
tf.app.flags.DEFINE_integer("epochs", 100, "training epochs")
tf.app.flags.DEFINE_boolean("shuffle", False, "shuffling training data before each epoch")
tf.app.flags.DEFINE_integer("checkpoint_freq", 5, "save model every x epochs")
tf.app.flags.DEFINE_integer("lstm_units", 10, "number of LSTM-RNN encoder units")
tf.app.flags.DEFINE_boolean("interactive", False, "interactive mode")
tf.app.flags.DEFINE_boolean("restore", False, "restoring last session from checkpoint")
FLAGS = tf.app.flags.FLAGS


def glorot_init(shape, distribution, type):
    """
    Glorot initialization scheme
    Glorot & Bengio: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    :param shape:
    :param distribution:
    :param type:
    :return:
    """
    fan_in, fan_out = shape
    if distribution=="uniform":
        val = np.sqrt(6./(fan_in+fan_out))
        return tf.random_uniform_initializer(dtype=type, minval=-val, maxval=val)
    else:
        val = np.sqrt(2./(fan_in+fan_out))
        return tf.random_normal_initializer(dtype=type, stddev=val)

def ef_single_state(inputs, labels, vocab_size, K, D, N, J, L, r,
                    lstm_units, concat):
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

    def forward(x, y):
        """
        Compute a forward step for the easy first model and return loss and predictions for a batch
        :param x:
        :param y:
        :return:
        """
        batch_size = tf.shape(x)[0]
        with tf.name_scope("ef_model"):
            with tf.name_scope("embedding"):
                M = tf.get_variable(name="M", shape=[vocab_size, D],
                                    initializer=glorot_init((vocab_size, D),
                                                            "uniform", type=tf.float32))
                emb = tf.nn.embedding_lookup(M, x, name="H")  # batch_size x L x D

            if lstm_units > 0:
                with tf.name_scope("lstm"):
                    # alternatively use LSTMCell supporting peep-holes and output projection
                    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_units, state_is_tuple=True)

                    # see: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py
                    # Permuting batch_size and n_steps
                    remb = tf.transpose(emb, [1, 0, 2])  # L x batch_size x D
                    # Reshaping to (n_steps*batch_size, n_input)
                    remb = tf.reshape(remb, [-1, D])  # L*batch_size x D
                    # Split to get a list of 'n_steps=L' tensors of shape (batch_size, D)
                    remb = tf.split(0, L, remb)

                    rnn_outputs, rnn_states = tf.nn.rnn(cell, remb,
                                    sequence_length=tf.fill([batch_size], L), dtype=tf.float32)
                    # TODO feed actual sentence length
                    # 'outputs' is a list of output at every timestep (until L)
                    rnn_outputs = tf.pack(rnn_outputs)
                    rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])  # batch_size x L x lstm_units
                    H = rnn_outputs
                    state_size = lstm_units
            else:
                H = emb
                state_size = D

            with tf.name_scope("alpha"):
                w_z = tf.get_variable(name="w_z", shape=[J],
                                      initializer=tf.random_uniform_initializer(dtype=tf.float32))
                V = tf.get_variable(name="V", shape=[J, L],
                                    initializer=tf.random_uniform_initializer(dtype=tf.float32))
                W_hz = tf.get_variable(name="W_hz", shape=[state_size, J],
                                       initializer=glorot_init((state_size, J),
                                                               "uniform", type=tf.float32))
                W_sz = tf.get_variable(name="W_sz", shape=[state_size*(2*r+1), J],
                                       initializer=glorot_init((state_size*(2*r+1), J),
                                                               "uniform", type=tf.float32))
                W_bz = tf.get_variable(name="W_bz", shape=[2*r+1, J],
                                       initializer=glorot_init((2*r+1, J),
                                                               "uniform", type=tf.float32))

            with tf.name_scope("beta"):
                W_hs = tf.get_variable(name="W_hs", shape=[state_size, state_size],
                                       initializer=glorot_init((state_size, state_size),
                                                               "uniform", type=tf.float32))
                W_ss = tf.get_variable(name="W_ss", shape=[state_size*(2*r+1), state_size],
                                       initializer=glorot_init((state_size*(2*r+1), state_size),
                                                               "uniform", type=tf.float32))
                w_s = tf.get_variable(name="w_s", shape=[state_size],
                                      initializer=tf.random_uniform_initializer(dtype=tf.float32))

            with tf.name_scope("prediction"):
                w_p = tf.get_variable(name="w_p", shape=[K],
                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))

                wsp_size = state_size
                if concat:  # h_i and s_i are concatenated before prediction
                    wsp_size = state_size*2
                W_sp = tf.get_variable(name="W_sp", shape=[wsp_size, K],
                                       initializer=glorot_init((wsp_size, K),
                                                               "uniform", type=tf.float32))

            with tf.name_scope("paddings"):
                padding_s_col = tf.constant([[0, 0], [0, 0], [r, r]], name="padding_s_col")
                padding_b = tf.constant([[0, 0], [r, r]], name="padding_b")


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
            for i in np.arange(r,L+r):
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
            _1 = tf.matmul(h_avg, W_hs)
            _2 = tf.matmul(s_avg, W_ss)
            s_n = tf.nn.tanh(_1 + _2 + w_s)  # batch_size x state_size
            S_update = tf.batch_matmul(tf.expand_dims(s_n, [2]), tf.expand_dims(a_n, [1]))
            S_n = S + S_update
            return n+1, b_n, S_n

        with tf.variable_scope("sketching"):
            n = tf.constant(1, dtype=tf.int32, name="n")
            S = tf.zeros(shape=[batch_size, state_size, L], dtype=tf.float32)
            b = tf.zeros(shape=[batch_size, L], dtype=tf.float32)  # TODO init uniformly

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
                l = tf.matmul(tf.reshape(s_i, [batch_size, state_size]), W_sp) + w_p
                return l  # batch_size x K

        with tf.variable_scope("scoring"):

            logits = []
            ps = []
            pred_labels = []
            losses = []
            for i in np.arange(L):  # compute score, probs and losses per word for whole batch
                word_label_score = score(i)
                logits.append(word_label_score)
                word_label_probs = tf.nn.softmax(word_label_score)
                ps.append(word_label_probs)
                word_preds = tf.argmax(word_label_probs, 1)
                pred_labels.append(word_preds)
                y_words = tf.reshape(tf.slice(y, [0,i], [batch_size, 1]), [batch_size])
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(word_label_score,
                                                                               y_words)
                losses.append(cross_entropy)
            pred_labels = tf.transpose(tf.pack(pred_labels), [1, 0])  # batch_size x L
            losses = tf.reduce_mean(tf.transpose(tf.pack(losses), [1, 0]), 1)  # batch_size x 1

        return losses, pred_labels


    print inputs, labels
    losses, predictions = forward(inputs, labels)
    print losses, predictions
    return losses, predictions


class EasyFirstModel():
    """
    Neural easy-first model
    """
    def __init__(self, K, D, N, J, L, r, vocab_size, batch_size, optimizer, learning_rate,
                 max_gradient_norm, lstm_units, concat, forward_only=False):
        """
        Initialize the model
        :param K:
        :param D:
        :param N:
        :param J:
        :param L:
        :param r:
        :param lstm_units:
        :param vocab_size:
        :param batch_size:
        :param optimizer:
        :param learning_rate:
        :param max_gradient_norm:
        :param forward_only:
        :return:
        """
        self.K = K
        self.D = D
        self.N = N
        self.J = J
        self.L = L
        self.r = r
        self.lstm_units = lstm_units
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.concat = concat
        self.global_step = tf.Variable(0, trainable=False)
        optimizer_map = {"sgd": tf.train.GradientDescentOptimizer, "adam": tf.train.AdamOptimizer,
                        "adagrad": tf.train.AdagradOptimizer, "adadelta": tf.train.AdadeltaOptimizer,
                        "rmsprop": tf.train.RMSPropOptimizer, "momemtum": tf.train.MomentumOptimizer}
        self.optimizer = optimizer_map.get(optimizer,
                                           tf.train.GradientDescentOptimizer)(self.learning_rate)

        if self.lstm_units > 0:
            self.state_size = self.lstm_units
            print "Model with LSTM RNN encoder of %d units and embeddings of size %d" % \
                  (self.lstm_units, self.D)
        else:
            self.state_size = D
            print "Model with simple embeddings of size %d" % self.D

        if self.N > 0:
            print "Model with %d sketches" % self.N
        else:
            print "No sketches"

        if self.concat or self.N == 0:
            print "Concatenating H and S for predictions"

        # feed whole batch
        self.inputs = tf.placeholder(tf.int32, shape=[None, self.L], name="input")
        self.labels = tf.placeholder(tf.int32, shape=[None, self.L], name="labels")

        def ef_f(inputs, labels):

            return ef_single_state(inputs, labels,
                                   vocab_size=vocab_size, K=self.K, D=self.D, N=self.N,
                                   J=self.J, L=self.L, r=self.r, lstm_units=lstm_units,
                                   concat=self.concat)

        self.losses, self.predictions = ef_f(self.inputs, self.labels)

        # gradients and update operation for training the model
        if not forward_only:
            params = tf.trainable_variables()
            gradients = tf.gradients(self.losses, params)
            if max_gradient_norm > -1:
                print "Clipping gradient to %f" % max_gradient_norm
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.gradient_norms = norm
                self.updates = (self.optimizer.apply_gradients(zip(clipped_gradients, params)))
            else:
                print "No gradient clipping"
                self.gradient_norms = tf.global_norm(gradients)
                self.updates = (self.optimizer.apply_gradients(zip(gradients, params)))

        self.saver = tf.train.Saver(tf.all_variables())

    def batch_update(self, session, inputs, labels, forward_only=False):
        """
        Training step
        :param session:
        :param inputs:
        :param labels:
        :param forward_only:
        :return:
        """
        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.inputs.name] = inputs  # list
        input_feed[self.labels.name] = labels  # list

        if not forward_only:
            output_feed = [self.updates,
                     self.gradient_norms,
                     self.losses,
                     self.predictions]
        else:
            output_feed = [self.losses, self.predictions]

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], outputs[3]  # grad norm, loss, predictions
        else:
            return outputs[0], outputs[1]  # loss, predictions


def create_model(session, forward_only=False):
    """
    Create a model
    :param session:
    :param forward_only:
    :return:
    """
    model = EasyFirstModel(K=FLAGS.K, D=FLAGS.D, N=FLAGS.N, J=FLAGS.J, L=FLAGS.L, r=FLAGS.r,
                           vocab_size=FLAGS.vocab_size, batch_size=FLAGS.batch_size,
                           optimizer=FLAGS.optimizer, learning_rate=FLAGS.learning_rate,
                           max_gradient_norm=FLAGS.max_gradient_norm, lstm_units=FLAGS.lstm_units,
                           concat=FLAGS.concat, forward_only=forward_only)
    checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
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
    print "Training"

    with tf.Session() as sess:
        model = create_model(sess, False)

        # TODO read the data
        # dummy data
        no_train_instances = 13
        no_dev_instances = 5
        print "Training on %d instances" % no_train_instances
        print "Validating on %d instances" % no_dev_instances

        xs, ys = random_interdependent_data([no_train_instances, no_dev_instances], FLAGS.L,
                                            FLAGS.vocab_size, FLAGS.K)
        X_train, Y_train, X_dev, Y_dev = xs[0], ys[0], xs[1], ys[1]

        # Training loop
        for epoch in xrange(FLAGS.epochs):
            current_sample = 0
            step_time, loss = 0.0, 0.0
            train_predictions = []

            if FLAGS.shuffle:
                X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

            while current_sample < len(X_train):

                x_i = X_train[current_sample:current_sample+FLAGS.batch_size]
                y_i = Y_train[current_sample:current_sample+FLAGS.batch_size]

                start_time = time.time()
                _, step_loss, predictions = model.batch_update(sess, x_i, y_i, False)

                step_time += (time.time() - start_time)
                loss += np.sum(step_loss)
                #print current_sample, x_i, _, y_i, predictions, step_loss, embeddings
                # TODO regularizer
                train_predictions.append(predictions[0])

                current_sample += FLAGS.batch_size

            # eval on dev
            eval_sample = 0
            dev_predictions = []
            while eval_sample < len(X_dev):
                x_i = X_dev[eval_sample:eval_sample+FLAGS.batch_size]
                y_i = Y_dev[eval_sample:eval_sample+FLAGS.batch_size]

                start_time = time.time()
                step_loss, predictions = model.batch_update(sess, x_i, y_i, True)
                step_time += (time.time() - start_time)
                loss += np.sum(step_loss)
                dev_predictions.append(predictions[0])

                eval_sample += FLAGS.batch_size

            train_accuracy = accuracy(Y_train, train_predictions)
            eval_acurracy = accuracy(Y_dev, dev_predictions)

            print "EPOCH %d: avg step time %fs, avg loss %f, train accuracy %f, dev accuracy %f" % \
                (epoch+1, step_time/len(X_train), loss/len(X_train),
                 train_accuracy, eval_acurracy)

            if epoch % FLAGS.checkpoint_freq == 0:
                 model.saver.save(sess, FLAGS.train_dir, global_step=model.global_step)

def accuracy(y_i, predictions):
    """
    Accuracy of word predictions
    :param y_i:
    :param predictions:
    :return:
    """
    correct_words, all = 0.0, 0.0
    for y, y_pred in zip(y_i, predictions):
        for y_w, y_pred_w in zip(y, y_pred):  # words
            all += 1
            if y_pred_w == y_w:
                correct_words += 1
    return correct_words/all


def test():
    """
    Test a model
    :return:
    """
    print "Testing"
    FLAGS.restore = True  # has to be loaded
    with tf.Session() as sess:
        # load model
        model = create_model(sess, True)

        # TODO use real data
        no_test_instances = 40
        print "Testing on %d instances" % no_test_instances

        # TODO doesn't make sense for testing, since distributions differ from training
        xs, ys = random_interdependent_data([no_test_instances], FLAGS.L, FLAGS.vocab_size,
                                            FLAGS.K)
        X_test, Y_test = xs[0], ys[0]

        # eval
        eval_sample = 0
        test_predictions = []
        loss = 0
        while eval_sample < len(X_test):
            x_i = X_test[eval_sample:eval_sample+FLAGS.batch_size]
            y_i = Y_test[eval_sample:eval_sample+FLAGS.batch_size]

            step_loss, predictions = model.batch_update(sess, x_i, y_i, True)
            loss += np.sum(step_loss)
            test_predictions.append(predictions[0])

            eval_sample += FLAGS.batch_size

        test_accuracy = accuracy(Y_test, test_predictions)

        print "Test avg loss %f, accuracy %f" % (loss/len(X_test), test_accuracy)


def demo():
    """
    Test a model dynamically by reading input from stdin
    :return:
    """
    FLAGS.restore = True
    with tf.Session() as sess:
        # load model
        model = create_model(sess, True)
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            inputs = sentence.split()
            if len(inputs) > model.L:
                print "Input too long. Only sequences of length %d allowed." % model.L
                break
            elif len(inputs) < model.L:
                print "Input too short. Only sequences of length %d allowed." % model.L
                break
            # TODO from words to vectors (word2id mapping or feature extraction)
            x = [float(char) for char in inputs]
            # filter OOV,  TODO ensure that there is an UNK symbol
            x = [x_i if x_i < FLAGS.vocab_size else 0 for x_i in x]
            y = [0 for char in inputs]  # dummy labels

            step_loss, predictions = model.batch_update(sess, [x], [y], True)
            outputs = predictions[0]
            print "prediction: ", outputs
            sys.stdout.flush()
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
