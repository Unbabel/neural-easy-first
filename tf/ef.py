# coding=utf-8
import tensorflow as tf
import numpy as np
import time
from sklearn.utils import shuffle
import sys


"""
Tensorflow implementation of the neural easy-first model
- Single-State Model
"""

# Flags
tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_string("optimizer", "sgd", "Optimizer [sgd, adam, adagrad, adadelta, momentum, "
                                               "rmsprop]")
tf.app.flags.DEFINE_integer("batch_size", 1,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("vocab_size", 10, "Vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "models/", "Model directory")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_float("max_gradient_norm", -1, "maximum gradient norm for clipping")
tf.app.flags.DEFINE_integer("L", 7, "length of sequences")
tf.app.flags.DEFINE_integer("K", 2, "number of labels")
tf.app.flags.DEFINE_integer("D", 11, "dimensionality of embeddings")
tf.app.flags.DEFINE_integer("N", 5, "number of sketches")
tf.app.flags.DEFINE_integer("J", 100, "dimensionality of hidden layer")
tf.app.flags.DEFINE_integer("r", 2, "context size")
tf.app.flags.DEFINE_boolean("train", False, "training model")
tf.app.flags.DEFINE_integer("epochs", 100, "training epochs")
tf.app.flags.DEFINE_boolean("shuffle", True, "shuffling training data before each epoch")
tf.app.flags.DEFINE_integer("checkpoint_freq", 10, "save model every x epochs")
tf.app.flags.DEFINE_boolean("lstm_units", 0, "number of LSTM-RNN encoder units")
tf.app.flags.DEFINE_boolean("interactive", False, "interactive mode")
FLAGS = tf.app.flags.FLAGS


def ef_single_state(inputs, labels, vocab_size, K, D, N, J, L, r,
                    lstm_units):
    """
    Single-state easy-first model with embeddings and optional LSTM-RNN encoder
    :param inputs:
    :param labels:
    :param K:
    :param D:
    :param N:
    :param J:
    :param L:
    :param r:
    :param lstm_units:
    :return:
    """

    def forward(x, y, S, b, word_indices, init=False):
        """
        Compute a forward step for the easy first model and return loss and predictions
        :param x:
        :param y:
        :param S:
        :param b:
        :param word_indices:
        :param init:
        :return:
        """

        print "x",x
        print "y", y


        with tf.variable_scope("ef_model") as scope:
            if not init:
                scope.reuse_variables()
                print "reusing"
            else:
                print "new"

            #print scope.reuse

            with tf.name_scope("embedding"):
                M = tf.get_variable(name="M", shape=[vocab_size, D],
                                    initializer=tf.random_uniform_initializer(dtype=tf.float32,
                                    minval=-1, maxval=1.0))
                emb = tf.nn.embedding_lookup(M, x, name="H")  # LxD
                print "emb", emb

            if lstm_units > 0:
                with tf.name_scope("lstm"):
                    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_units, state_is_tuple=True)
                    rnn_outputs, rnn_states = tf.nn.rnn(cell, [emb],
                                    sequence_length=tf.fill([tf.shape(x)[0]], L), dtype=tf.float32)
                    H = rnn_states.c
                    state_size = lstm_units
            else:
                H = emb
                state_size = D

            with tf.name_scope("alpha"):
                w_z = tf.get_variable(name="w_z", shape=[J, 1],
                                      initializer=tf.random_uniform_initializer(dtype=tf.float32))
                V = tf.get_variable(name="V", shape=[L, J],
                                    initializer=tf.random_uniform_initializer(dtype=tf.float32))
                W_hz = tf.get_variable(name="W_hz", shape=[J, state_size],
                                            initializer=tf.random_uniform_initializer(
                                                dtype=tf.float32))
                W_sz = tf.get_variable(name="W_sz", shape=[J, state_size*(2*r+1)],
                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))
                W_bz = tf.get_variable(name="W_bz", shape=[J, 2*r+1],
                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))

            with tf.name_scope("beta"):
                W_hs = tf.get_variable(name="W_hs", shape=[state_size, state_size],
                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))
                W_ss = tf.get_variable(name="W_ss", shape=[state_size*(2*r+1), state_size],
                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))
                w_s = tf.get_variable(name="w_s", shape=[state_size],
                                      initializer=tf.random_uniform_initializer(dtype=tf.float32))

            with tf.name_scope("prediction"):
                w_p = tf.get_variable(name="w_p", shape=[K],
                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))
                W_sp = tf.get_variable(name="W_sp", shape=[state_size, K],
                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))

            with tf.name_scope("paddings"):
                padding_s_col = tf.constant([[0, 0], [r, r]], name="padding_s_col")
                padding_b = tf.constant([[r, r]], name="padding_b")

            with tf.name_scope("looping"):
                def fn(i):
                    ra = tf.range((i-r)+r, (i+r)+r+1, name="brange")
                    return ra
                all_indices = tf.map_fn(fn, word_indices, name="mapb")

        def z_i(i):
            """
            Compute attention weight
            :param i:
            :return:
            """
            h_i = tf.slice(H, [i, 0], [1, state_size], name="h_i")
            v_i = tf.slice(V, [i, 0], [1, J], name="v_i")

            S_padded = tf.pad(S, padding_s_col, "CONSTANT", name="S_padded")
            S_sliced = tf.slice(S_padded, [0, i], [state_size, 2*r+1])  # slice columns around i
            s_context = tf.reshape(S_sliced, [-1, 1], name="s_context")

            b_padded = tf.pad(b, padding_b, "CONSTANT", name="padded")
            b_sliced = tf.slice(b_padded, [i-r+r], [2*r+1], name="b_sliced")
            b_context = tf.reshape(b_sliced, [-1, 1], name="b_context")

            _1 = tf.matmul(W_sz, s_context)
            _2 = tf.matmul(W_bz, b_context)
            _3 = tf.matmul(W_hz, h_i, transpose_b=True)
            tanh = tf.tanh(_1 + _2 + _3 + w_z)
            _4 = tf.matmul(v_i, tanh, transpose_b=False)
            z_i = tf.reshape(_4, [1])
            return z_i

        def alpha():
            """
            Compute attention weight for all words in sequence
            :return:
            """
            # implementation with loop

            # if L is fixed
            z = []
            for i in np.arange(L):
                z.append(z_i(i))
            z_packed = tf.pack(z)
            rz = tf.reshape(z_packed, [1, L])
            a_n = tf.nn.softmax(rz)

            # variable L, WARNING: only works if nested while loops are possible
            #z = tf.map_fn(self.z_i, self.word_indices, dtype=tf.float32, name="map_z")
            #a_n = tf.nn.softmax(z)

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
            gathered_S = tf.gather(tf.transpose(S_padded), tf.transpose(all_indices),
                                   name="gathered_S")
            conv_result_S = tf.transpose(tf.reshape(gathered_S, [tf.size(b), -1],
                                                    name="reshape_S"))
            return conv_result_S  # shape: LxD*(2*r+1)

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
            a_n = alpha()  # 1xL
            b_n = tf.add(tf.mul(tf.sub(tf.cast(n, tf.float32), -1.), b), a_n)
            b_n = tf.div(b_n, tf.cast(n, tf.float32))
            h_avg = tf.matmul(a_n, H)  # 1xD
            conv = conv_r(S)
            s_avg = tf.matmul(a_n, conv, transpose_b=True)
            _1 = tf.matmul(h_avg, W_hs)
            _2 = tf.matmul(s_avg, W_ss)
            s_n = tf.nn.tanh(_1 + _2 + w_s)
            S_n = S + tf.matmul(s_n, a_n, transpose_a=True, transpose_b=False)
            return n+1, b_n, S_n

        with tf.variable_scope("sketching"):
            n = tf.constant(1, dtype=tf.int32, name="n")
            (final_n, final_b, final_S) = tf.while_loop(
                cond=lambda n, _1, _2: n <= N,
                body=sketch_step,
                loop_vars=(n, b, S)
            )

            def score(i):
                """
                Score the word at index i
                """
                S_i = tf.slice(final_S, [0, i], [state_size, 1])  # state vector for this word (column)
                l = tf.matmul(S_i, W_sp, transpose_a=True) + w_p
                shaped = tf.reshape(l, [K])
                return shaped

        with tf.variable_scope("scoring"):
            # if L is not fixed, WARNING: only works if nested whiles are allowed
            #logits = tf.map_fn(score, self.word_indices, dtype=tf.float32, name="map_logits")
            #p = tf.nn.softmax(logits)

            #if L is fixed
            logits = []
            for i in np.arange(L):
                logits.append(score(i))
            logits_packed = tf.pack(logits)
            p = tf.nn.softmax(logits_packed)
            labels_pred = tf.argmax(p, 1)

        with tf.variable_scope("xent"):
            cross_entropy = -tf.reduce_sum(tf.one_hot(y, depth=K) * tf.log(p),
                                           reduction_indices=[1]) # xent over words
        return cross_entropy, labels_pred


    print inputs, labels
    state_size = lstm_units if lstm_units > 0 else D

    def single_step(i):
        input = tf.squeeze(tf.slice(inputs, [i, 0], size=[1, L]))
        label = tf.squeeze(tf.slice(labels, [i, 0], size=[1, L]))
        sketch = tf.zeros(shape=[state_size, L], dtype=tf.float32)
        acc_sketch = tf.zeros(shape=[L], dtype=tf.float32)
        indices = tf.range(L)
        init = tf.equal(i, 0)
        def forward_init():
            return forward(input, label, sketch, acc_sketch, indices, init=True)

        def forward_reuse():
            return forward(input, label, sketch, acc_sketch, indices, init=False)

        cross_entropy, labels_pred = tf.cond(init, forward_init, forward_reuse)
        print cross_entropy, labels_pred

        # labels needs cast because packing after map
        return cross_entropy, tf.cast(labels_pred, tf.float32)

    iterator = tf.range(tf.shape(inputs))  # forward pass for all inputs in batch
    losses_and_predictions = tf.map_fn(single_step, iterator, dtype=tf.float32)
    print "Losses and pred", losses_and_predictions
    return losses_and_predictions


class EasyFirstModel():
    """
    Neural easy-first model
    """
    def __init__(self, K, D, N, J, L, r, vocab_size, batch_size, optimizer, learning_rate,
                 max_gradient_norm, lstm_units, forward_only=False):
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

        # feed whole batch
        self.inputs = tf.placeholder(tf.int32, shape=[None, self.L], name="input")
        self.labels = tf.placeholder(tf.int32, shape=[None, self.L], name="labels")

        def ef_f(inputs, labels):

            return ef_single_state(inputs, labels,
                                    vocab_size=vocab_size, K=self.K, D=self.D, N=self.N,
                                    J=self.J, L=self.L, r=self.r, lstm_units=lstm_units)

        self.output = ef_f(self.inputs, self.labels)  # output contains losses and labels for all instances in batch -> need to be extracted separately
        print "losses and labels", self.output
        batch_size = tf.shape(self.inputs)[0]
        print "batch size", batch_size
        instances_iterator = tf.range(batch_size)
        def get_loss(i):
            return tf.squeeze(tf.slice(self.output, [2*i, 0, 0], [1, 1, L]))
        def get_predictions(i):
            return tf.squeeze(tf.slice(self.output, [2*i+1, 0, 0], [1, 1, L]))
        self.losses = tf.reduce_mean(tf.map_fn(get_loss, instances_iterator, dtype=tf.float32), 1)  # mean loss per sentence
        print "losses", self.losses
        self.predictions = tf.map_fn(get_predictions, instances_iterator, name="predictions", dtype=tf.float32)
        print "predictions", self.predictions
        # gradients and update operation for training the model

        if not forward_only:
            params = tf.trainable_variables()
            print "updating",  [param.name for param in params]
            gradients = tf.gradients(self.losses, params)
            print "gradients", gradients
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

    def step(self, session, inputs, labels, forward_only=False):
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
        #for l in xrange(self.batch_size):
            # variables dependent on sequence length
            #L_i = len(inputs[l])
            #S_0 = np.zeros((self.state_size, L_i), dtype=float)
            #b_0 = np.array([1./L_i]*L_i)
            #print S_0, b_0
            #word_indices_i = np.arange(L_i)
            #input_feed[self.inputs[l].name] = inputs[l]
            #input_feed[self.labels[l].name] = labels[l]
            #input_feed[self.sketches[l].name] = S_0
            #input_feed[self.acc_sketches[l].name] = b_0
            #input_feed[self.indices[l].name] = word_indices_i

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
            return outputs[0], outputs[1]  # loss, predictions


def create_model(session, forward_only=False):
    """
    Create a model
    :param session:
    :param forward_only:
    :return:
    """
    model = EasyFirstModel(FLAGS.K, FLAGS.D, FLAGS.N, FLAGS.J, FLAGS.L, FLAGS.r, FLAGS.vocab_size,
                           FLAGS.batch_size, FLAGS.optimizer, FLAGS.learning_rate,
                           FLAGS.max_gradient_norm, FLAGS.lstm_units, forward_only)
    checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path):
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
        no_train_instances = 40
        X_train = np.maximum(np.round(
            np.random.rand(no_train_instances, FLAGS.L)*FLAGS.vocab_size-1, 0), 0)
        Y_train = np.round(np.random.rand(no_train_instances, FLAGS.L), 0)

        no_dev_instances = 10
        X_dev = np.maximum(np.round(
            np.random.rand(no_dev_instances, FLAGS.L)*FLAGS.vocab_size-1, 0), 0)
        Y_dev = np.round(np.random.rand(no_dev_instances, FLAGS.L), 0)

        # Training loop
        for epoch in xrange(FLAGS.epochs):
            current_sample = 0
            step_time, loss = 0.0, 0.0
            train_predictions = []

            if FLAGS.shuffle:
                X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

            while current_sample < len(X_train):

                x_i = X_train[current_sample:current_sample+FLAGS.batch_size]  # TODO what if train_size % batch_size !=0
                y_i = Y_train[current_sample:current_sample+FLAGS.batch_size]

                start_time = time.time()
                _, step_loss, predictions, logits, final_sketches, embeddings = model.step(sess, x_i, y_i, False)

                step_time += (time.time() - start_time)
                loss += np.sum(step_loss)
                #print current_sample, x_i, _, y_i, predictions, step_loss, embeddings
                # TODO regularizer
                train_predictions.append(predictions)

                current_sample += FLAGS.batch_size

            # TODO make batch size flexible

            # eval on dev
            eval_sample = 0
            dev_predictions = []
            while eval_sample < len(X_dev):
                x_i = X_dev[eval_sample:eval_sample+FLAGS.batch_size]
                y_i = Y_dev[eval_sample:eval_sample+FLAGS.batch_size]

                start_time = time.time()
                step_loss, predictions = model.step(sess, x_i, y_i, True)
                step_time += (time.time() - start_time)
                loss += np.sum(step_loss)
                dev_predictions.append(predictions)

                eval_sample += FLAGS.batch_size

            train_accuracy = accuracy(Y_train, train_predictions[0])
            eval_acurracy = accuracy(Y_dev, dev_predictions[0])

            print "EPOCH %d: avg step time %fs, avg loss %f, train accuracy %f, dev accuracy %f" % \
                (epoch+1, step_time/len(X_train), loss/len(X_train),
                 train_accuracy, eval_acurracy)

            if epoch % FLAGS.checkpoint_freq == 0:
                 model.saver.save(sess, FLAGS.train_dir, global_step=model.global_step)

def accuracy(y_i, predictions):
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
    with tf.Session() as sess:
        # load model
        model = create_model(sess, True)

        # TODO read data
        no_test_instances = 40
        X_test = np.maximum(np.round(
            np.random.rand(no_test_instances, FLAGS.L)*FLAGS.vocab_size-1, 0), 0)
        Y_test = np.round(np.random.rand(no_test_instances, FLAGS.L), 0)

        # eval
        eval_sample = 0
        test_predictions = []
        loss = 0
        while eval_sample < len(X_test):
            x_i = X_test[eval_sample:eval_sample+FLAGS.batch_size]  # TODO should be without batch
            y_i = Y_test[eval_sample:eval_sample+FLAGS.batch_size]

            step_loss, predictions = model.step(sess, x_i, y_i, True)
            loss += np.sum(step_loss)
            test_predictions.append(predictions)

            eval_sample += FLAGS.batch_size

        test_accuracy = accuracy(Y_test, test_predictions[0])

        print "Test avg loss %f, accuracy %f" % (loss/len(X_test), test_accuracy)


def demo():
    """
    Test a model dynamically by reading input from stdin
    :return:
    """
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
            # flter OOV
            x = [x_i if x_i < FLAGS.vocab_size else 0 for x_i in x]  # TODO ensure that there is an UNK symbol
            y = [0 for char in inputs]  # dummy labels

            step_loss, predictions = model.step(sess, [x], [y], True)
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