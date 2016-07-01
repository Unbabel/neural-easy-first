# coding=utf-8
import tensorflow as tf
import numpy as np
import time
from sklearn.utils import shuffle


"""
Tensorflow implementation of the neural easy-first model
- Single-State Model
"""

# Flags
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_string("optimizer", "sgd", "Optimizer [sgd, adam, adagrad, adadelta, momentum, "
                                               "rmsprop]")
tf.app.flags.DEFINE_integer("batch_size", 2,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("vocab_size", 10, "Vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_float("max_gradient_norm", 1, "maximum gradient norm for clipping")
tf.app.flags.DEFINE_integer("L", 7, "length of sequences")
tf.app.flags.DEFINE_integer("K", 2, "number of labels")
tf.app.flags.DEFINE_integer("D", 50, "dimensionality of embeddings")
tf.app.flags.DEFINE_integer("N", 5, "number of sketches")
tf.app.flags.DEFINE_integer("J", 100, "dimensionality of hidden layer")
tf.app.flags.DEFINE_integer("r", 2, "context size")
tf.app.flags.DEFINE_boolean("train", True, "training model")
tf.app.flags.DEFINE_integer("epochs", 100, "training epochs")
tf.app.flags.DEFINE_boolean("shuffle", True, "shuffling training data before each epoch")
tf.app.flags.DEFINE_integer("checkpoint_freq", 10, "save model every x epochs")
FLAGS = tf.app.flags.FLAGS

def ef_embedding(inputs, labels, sketches, acc_sketches, indices, vocab_size, K, D, N, J, L, r):
    """
    Single-state easy-first model with embeddings
    :param inputs:
    :param labels:
    :param sketches:
    :param acc_sketches:
    :param indices:
    :param vocab_size:
    :param K:
    :param D:
    :param N:
    :param J:
    :param L:
    :param r:
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

        with tf.variable_scope("ef_model") as scope:
            if not init:
                scope.reuse_variables()
            with tf.name_scope("embedding"):
                M = tf.get_variable(name="M", shape=[vocab_size, D],
                                    initializer=tf.random_uniform_initializer(dtype=tf.float32))

            with tf.name_scope("alpha"):
                w_z = tf.get_variable(name="w_z", shape=[J, 1],
                                      initializer=tf.random_uniform_initializer(dtype=tf.float32))
                V = tf.get_variable(name="V", shape=[L, J],
                                    initializer=tf.random_uniform_initializer(dtype=tf.float32))
                W_hz = tf.get_variable(name="W_hz", shape=[J, D],
                                            initializer=tf.random_uniform_initializer(
                                                dtype=tf.float32))
                W_sz = tf.get_variable(name="W_sz", shape=[J, D*(2*r+1)],
                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))
                W_bz = tf.get_variable(name="W_bz", shape=[J, 2*r+1],
                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))

            with tf.name_scope("beta"):
                W_hs = tf.get_variable(name="W_hs", shape=[D, D],
                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))
                W_ss = tf.get_variable(name="W_ss", shape=[D*(2*r+1), D],
                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))
                w_s = tf.get_variable(name="w_s", shape=[D],
                                      initializer=tf.random_uniform_initializer(dtype=tf.float32))

            with tf.name_scope("prediction"):
                w_p = tf.get_variable(name="w_p", shape=[K],
                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))
                W_sp = tf.get_variable(name="W_sp", shape=[D, K],
                                       initializer=tf.random_uniform_initializer(dtype=tf.float32))

            with tf.name_scope("paddings"):
               # padding_s_col = tf.get_variable(shape=[[0, 0], [r, r]], name="padding_s_col", trainable=False)
               # padding_b = tf.get_variable(shape=[[r, r]], name="padding_b", trainable=False)
                #padding_s_col = tf.get_variable(name="padding_s_col", shape=[[0,0], [r,r]],
                #                       initializer=tf.random_uniform_initializer(dtype=tf.float32), trainable=False)
                #padding_b = tf.get_variable(name="padding_b", shape=[[r,r]],
                 #                      initializer=tf.random_uniform_initializer(dtype=tf.float32), trainable=False)
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
            h_i = tf.slice(H, [i, 0], [1, D], name="h_i")
            v_i = tf.slice(V, [i, 0], [1, J], name="v_i")

            S_padded = tf.pad(S, padding_s_col, "CONSTANT", name="S_padded")
            S_sliced = tf.slice(S_padded, [0, i], [D, 2*r+1])  # slice columns around i
            s_context = tf.reshape(S_sliced, [-1,1], name="s_context")

            b_padded = tf.pad(b, padding_b, "CONSTANT", name="padded")
            b_sliced = tf.slice(b_padded, [i-r+r], [2*r+1], name="b_sliced")
            b_context = tf.reshape(b_sliced, [-1,1], name="b_context")

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
            H = tf.nn.embedding_lookup(M, x, name="H")  # LxD
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
                S_i = tf.slice(final_S, [0, i], [D, 1])  # state vector for this word (column)
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
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(y, depth=K) * tf.log(p),
                                               reduction_indices=[1]))  # avg xent over words
        return cross_entropy, labels_pred, logits_packed

    outputs = []
    losses = []
    logits = []
    init = True
    for x, y, S, b, word_indices in zip(inputs, labels, sketches, acc_sketches, indices):
        loss, output, logit = forward(x, y, S, b, word_indices, init=init)
        outputs.append(output)
        losses.append(loss)
        logits.append(logit)
        init = False
    outputs = tf.pack(outputs)
    losses = tf.pack(losses)
    logits = tf.pack(logits)
    return outputs, losses, logits


class EasyFirstModel():
    """
    Neural easy-first model
    """
    def __init__(self, K, D, N, J, L, r, vocab_size, batch_size, optimizer, learning_rate,
                 max_gradient_norm, forward_only=False):
        """
        Initialize the model
        :param K:
        :param D:
        :param N:
        :param J:
        :param L:
        :param r:
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
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False)
        optimizer_map = {"sgd": tf.train.GradientDescentOptimizer, "adam": tf.train.AdamOptimizer,
                        "adagrad": tf.train.AdagradOptimizer, "adadelta": tf.train.AdadeltaOptimizer,
                        "rmsprop": tf.train.RMSPropOptimizer, "momemtum": tf.train.MomentumOptimizer}
        self.optimizer = optimizer_map.get(optimizer,
                                          tf.train.GradientDescentOptimizer)(self.learning_rate)

        # prepare input feeds
        self.inputs = []
        self.labels = []
        self.sketches = []
        self.acc_sketches = []
        self.indices = []
        for i in xrange(batch_size):  # feed batch_size many xs and ys to graph
            self.inputs.append(tf.placeholder(tf.int32, shape=[self.L], name="input{0}".format(i)))
            self.labels.append(tf.placeholder(tf.int32, shape=[self.L], name="label{0}".format(i)))
            self.sketches.append(tf.placeholder(tf.float32, shape=[self.D, self.L],
                                                name="sketch{0}".format(i)))
            self.acc_sketches.append(tf.placeholder(tf.float32, shape=[self.L],
                                                    name="acc_sketch{0}".format(i)))
            self.indices.append(tf.placeholder(tf.int32, shape=[self.L],
                                               name="indices{0}".format(i)))

        def ef_f(inputs, labels, sketches, acc_sketches, indices):
            return ef_embedding(inputs, labels, sketches, acc_sketches, indices,
                                vocab_size=vocab_size, K=self.K, D=self.D, N=self.N,
                                J=self.J, L=self.L, r=self.r)

        self.outputs, self.losses, self.logits = ef_f(self.inputs, self.labels, self.sketches, self.acc_sketches,
                                         self.indices)

        # gradients and update operation for training the model
        if not forward_only:
            params = tf.trainable_variables()
            #print "updating",  [param.name for param in params]
            gradients = tf.gradients(self.losses, params)
            if max_gradient_norm > -1:
                print "Clipping gradient to %f" % max_gradient_norm
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.gradient_norms = norm
                self.updates = (self.optimizer.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))
            else:
                print "No gradient clipping"
                self.gradient_norms = tf.global_norm(gradients)
                self.updates = (self.optimizer.apply_gradients(
                    zip(gradients, params), global_step=self.global_step))

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
        for l in xrange(self.batch_size):
            # variables dependent on sequence length
            L_i = len(inputs[l])
            S_0 = np.zeros((FLAGS.D, L_i), dtype=float)
            b_0 = np.array([1./L_i]*L_i)
            #print S_0, b_0
            word_indices_i = np.arange(L_i)
            input_feed[self.inputs[l].name] = inputs[l]
            input_feed[self.labels[l].name] = labels[l]
            input_feed[self.sketches[l].name] = S_0
            input_feed[self.acc_sketches[l].name] = b_0
            input_feed[self.indices[l].name] = word_indices_i

        if not forward_only:
            output_feed = [self.updates,  # Update Op that does SGD.
                     self.gradient_norms,  # Gradient norm.
                     self.losses,
                     self.outputs,
                     self.logits]  # Loss for this batch.
        else:
            output_feed = [self.losses, self.outputs]  # loss and predictions for this batch
            #for l in xrange(inputs):  # Output logits.
            #    output_feed.append(self.outputs[l])
        #print "input_feed", input_feed.keys()
        #print "output_feed", output_feed

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], outputs[3], outputs[4] # Gradient norm, loss, predictions
        else:
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
                           FLAGS.max_gradient_norm, forward_only)
    # TODO add loading from checkpoint
    session.run(tf.initialize_all_variables())
    return model


def train():
    """
    Train a model
    :return:
    """
    print "Training"
    checkpoint_path = "checkpoints/"

    with tf.Session() as sess:
        model = create_model(sess, False)

        # TODO read the data
        # dummy data
        no_train_instances = 10
        X_train = np.maximum(np.round(
            np.random.rand(no_train_instances, FLAGS.L)*FLAGS.vocab_size-1, 0), 0)
        Y_train = np.round(np.random.rand(no_train_instances, FLAGS.L), 0)

        no_dev_instances = 4
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

                x_i = X_train[current_sample:current_sample+FLAGS.batch_size]
                y_i = Y_train[current_sample:current_sample+FLAGS.batch_size]

                start_time = time.time()
                _, step_loss, predictions, logits = model.step(sess, x_i, y_i, False)

                step_time += (time.time() - start_time)
                loss += np.sum(step_loss)
                print current_sample, x_i, _, y_i, predictions, step_loss, logits
                train_predictions.append(predictions)
                #print current_sample, predictions

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
                 model.saver.save(sess, checkpoint_path, global_step=model.global_step)

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
    # TODO


def main(_):
    training = FLAGS.train

    if training:
        train()
    else:
        test()


if __name__ == "__main__":
    tf.app.run()