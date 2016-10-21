import tensorflow as tf
import tensorflow.python.util.nest as nest
import numpy as np
import logging
import cPickle as pkl
import pdb

logger = logging.getLogger("NEF")

class EasyFirstModel(object):
    """
    Neural easy-first model
    """
    def __init__(self, num_labels, embedding_size, hidden_size, context_size,
                 vocabulary_size, lstm_size, concatenate_last_layer, use_bilstm,
                 batch_size, optimizer, learning_rate, max_gradient_norm,
                 keep_prob=1.0, keep_prob_sketch=1.0, label_weights=None,
                 l2_scale=0.0, l1_scale=0.0, embeddings=None,
                 update_embeddings=True,
                 activation="tanh", buckets=None, track_sketches=False,
                 model_dir="models/", is_train=True):
        """
        Initialize the model.
        :param xxx:
        """
        self.num_labels = num_labels
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.vocabulary_size = vocabulary_size
        self.lstm_size = lstm_size
        self.concatenate_last_layer = concatenate_last_layer
        self.use_bilstm = use_bilstm

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False)
        optimizer_map = {"sgd": tf.train.GradientDescentOptimizer,
                         "adam": tf.train.AdamOptimizer,
                        "adagrad": tf.train.AdagradOptimizer,
                         "adadelta": tf.train.AdadeltaOptimizer,
                        "rmsprop": tf.train.RMSPropOptimizer,
                         "momemtum": tf.train.MomentumOptimizer}
        self.optimizer = optimizer_map. \
            get(optimizer,
                tf.train.GradientDescentOptimizer)(self.learning_rate)
        self.l2_scale = l2_scale
        self.l1_scale = l1_scale
        self.label_weights = label_weights \
            if label_weights is not None else [1.]*num_labels
        self.keep_prob = keep_prob
        self.keep_prob_sketch = keep_prob_sketch
        self.max_gradient_norm = max_gradient_norm

        self.embeddings = embeddings
        self.update_embeddings = update_embeddings

        self.model_dir = model_dir

        model = 'easy_first'
        self.path = "%s/%s_K%d_D%d_J%d_r%d_batch%d_opt%s_lr%0.4f_gradnorm%0.2f" \
                    "_lstm%d_concat%r_l2r%0.4f_l1r%0.4f_dropout%0.2f_sketchdrop%0.2f_updateemb%s_voc%d.model" % \
                    (self.model_dir, model, self.num_labels,
                     self.embedding_size, self.hidden_size,
                     self.context_size, self.batch_size, optimizer,
                     self.learning_rate, self.max_gradient_norm, self.lstm_size,
                     self.concatenate_last_layer,
                     self.l2_scale, self.l1_scale, self.keep_prob,
                     self.keep_prob_sketch, self.update_embeddings,
                     self.vocabulary_size)
        logger.info("Model path: %s"  % self.path)

        if self.lstm_size > 0:
            if self.use_bilstm:
                logger.info(
                    "Model with bi-directional LSTM RNN encoder of %d units" %
                    self.lstm_size)
            else:
                logger.info(
                    "Model with uni-directional LSTM RNN encoder of %d units" %
                    self.lstm_size)
        else:
            if self.embeddings.table is None:
                logger.info("Model with simple embeddings of size %d" %
                            self.embedding_size)
            else:
                logger.info("Model with simple embeddings of size %d" %
                            self.embeddings.table.shape[0])

        if update_embeddings:
            logger.info("Updating the embeddings during training")
        else:
            logger.info("Keeping the embeddings fixed")

        if self.concatenate_last_layer:
            logger.info("Concatenating H and S for predictions")

        if self.l2_scale > 0:
            logger.info("L2 regularizer with weight %f" % self.l2_scale)

        if self.l1_scale > 0:
            logger.info("L1 regularizer with weight %f" % self.l1_scale)

        if not is_train:
            self.keep_prob = 1.
            self.keep_prob_sketch = 1.
        if self.keep_prob < 1.:
            logger.info("Dropout with p=%f" % self.keep_prob)
        if self.keep_prob_sketch < 1:
            logger.info("Dropout during sketching with p=%f" %
                        self.keep_prob_sketch)

        self.buckets = buckets
        buckets_path = self.path.split(".model", 2)[0] + ".buckets.pkl"
        if self.buckets is not None:  # store bucket edges
            logger.info("Dumping bucket edges in %s" % buckets_path)
            pkl.dump(self.buckets, open(buckets_path, "wb"))
        else:  # load bucket edges
            logger.info("Loading bucket edges from %s" % buckets_path)
            self.buckets = pkl.load(open(buckets_path, "rb"))
        logger.info("Buckets: %s" % str(self.buckets))

        if activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == "relu":
            self.activation = tf.nn.relu
        elif activation == "sigmoid":
            self.activation = tf.nn.sigmoid
        else:
            raise NotImplementedError
        logger.info("Activation function %s" % self.activation.__name__)

        self.track_sketches = track_sketches
        if self.track_sketches:
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
        window_size = 1
        for j, max_len in enumerate(self.buckets):
            self.inputs.append(tf.placeholder(tf.int32,
                                              shape=[None, max_len, window_size],
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

                bucket_losses, bucket_losses_reg, bucket_predictions, table, \
                    sketches = self.forward(self.inputs[j], self.labels[j],
                                            self.masks[j], max_len, self.seq_lens[j],
                                            self.label_weights)

                self.losses_reg.append(bucket_losses_reg)
                self.losses.append(bucket_losses) # list of tensors, one for each bucket
                self.predictions.append(bucket_predictions)  # list of tensors, one for each bucket
                self.table = table  # shared for all buckets
                if self.track_sketches:  # else sketches are just empty
                    self.sketches_tfs.append(sketches)

        # gradients and update operation for training the model
        if is_train:
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


    def _score(self, hs_j, state_size, W_sp, w_p):
        """
        Score the word at index j, returns state vector for this word (column) across batch
        """
        if self.concatenate_last_layer:
            l = tf.matmul(tf.reshape(hs_j, [self.batch_size, 2*state_size]), W_sp) + w_p
        else:
            l = tf.matmul(tf.reshape(hs_j, [self.batch_size, state_size]), W_sp) + w_p
        return l  # batch_size x K


    def _score_predict_loss(self, score_input, state_size, W_sp, w_p):
        """
        Predict a label for an input, compute the loss and return label and loss
        """
        [hs_i, y_words] = score_input
        word_label_score = self._score(hs_i, state_size, W_sp, w_p)
        word_label_probs = tf.nn.softmax(word_label_score)
        word_preds = tf.argmax(word_label_probs, 1)
        y_words_full = tf.one_hot(tf.squeeze(y_words), depth=self.num_labels, on_value=1.0, off_value=0.0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(word_label_score,
                                                                                y_words_full)
        if self.label_weights is not None:
            label_weights = tf.reduce_sum(tf.mul(y_words_full, self.label_weights), 1)
            cross_entropy = tf.mul(cross_entropy, label_weights)
        return [word_preds, cross_entropy]


    def _softmax_with_mask(self, tensor, mask, tau=1.0):
        """
        compute the softmax including the mask
        the mask is multiplied with exp(x), before the normalization
        :param tensor: 2D
        :param mask: 2D, same shape as tensor
        :param tau: temperature, the cooler the more spiked is distribution
        :return:
        """
        row_max = tf.expand_dims(tf.reduce_max(tensor, 1), 1)
        t_shifted = tensor - row_max
        nom = tf.exp(t_shifted/tau)*tf.cast(mask, tf.float32)
        row_sum = tf.expand_dims(tf.reduce_sum(nom, 1), 1)
        softmax = nom / row_sum
        return softmax


    def _conv_r(self, max_sequence_length, state_size, padded_matrix, r):
        """
        Extract r context columns around each column and concatenate
        :param padded_matrix: batch_size x L+(2*r) x 2*state_size
        :param r: context size
        :return:
        """
        # gather indices of padded
        time_major_matrix = tf.transpose(padded_matrix, [1, 2, 0])  # time-major  -> L x 2*state_size x batch_size
        contexts = []
        for j in np.arange(r, max_sequence_length+r):
            # extract 2r+1 rows around i for each batch
            context_j = time_major_matrix[j-r:j+r+1, :, :]  # 2*r+1 x 2*state_size x batch_size
            # concatenate
            pdb.set_trace()
            context_j = tf.reshape(context_j, [(2*r+1)*2*state_size, self.batch_size])  # (2*r+1)*(state_size) x batch_size
            contexts.append(context_j)
        contexts = tf.pack(contexts)  # L x (2*r+1)* 2*(state_size) x batch_size
        batch_major_contexts = tf.transpose(contexts, [2, 0, 1]) # switch back: batch_size x L x (2*r+1)*2(state_size) (batch-major)
        return batch_major_contexts


    def _sketch_step(self, n_counter, max_sequence_length, state_size,
                     W_hsz, w_z, v,
                     W_hss, W_hss_mask,
                     w_s,
                     mask, sketch_embedding_matrix,
                     padding_hs_col, b):
        """
        Compute the sketch vector and update the sketch according to attention over words
        :param sketch_embedding_matrix: updated sketch, batch_size x L x 2*state_size (concatenation of H and S)
        :return:
        """
        attention_discount_factor = 0.
        attention_temperature = 1.

        # Add column on right and left.
        sketch_embedding_matrix_padded = tf.pad(sketch_embedding_matrix,
                                                padding_hs_col,
                                                "CONSTANT",
                                                name="HS_padded")

        # beta function
        a_n, _ = self._compute_attention(max_sequence_length,
                                         state_size,
                                         W_hsz, w_z, v,
                                         mask,
                                         sketch_embedding_matrix_padded,
                                         b,
                                         discount_factor=attention_discount_factor,
                                         temperature=attention_temperature)

        # cumulative attention scores
        #b_n = b + a_n
        b_n = (tf.cast(n_counter, tf.float32)-1)*b + a_n #rz
        b_n /= tf.cast(n_counter, tf.float32)

        # Andre: wasn't conv computed already??
        conv = self._conv_r(max_sequence_length, state_size,
                            sketch_embedding_matrix_padded,
                            self.context_size)  # batch_size x L x 2*state_size*(2*r+1)
        hs_avg = tf.batch_matmul(tf.expand_dims(a_n, [1]), conv)  # batch_size x 1 x 2*state_size*(2*r+1)
        hs_avg = tf.reshape(hs_avg,
                            [self.batch_size,
                             2*state_size*(2*self.context_size+1)])

        # same dropout for all steps (http://arxiv.org/pdf/1512.05287v3.pdf), mask is ones if no dropout
        a = tf.matmul(hs_avg, tf.mul(W_hss, W_hss_mask))
        hs_n = self.activation(a + w_s)  # batch_size x state_size

        sketch_update = tf.batch_matmul(tf.expand_dims(a_n, [2]), tf.expand_dims(hs_n, [1]))  # batch_size x L x state_size
        embedding_update = tf.zeros(shape=[self.batch_size, max_sequence_length,
                                           state_size], dtype=tf.float32)  # batch_size x L x state_size
        sketch_embedding_matrix += tf.concat(2, [embedding_update, sketch_update])
        return n_counter+1, sketch_embedding_matrix, b_n, a_n


    def _compute_attention(self, sequence_len, state_size, W_hsz, w_z, v,
                           mask, padded_matrix, b,
                           discount_factor=0.0, temperature=1.0):
        """
        Compute attention weight for all words in sequence in batch
        :return:
        """
        z = []
        for j in np.arange(sequence_len):
            matrix_sliced = tf.slice(padded_matrix,
                                     [0, j, 0],
                                     [self.batch_size, 2*self.context_size+1, 2*state_size])
            matrix_context = tf.reshape(matrix_sliced,
                                        [self.batch_size,
                                         2*state_size*(2*self.context_size+1)],
                                        name="s_context")  # batch_size x 2*state_size*(2*r+1)
            activ = self.activation(tf.matmul(matrix_context, W_hsz) + w_z)
            z_j = tf.matmul(activ, v)
            z.append(z_j)

        z_packed = tf.pack(z)  # seq_len, batch_size, 1
        rz = tf.transpose(z_packed, [1, 0, 2])  # batch-major
        rz = tf.reshape(rz, [self.batch_size, sequence_len])
        # subtract cumulative attention
        d = discount_factor # 5.0  # discount factor
        tau = temperature # 0.2 # temperature.
        a_n = rz
        #a_n = a_n - d*b
        #a_n = softmax_with_mask(a_n, mask, tau=tau)
        a_n = self._softmax_with_mask(a_n, mask, tau=1.0)
        return a_n, rz


    def forward(self, x, y, mask, max_sequence_length, sequence_lengths,
                label_weights):
        """
        Compute a forward step for the easy first model and return loss and predictions for a batch
        :param x:
        :param y:
        :return:
        """
        batch_size = tf.shape(x)[0]
        with tf.name_scope("embedding"):
            if self.embeddings.table is None:
                M = tf.get_variable(name="M",
                                    shape=[self.vocabulary_size,
                                           self.embedding_size],
                                    initializer=\
                                        tf.contrib.layers.xavier_initializer( \
                                            uniform=True, dtype=tf.float32))
            else:
                M = tf.get_variable(name="M",
                                    shape=[self.embeddings.table.shape[0],
                                           self.embeddings.table.shape[1]],
                                    initializer=\
                                        tf.constant_initializer( \
                                            self.embeddings.table),
                                    trainable=self.update_embeddings)
                assert len(self.embeddings.table[0] == self.embedding_size)

            # Dropout on embeddings.
            M = tf.nn.dropout(M, self.keep_prob)  # TODO make param

            emb_orig = tf.nn.embedding_lookup(M, x, name="emb_orig")  # batch_size x L x window_size x emb_size
            window_size = 1
            emb = tf.reshape(emb_orig,
                             [batch_size, max_sequence_length,
                              window_size * self.embedding_size],
                             name="emb") # batch_size x L x window_size*emb_size

        with tf.name_scope("hidden"):
            if self.lstm_size > 0:
                fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_size,
                                                  state_is_tuple=True)
                if self.use_bilstm:
                    with tf.name_scope("bi-lstm"):
                        bw_cell = tf.nn.rnn_cell.LSTMCell( \
                            num_units=self.lstm_units, state_is_tuple=True)

                        # dropout on lstm
                        fw_cell = \
                            tf.nn.rnn_cell.DropoutWrapper( \
                                fw_cell, input_keep_prob=1.0, \
                                output_keep_prob=keep_prob) # TODO make params, input is already dropped out
                        bw_cell = \
                            tf.nn.rnn_cell.DropoutWrapper( \
                                bw_cell, input_keep_prob=1.0, \
                                output_keep_prob=keep_prob)

                        outputs, _ = \
                            tf.nn.bidirectional_dynamic_rnn(
                                fw_cell, bw_cell, emb,
                                sequence_length=sequence_lengths,
                                dtype=tf.float32, time_major=False)
                        outputs = tf.concat(2, outputs)
                        state_size = 2*lstm_size # concat of fw and bw lstm output
                else:
                    with tf.name_scope("lstm"):

                        # dropout on lstm
                        fw_cell = tf.nn.rnn_cell.DropoutWrapper( \
                            fw_cell, input_keep_prob=1.0, \
                            output_keep_prob=self.keep_prob) # TODO make params, input is already dropped out

                        outputs, _, = tf.nn.dynamic_rnn(
                            cell=fw_cell, inputs=emb,
                            sequence_length=sequence_lengths,
                            dtype=tf.float32, time_major=False)
                        state_size = self.lstm_size

                H = outputs

            else:
                # fully-connected layer on top of embeddings to reduce size
                remb = tf.reshape(emb, [batch_size*L, window_size*emb_size])
                W_fc = tf.get_variable(name="W_fc",
                                       shape=[window_size*embedding_size,
                                              self.hidden_size],  # TODO another param?
                                       initializer=\
                                           tf.contrib.layers.xavier_initializer(
                                               uniform=True, dtype=tf.float32))
                b_fc = tf.get_variable(shape=[hidden_size],
                                       initializer=\
                                           tf.random_uniform_initializer(
                                               dtype=tf.float32), name="b_fc")
                H = tf.reshape(activation(tf.matmul(remb, W_fc)+b_fc), \
                               [batch_size, maximum_sequence_length,
                                hidden_size])
                state_size = hidden_size


        with tf.name_scope("sketching"):
            from layers import SketchLayer
            sketch_layer = SketchLayer(sequence_length=max_sequence_length,
                                       input_size=self.hidden_size,
                                       context_size=self.context_size,
                                       hidden_size=self.hidden_size,
                                       batch_size=batch_size,
                                       batch_mask=mask,
                                       keep_prob=self.keep_prob_sketch)
            S, sketches_tf = sketch_layer.forward(H)

            if False:
                W_hss = tf.get_variable(name="W_hss",
                                        shape=[2*state_size*(2*self.context_size+1),
                                               state_size],
                                        initializer=\
                                        tf.contrib.layers.xavier_initializer(
                                            uniform=True, dtype=tf.float32))
                w_s = tf.get_variable(name="w_s", shape=[state_size],
                                      initializer=\
                                      tf.random_uniform_initializer(
                                          dtype=tf.float32))
                w_z = tf.get_variable(name="w_z", shape=[self.hidden_size],
                                      initializer=tf.random_uniform_initializer(
                                          dtype=tf.float32))
                v = tf.get_variable(name="v", shape=[self.hidden_size, 1],
                                    initializer=tf.random_uniform_initializer(
                                        dtype=tf.float32))
                W_hsz = tf.get_variable(name="W_hsz",
                                        shape=[2*state_size*(2*self.context_size+1),
                                               self.hidden_size],
                                        initializer=\
                                        tf.contrib.layers.xavier_initializer(
                                            uniform=True, dtype=tf.float32))
                # dropout within sketch
                # see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_ops.py#L1078 (inverted dropout)
                W_hss_mask = \
                             tf.to_float(tf.less_equal(tf.random_uniform(tf.shape(W_hss)),
                                                       self.keep_prob_sketch)) * \
                             tf.inv(self.keep_prob_sketch)

                S = tf.zeros(shape=[batch_size, max_sequence_length, state_size], dtype=tf.float32)
                HS = tf.concat(2, [H, S])
                sketches = []
                # Cumulative attention.
                b = tf.ones(shape=[batch_size, max_sequence_length],
                            dtype=tf.float32) / max_sequence_length
                #b = tf.zeros(shape=[batch_size, L], dtype=tf.float32)
                b_n = b

                padding_hs_col = \
                                 tf.constant([[0, 0],
                                              [self.context_size, self.context_size],
                             [0, 0]], name="padding_hs_col")
                n = tf.constant(1, dtype=tf.int32, name="n")

                num_sketches = max_sequence_length
                if num_sketches > 0:
                    for i in xrange(num_sketches):
                        n, HS, b_n, a_n = self._sketch_step(n,
                                                            max_sequence_length,
                                                            state_size,
                                                            W_hsz, w_z, v,
                                                            W_hss, W_hss_mask,
                                                            w_s,
                                                            mask,
                                                            HS,
                                                            padding_hs_col,
                                                            b_n)
                        sketch = tf.split(2, 2, HS)[1]
                        # append attention to sketch
                        #sketch_attention_cumulative = tf.concat(2, [sketch, tf.expand_dims(a_n, 2), tf.expand_dims(b_n, 2)])
                        sketch_attention_cumulative = tf.concat(2, [tf.expand_dims(a_n, 2), tf.expand_dims(b_n, 2)])
                        sketches.append(sketch_attention_cumulative)

                sketches_tf = tf.pack(sketches)



        with tf.name_scope("scoring"):
            from layers import ScoreLayer

            if self.concatenate_last_layer:
                state_size = 2*self.hidden_size
                S = tf.concat(2, [H, S])
            else:
                state_size = self.hidden_size

            score_layer = ScoreLayer(sequence_length=max_sequence_length,
                                     input_size=state_size,
                                     num_labels=self.num_labels,
                                     label_weights=label_weights,
                                     batch_size=batch_size,
                                     batch_mask=mask)

            losses, pred_labels = score_layer.forward(S, y)
            losses_reg = losses
            if self.l2_scale > 0:
                weights_list = [W_hss, W_sp]  # M_src, M_tgt word embeddings not included
                l2_loss = tf.contrib.layers.apply_regularization(
                    tf.contrib.layers.l2_regularizer(self.l2_scale), weights_list=weights_list)
                losses_reg += l2_loss
            if self.l1_scale > 0:
                weights_list = [W_hss, W_sp]
                l1_loss = tf.contrib.layers.apply_regularization(
                    tf.contrib.layers.l1_regularizer(l1_scale), weights_list=weights_list)
                losses_reg += l1_loss


            if False:
                w_p = tf.get_variable(name="w_p", shape=[self.num_labels],
                                      initializer=tf.random_uniform_initializer(dtype=tf.float32))
                if self.concatenate_last_layer:
                    wsp_size = 2*state_size
                else:
                    wsp_size = state_size
                W_sp = tf.get_variable(name="W_sp", shape=[wsp_size, self.num_labels],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))

                if label_weights is not None:
                    label_weights = tf.constant(self.label_weights, name="label_weights")

                #if self.concatenate_last_layer:
                    #S = HS
                #else:
                    #S = tf.slice(HS, [0, 0, state_size], [batch_size, L, state_size])
                if self.concatenate_last_layer:
                    S = tf.concat(2, [H, S])

                f = lambda score_input : self._score_predict_loss(score_input, state_size, W_sp, w_p)
                scores_pred = tf.map_fn(f,
                                        [tf.transpose(S, [1, 0, 2]), tf.transpose(y, [1,0])],
                                        dtype=[tf.int64, tf.float32])  # elems are unpacked along dim 0 -> L
                pred_labels = scores_pred[0]
                losses = scores_pred[1]

                pred_labels = mask*tf.transpose(pred_labels, [1, 0])  # masked, batch_size x L
                losses = tf.reduce_mean(tf.cast(mask, tf.float32)*tf.transpose(losses, [1, 0]),
                                        1)  # masked, batch_size x 1
                losses_reg = losses
                if self.l2_scale > 0:
                    weights_list = [W_hss, W_sp]  # M_src, M_tgt word embeddings not included
                    l2_loss = tf.contrib.layers.apply_regularization(
                        tf.contrib.layers.l2_regularizer(self.l2_scale), weights_list=weights_list)
                    losses_reg += l2_loss
                if self.l1_scale > 0:
                    weights_list = [W_hss, W_sp]
                    l1_loss = tf.contrib.layers.apply_regularization(
                        tf.contrib.layers.l1_regularizer(l1_scale), weights_list=weights_list)
                    losses_reg += l1_loss

        return losses, losses_reg, pred_labels, M, sketches_tf
