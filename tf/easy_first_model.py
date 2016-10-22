import tensorflow as tf
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

                bucket_losses, bucket_losses_reg, bucket_predictions, \
                    sketches = self.forward(self.inputs[j], self.labels[j],
                                            self.masks[j], max_len, self.seq_lens[j],
                                            self.label_weights)

                self.losses_reg.append(bucket_losses_reg)
                self.losses.append(bucket_losses) # list of tensors, one for each bucket
                self.predictions.append(bucket_predictions)  # list of tensors, one for each bucket
                #self.table = table  # shared for all buckets
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

        self.saver = tf.train.Saver(tf.all_variables(), write_version=tf.train.SaverDef.V2)


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
            output_feed = [self.losses[bucket_id],
                           self.predictions[bucket_id],
                           self.losses_reg[bucket_id]]
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
            from layers import EmbeddingLayer
            embedding_layer = EmbeddingLayer(self.vocabulary_size,
                                             self.embedding_size,
                                             self.keep_prob,
                                             self.embeddings.table,
                                             self.update_embeddings)
            emb = embedding_layer.forward(x)

        with tf.name_scope("hidden"):
            from layers import FeedforwardLayer, RNNLayer
            if self.lstm_size > 0:
                rnn_layer = RNNLayer(sequence_lengths=sequence_lengths,
                                     hidden_size=self.lstm_size,
                                     batch_size=batch_size,
                                     use_bilstm=self.use_bilstm,
                                     keep_prob=self.keep_prob)
                H = rnn_layer.forward(emb)
            else:
                input_size = tf.shape(emb)[2]
                feedforward_layer = FeedforwardLayer(sequence_length=sequence_length,
                                                     input_size=input_size,
                                                     hidden_size=self.hidden_size,
                                                     batch_size=batch_size)
                H = feedforward_layer.forward(emb)
                
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

        return losses, losses_reg, pred_labels, sketches_tf
