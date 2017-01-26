# -*- coding: utf-8 -*-
'''This module implements a neural easy-first model.'''

import tensorflow as tf
import logging
import cPickle as pkl

LOGGER = logging.getLogger("NEF")

class EasyFirstModel(object):
    '''A class for a neural easy-first model.'''
    def __init__(self, num_embedding_features,
                 num_labels, embedding_sizes, hidden_size, hidden_size_2,
                 context_size, vocabulary_sizes, num_sketches, encoder,
                 concatenate_last_layer, batch_size, optimizer, learning_rate,
                 max_gradient_norm, keep_prob=1.0, keep_prob_sketch=1.0,
                 label_weights=None, l2_scale=0.0, l1_scale=0.0,
                 sketch_scale=0.0,
                 discount_factor=0.0, temperature=1.0,
                 embeddings=None, update_embeddings=True, activation="tanh",
                 buckets=None, track_sketches=False, model_dir="models/",
                 is_train=True):
        self.num_embedding_features = num_embedding_features
        self.num_labels = num_labels
        self.embedding_sizes = embedding_sizes
        self.hidden_size = hidden_size
        self.hidden_size_2 = hidden_size_2
        self.context_size = context_size
        self.vocabulary_sizes = vocabulary_sizes
        self.num_sketches = num_sketches
        self.encoder = encoder
        self.concatenate_last_layer = concatenate_last_layer
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
        self.sketch_scale = sketch_scale
        self.discount_factor = discount_factor
        self.temperature = temperature
        self.label_weights = label_weights \
            if label_weights is not None else [1.]*num_labels
        self.keep_prob = keep_prob
        self.keep_prob_sketch = keep_prob_sketch
        self.max_gradient_norm = max_gradient_norm
        self.embeddings = embeddings
        self.update_embeddings = update_embeddings
        self.model_dir = model_dir

        model = 'easy_first'
        self.path = "%s/%s_K%d_D%s_J%d_r%d_batch%d_opt%s_lr%0.4f" \
                    "_gradnorm%0.2f_concat%r_l2r%0.4f_l1r%0.4f_sketchl2r%0.4f" \
                    "_dropout%0.2f" \
                    "_sketchdrop%0.2f_updateemb%s_voc%s.model" % \
                    (self.model_dir, model, self.num_labels,
                     '-'.join([str(d) for d in self.embedding_sizes]),
                     self.hidden_size,
                     self.context_size, self.batch_size, optimizer,
                     self.learning_rate, self.max_gradient_norm,
                     self.concatenate_last_layer,
                     self.l2_scale, self.l1_scale, self.sketch_scale,
                     self.keep_prob,
                     self.keep_prob_sketch, self.update_embeddings,
                     '-'.join([str(v) for v in self.vocabulary_sizes]))
        LOGGER.info("Model path: %s", self.path)
        LOGGER.info("Model with %s encoder", self.encoder)

        if self.update_embeddings:
            LOGGER.info("Updating the embeddings during training")
        else:
            LOGGER.info("Keeping the embeddings fixed")

        if self.concatenate_last_layer:
            LOGGER.info("Concatenating H and S for predictions")

        if self.l2_scale > 0:
            LOGGER.info("L2 regularizer with weight %f", self.l2_scale)

        if self.l1_scale > 0:
            LOGGER.info("L1 regularizer with weight %f", self.l1_scale)

        if self.sketch_scale > 0:
            LOGGER.info("L2 regularizer on sketch with weight %f",
                         self.sketch_scale)

        if not is_train:
            self.keep_prob = 1.
            self.keep_prob_sketch = 1.
        if self.keep_prob < 1.:
            LOGGER.info("Dropout with p=%f", self.keep_prob)
        if self.keep_prob_sketch < 1:
            LOGGER.info("Dropout during sketching with p=%f",
                        self.keep_prob_sketch)

        self.buckets = buckets
        buckets_path = self.path.split(".model", 2)[0] + ".buckets.pkl"
        if self.buckets is not None: # Store bucket lengths.
            LOGGER.info("Dumping bucket edges in %s", buckets_path)
            pkl.dump(self.buckets, open(buckets_path, "wb"))
        else:  # load bucket edges
            LOGGER.info("Loading bucket edges from %s", buckets_path)
            self.buckets = pkl.load(open(buckets_path, "rb"))
        LOGGER.info("Buckets: %s", str(self.buckets))

        if activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == "relu":
            self.activation = tf.nn.relu
        elif activation == "sigmoid":
            self.activation = tf.nn.sigmoid
        else:
            raise NotImplementedError
        LOGGER.info("Activation function %s", self.activation.__name__)

        self.track_sketches = track_sketches
        if self.track_sketches:
            LOGGER.info("Tracking sketches")

        self.inputs = []
        self.labels = []
        self.masks = []
        self.lengths = []
        self.losses = []
        self.losses_reg = []
        self.predictions = []
        self.sketches_tfs = []
        self.keep_probs = []
        self.keep_prob_sketches = []
        self.is_trains = []
        self.gradient_norms = []
        self.updates = []
        self.saver = None
        self._create_computation_graphs(is_train=is_train)

    def _create_computation_graphs(self, is_train=True):
        '''Creates the computation graphs (one per bucket). If is_train=False,
        smaller computation graphs are created which do not compute gradients
        and updates.'''
        # Prepare input feeds.
        self.inputs = []
        self.labels = []
        self.masks = []
        self.lengths = []
        self.losses = []
        self.losses_reg = []
        self.predictions = []
        self.sketches_tfs = []
        self.keep_probs = []
        self.keep_prob_sketches = []
        self.is_trains = []
        num_features = sum(self.num_embedding_features)
        for j, max_length in enumerate(self.buckets):
            self.inputs.append(tf.placeholder(tf.int32,
                                              shape=[None, max_length,
                                                     num_features],
                                              name="inputs{0}".format(j)))
            self.labels.append(tf.placeholder(tf.int32,
                                              shape=[None, max_length],
                                              name="labels{0}".format(j)))
            self.masks.append(tf.placeholder(tf.int64,
                                             shape=[None, max_length],
                                             name="masks{0}".format(j)))
            self.lengths.append(tf.placeholder(tf.int64,
                                               shape=[None],
                                               name="lengths{0}".format(j)))
            self.keep_prob_sketches.append(
                tf.placeholder(tf.float32,
                               name="keep_prob_sketch{0}".format(j)))
            self.keep_probs.append(
                tf.placeholder(tf.float32, name="keep_prob{0}".format(j)))
            self.is_trains.append(
                tf.placeholder(tf.bool, name="is_train{0}".format(j)))

            with tf.variable_scope(tf.get_variable_scope(),
                                   reuse=True if j > 0 else None):
                LOGGER.info("Initializing parameters for bucket with max len "
                            "%d", max_length)

                bucket_losses, bucket_losses_reg, bucket_predictions, \
                    sketches = self.forward(self.inputs[j],
                                            self.labels[j],
                                            self.masks[j],
                                            max_length,
                                            self.lengths[j],
                                            self.label_weights)

                # List of tensors, one for each bucket.
                self.losses_reg.append(bucket_losses_reg)
                self.losses.append(bucket_losses)
                self.predictions.append(bucket_predictions)
                if self.track_sketches:  # else sketches are just empty.
                    self.sketches_tfs.append(sketches)

        # gradients and update operation for training the model
        if is_train:
            params = tf.trainable_variables()
            self.gradient_norms = []
            self.updates = []
            self.gradients = [] # Remove this later. 
            for j in xrange(len(self.buckets)):
                gradients = tf.gradients(tf.reduce_mean(self.losses_reg[j], 0),
                                         params)  # batch normalization
                self.gradients.append(gradients)
                if self.max_gradient_norm > -1:
                    clipped_gradients, norm = \
                        tf.clip_by_global_norm(gradients,
                                               self.max_gradient_norm)
                    self.gradient_norms.append(norm)
                    update = self.optimizer.apply_gradients(
                        zip(clipped_gradients, params))
                    self.updates.append(update)

                else:
                    self.gradient_norms.append(tf.global_norm(gradients))
                    update = self.optimizer.apply_gradients(zip(gradients,
                                                                params))
                    self.updates.append(update)

        self.saver = tf.train.Saver(tf.all_variables())
        # Replace the above with this line, if using the bleeding edge
        # version of TensorFlow.
        #self.saver = tf.train.Saver(tf.all_variables(),
        #                            write_version=tf.train.SaverDef.V2)

    def batch_update(self, session, bucket_id, batch_data, forward_only=False):
        '''Performs a Batch update of the model parameters.
        :param session: the TensorFlow session.
        :param bucket_id: the bucket being processed.
        :param batch_data: the data being processed within that bucket.
        :param forward_only: True if no backward step is required (e.g. test
        time).
        :return: loss, predictions, and regularized loss for this batch.
        '''
        # Get input feed for bucket.
        input_feed = {}
        input_feed[self.inputs[bucket_id].name] = batch_data.inputs
        input_feed[self.labels[bucket_id].name] = batch_data.labels
        input_feed[self.masks[bucket_id].name] = batch_data.masks
        input_feed[self.lengths[bucket_id].name] = batch_data.lengths
        input_feed[self.keep_probs[bucket_id].name] = \
            1 if forward_only else self.keep_prob
        input_feed[self.keep_prob_sketches[bucket_id].name] = \
            1 if forward_only else self.keep_prob_sketch
        input_feed[self.is_trains[bucket_id].name] = not forward_only

        if not forward_only:
            if self.track_sketches:
                # I'm passing the gradients in the output feed since it's often
                # useful to debug. We should remove it later.
                output_feed = [self.losses[bucket_id],
                               self.predictions[bucket_id],
                               self.losses_reg[bucket_id],
                               self.sketches_tfs[bucket_id],
                               self.updates[bucket_id],
                               self.gradient_norms[bucket_id],
                               self.gradients[bucket_id]]
            else:
                output_feed = [self.losses[bucket_id],
                               self.predictions[bucket_id],
                               self.losses_reg[bucket_id],
                               self.updates[bucket_id],
                               self.gradient_norms[bucket_id]]
        else:
            if self.track_sketches:
                output_feed = [self.losses[bucket_id],
                               self.predictions[bucket_id],
                               self.losses_reg[bucket_id],
                               self.sketches_tfs[bucket_id]]
            else:
                output_feed = [self.losses[bucket_id],
                               self.predictions[bucket_id],
                               self.losses_reg[bucket_id]]

        outputs = session.run(output_feed, input_feed)
        predictions = []
        for length, pred in zip(batch_data.lengths, outputs[1]):
            predictions.append(pred[:length].tolist())

        #import pdb
        #import numpy as np
        #if not forward_only:
        #    if np.isnan(outputs[5]):
        #        pdb.set_trace()

        # Outputs are: loss, predictions, regularized loss.
        return outputs[0], predictions, outputs[2], \
               outputs[3] if self.track_sketches else None


    def forward(self, x, y, mask, max_sequence_length, sequence_lengths,
                label_weights):
        '''Computes a forward step for the easy first model and return loss and
        predictions for a batch.
        :param x: the input sequences for a batch.
        :param y: the label sequences for that batch.
        :param mask: a mask telling the sequence lengths.
        :param max_sequence_length: sequence length for the batch.
        :param label_weights: costs for each label (used to compute the loss).
        :return: loss, regularized loss, predictions, and sketches for this
        batch.
        '''
        batch_size = tf.shape(x)[0]
        with tf.name_scope("embedding"):
            from layers import EmbeddingLayer
            j = 0
            embedded_features = []
            for k, embedding in enumerate(self.embeddings):
                num_features = self.num_embedding_features[k]
                embedding_layer = EmbeddingLayer(self.vocabulary_sizes[k],
                                                 self.embedding_sizes[k],
                                                 self.keep_prob,
                                                 k,
                                                 num_features,
                                                 embedding.table,
                                                 self.update_embeddings)
                embedded_features.append(embedding_layer.forward(
                    x[:, :, j:(j+num_features)]))
                j += num_features
            emb = tf.concat(2, embedded_features)

        with tf.name_scope("hidden"):
            from layers import FeedforwardLayer, LSTMLayer, BILSTMLayer
            if self.encoder == 'lstm':
                rnn_layer = LSTMLayer(sequence_lengths=sequence_lengths,
                                      hidden_size=self.hidden_size,
                                      batch_size=batch_size,
                                      keep_prob=self.keep_prob)
                H = rnn_layer.forward(emb)
                state_size = self.hidden_size
            elif self.encoder == 'bilstm':
                rnn_layer = BILSTMLayer(sequence_lengths=sequence_lengths,
                                        hidden_size=self.hidden_size,
                                        batch_size=batch_size,
                                        keep_prob=self.keep_prob)
                H = rnn_layer.forward(emb)
                state_size = 2*self.hidden_size
            else:
                input_size = emb.get_shape().as_list()[2]
                feedforward_layer = \
                    FeedforwardLayer(sequence_length=max_sequence_length,
                                     input_size=input_size,
                                     hidden_size=self.hidden_size,
                                     batch_size=batch_size)
                H = feedforward_layer.forward(emb)
                state_size = self.hidden_size

        with tf.name_scope("sketching"):
            from layers import SketchLayer
            if self.num_sketches < 0:
                num_sketches = max_sequence_length
            else:
                num_sketches = self.num_sketches
            sketch_layer = SketchLayer(num_sketches=num_sketches,
                                       sequence_length=max_sequence_length,
                                       input_size=state_size,
                                       context_size=self.context_size,
                                       hidden_size=self.hidden_size,
                                       hidden_size_2=self.hidden_size_2,
                                       batch_size=batch_size,
                                       batch_mask=mask,
                                       keep_prob=self.keep_prob_sketch,
                                       discount_factor=self.discount_factor,
                                       temperature=self.temperature)
            S, sketches_tf = sketch_layer.forward(H)

        with tf.name_scope("scoring"):
            from layers import ScoreLayer

            if self.concatenate_last_layer:
                state_size = state_size + self.hidden_size
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
                # M word embeddings not included.
                weights_list = [sketch_layer.W_cs, score_layer.W_sp]
                l2_loss = tf.contrib.layers.apply_regularization(
                    tf.contrib.layers.l2_regularizer(self.l2_scale),
                    weights_list=weights_list)
                losses_reg += l2_loss
            if self.l1_scale > 0:
                weights_list = [sketch_layer.W_cs, score_layer.W_sp]
                l1_loss = tf.contrib.layers.apply_regularization(
                    tf.contrib.layers.l1_regularizer(self.l1_scale),
                    weights_list=weights_list)
                losses_reg += l1_loss
            if self.sketch_scale > 0:
                weights_list = [S]
                sketch_loss = tf.contrib.layers.apply_regularization(
                    tf.contrib.layers.l2_regularizer(self.sketch_scale),
                    weights_list=weights_list)
                losses_reg += sketch_loss

        return losses, losses_reg, pred_labels, sketches_tf
