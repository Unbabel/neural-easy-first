import tensorflow as tf
import tensorflow.python.util.nest as nest
import numpy as np

def ef_single_state(inputs, labels, masks, seq_lens, src_vocab_size, tgt_vocab_size, K, D, N, J, L, r,
                    lstm_units, concat, window_size, keep_prob, keep_prob_sketch,
                    l2_scale, l1_scale,
                    attention_discount_factor=0.0,
                    attention_temperature=1.0,
                    src_embeddings=None, tgt_embeddings=None,
                    class_weights=None, bilstm=True, activation=tf.nn.tanh,
                    update_emb=True, track_sketches=False, is_train=False):
    """
    Single-state easy-first model with embeddings and optional LSTM-RNN encoder
    :param inputs:
    :param labels:
    :param src_vocab_size:
    :param tgt_vocab_size:
    :param K:
    :param D:
    :param N:
    :param J:
    :param L:
    :param r:
    :param lstm_units:
    :return:
    """

    def forward(x, y, mask, seq_lens, class_weights=None):
        """
        Compute a forward step for the easy first model and return loss and predictions for a batch
        :param x:
        :param y:
        :return:
        """
        batch_size = tf.shape(x)[0]
        with tf.name_scope("embedding"):
            if src_embeddings.table is None:
                #print "Random src embeddings of dimensionality %d" % D
                M_src = tf.get_variable(name="M_src", shape=[src_vocab_size, D],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
                emb_size = D
            else:
                M_src = tf.get_variable(name="M_src",
                                shape=[src_embeddings.table.shape[0],
                                       src_embeddings.table.shape[1]],
                                initializer=tf.constant_initializer(src_embeddings.table),
                                trainable=update_emb)
                D_loaded = len(tgt_embeddings.table[0])
                #print "Loading existing src embeddings of dimensionality %d" % D_loaded
                emb_size = D_loaded


            if tgt_embeddings.table is None:
                #print "Random tgt embeddings of dimensionality %d" % D
                M_tgt = tf.get_variable(name="M_tgt", shape=[tgt_vocab_size, D],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
                emb_size += D
            else:
                M_tgt = tf.get_variable(name="M_tgt",
                    shape=[tgt_embeddings.table.shape[0],
                           tgt_embeddings.table.shape[1]],
                    initializer=tf.constant_initializer(tgt_embeddings.table),
                    trainable=update_emb)
                D_loaded = len(tgt_embeddings.table[0])
                #print "Loading existing tgt embeddings of dimensionality %d" % D_loaded
                emb_size += D_loaded

            # dropout on embeddings
            M_tgt = tf.nn.dropout(M_tgt, keep_prob)  # TODO make param
            M_src = tf.nn.dropout(M_src, keep_prob)

            #print "embedding size", emb_size
            x_tgt, x_src = tf.split(2, 2, x)  # split src and tgt part of input
            emb_tgt = tf.nn.embedding_lookup(M_tgt, x_tgt, name="emg_tgt")  # batch_size x L x window_size x emb_size
            emb_src = tf.nn.embedding_lookup(M_src, x_src, name="emb_src")  # batch_size x L x window_size x emb_size
            emb_comb = tf.concat(2, [emb_src, emb_tgt], name="emb_comb") # batch_size x L x 2*window_size x emb_size
            emb = tf.reshape(emb_comb, [batch_size, L, window_size*emb_size],
                             name="emb") # batch_size x L x window_size*emb_size

        with tf.name_scope("hidden"):
            if lstm_units > 0:

                fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, state_is_tuple=True)

                if bilstm:
                    with tf.name_scope("bi-lstm"):
                        bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, state_is_tuple=True)

                        # dropout on lstm
                        fw_cell = \
                            tf.nn.rnn_cell.DropoutWrapper(
                                fw_cell, input_keep_prob=1, output_keep_prob=keep_prob)  # TODO make params, input is already dropped out
                        bw_cell = \
                            tf.nn.rnn_cell.DropoutWrapper(
                                bw_cell, input_keep_prob=1, output_keep_prob=keep_prob)

                        outputs, _ = \
                            tf.nn.bidirectional_dynamic_rnn(
                                fw_cell, bw_cell, emb, sequence_length=seq_lens,
                                dtype=tf.float32, time_major=False)
                        outputs = tf.concat(2, outputs)
                        state_size = 2*lstm_units  # concat of fw and bw lstm output
                else:
                    with tf.name_scope("lstm"):

                        # dropout on lstm
                        fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                            fw_cell, input_keep_prob=1, output_keep_prob=keep_prob)  # TODO make params, input is already dropped out

                        outputs, _, = tf.nn.dynamic_rnn(
                            cell=fw_cell, inputs=emb, sequence_length=seq_lens,
                            dtype=tf.float32, time_major=False)
                        state_size = lstm_units

                H = outputs

            else:
                # fully-connected layer on top of embeddings to reduce size
                remb = tf.reshape(emb, [batch_size*L, window_size*emb_size])
                W_fc = tf.get_variable(name="W_fc", shape=[window_size*emb_size, J],  # TODO another param?
                              initializer=tf.contrib.layers.xavier_initializer(
                                  uniform=True, dtype=tf.float32))
                b_fc = tf.get_variable(shape=[J], initializer=tf.random_uniform_initializer(
                    dtype=tf.float32), name="b_fc")
                H = tf.reshape(activation(tf.matmul(remb, W_fc)+b_fc), [batch_size, L, J])
                state_size = J

        with tf.name_scope("sketching"):
            W_hss = tf.get_variable(name="W_hss", shape=[2*state_size*(2*r+1), state_size],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
            w_s = tf.get_variable(name="w_s", shape=[state_size],
                                      initializer=tf.random_uniform_initializer(dtype=tf.float32))
            w_z = tf.get_variable(name="w_z", shape=[J],
                                      initializer=tf.random_uniform_initializer(dtype=tf.float32))
            v = tf.get_variable(name="v", shape=[J, 1],
                                initializer=tf.random_uniform_initializer(dtype=tf.float32))
            W_hsz = tf.get_variable(name="W_hsz", shape=[2*state_size*(2*r+1), J],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
            # dropout within sketch
            # see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_ops.py#L1078 (inverted dropout)
            W_hss_mask = tf.to_float(tf.less_equal(tf.random_uniform(tf.shape(W_hss)), keep_prob_sketch)) * tf.inv(keep_prob_sketch)

            def softmax_to_hard(tensor):
                max_att = tf.reduce_max(tensor, 1)
                a_n = tf.cast(tf.equal(tf.expand_dims(max_att, 1), tensor), tf.float32)
                return a_n

            def normalize(tensor):
                """
                turn a tensor into a probability distribution
                :param tensor: 2D tensor
                :return:
                """
                z = tf.reduce_sum(tensor, 1)
                t = tensor / tf.expand_dims(z, 1)
                return t

            def softmax_with_mask(tensor, mask, tau=1.0):
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

            def z_j(j, padded_matrix):
                """
                Compute attention weight
                :param j:
                :return:
                """
                matrix_sliced = tf.slice(padded_matrix, [0, j, 0], [batch_size, 2*r+1, 2*state_size])
                matrix_context = tf.reshape(matrix_sliced, [batch_size, 2*state_size*(2*r+1)], name="s_context")  # batch_size x 2*state_size*(2*r+1)
                activ = activation(tf.matmul(matrix_context, W_hsz) + w_z)
                z_i = tf.matmul(activ, v)
                return z_i

            #def alpha(sequence_len, padded_matrix, b_i, a_previous, discount_factor=0.0, temperature=1.0):       
            def alpha(sequence_len, padded_matrix, discount_factor=0.0, temperature=1.0):
                """
                Compute attention weight for all words in sequence in batch
                :return:
                """
                z = []
                for j in np.arange(sequence_len):
                    z.append(z_j(j, padded_matrix))
                z_packed = tf.pack(z)  # seq_len, batch_size, 1
                rz = tf.transpose(z_packed, [1, 0, 2])  # batch-major
                rz = tf.reshape(rz, [batch_size, sequence_len])
                # subtract cumulative attention
                #a_n = softmax_with_mask(rz, mask, tau=1.0)  # make sure that no attention is spent on padded areas
                a_i = rz
                #a_i = a_i - discount_factor*b_i
                a_i = softmax_with_mask(a_i, mask, tau=temperature)
                # interpolation gate
                #a_i = softmax_with_mask(rz, mask, tau=1.0)
                #g_n = tf.sigmoid(g)  # range (0,1)
                #a_i = tf.mul(g_n, a_previous) + tf.mul((1-g_n), a_i)
                # normalization
                #a_i = normalize(a_i)
                return a_i #, rz

            def conv_r(padded_matrix, r):
                """
                Extract r context columns around each column and concatenate
                :param padded_matrix: batch_size x L+(2*r) x 2*state_size
                :param r: context size
                :return:
                """
                # gather indices of padded
                time_major_matrix = tf.transpose(padded_matrix, [1, 2, 0])  # time-major  -> L x 2*state_size x batch_size
                contexts = []
                for j in np.arange(r, L+r):
                    # extract 2r+1 rows around i for each batch
                    context_j = time_major_matrix[j-r:j+r+1, :, :]  # 2*r+1 x 2*state_size x batch_size
                    # concatenate
                    context_j = tf.reshape(context_j, [(2*r+1)*2*state_size, batch_size])  # (2*r+1)*(state_size) x batch_size
                    contexts.append(context_j)
                contexts = tf.pack(contexts)  # L x (2*r+1)* 2*(state_size) x batch_size
                batch_major_contexts = tf.transpose(contexts, [2, 0, 1]) # switch back: batch_size x L x (2*r+1)*2(state_size) (batch-major)
                return batch_major_contexts

            #def sketch_step(n_counter, sketch_embedding_matrix, a, b):
            def sketch_step(n_counter, sketch_embedding_matrix, a):
                """
                Compute the sketch vector and update the sketch according to attention over words
                :param sketch_embedding_matrix: updated sketch, batch_size x L x 2*state_size (concatenation of H and S)
                :return:
                """
                sketch_embedding_matrix_padded = tf.pad(sketch_embedding_matrix, padding_hs_col, "CONSTANT", name="HS_padded")  # add column on right and left

                # beta function
                a_j = alpha(L, sketch_embedding_matrix_padded, #b, a,  # TODO a_j, _ = alpha(...)
                               discount_factor=attention_discount_factor,
                               temperature=attention_temperature)

                # make "hard"
                #a_j = softmax_to_hard(a_j)

                # cumulative attention scores
                #b_j = (tf.cast(n_counter, tf.float32)-1)*b + a_j #rz
                #b_j /= tf.cast(n_counter, tf.float32)
		#b_j = b + a_j
                #b_j = b

                conv = conv_r(sketch_embedding_matrix_padded, r)  # batch_size x L x 2*state_size*(2*r+1)
                hs_avg = tf.batch_matmul(tf.expand_dims(a_j, [1]), conv)  # batch_size x 1 x 2*state_size*(2*r+1)
                hs_avg = tf.reshape(hs_avg, [batch_size, 2*state_size*(2*r+1)])

                # same dropout for all steps (http://arxiv.org/pdf/1512.05287v3.pdf), mask is ones if no dropout
                ac = tf.matmul(hs_avg, tf.mul(W_hss, W_hss_mask))
                hs_n = activation(ac + w_s)  # batch_size x state_size

                sketch_update = tf.batch_matmul(tf.expand_dims(a_j, [2]), tf.expand_dims(hs_n, [1]))  # batch_size x L x state_size
                embedding_update = tf.zeros(shape=[batch_size, L, state_size], dtype=tf.float32)  # batch_size x L x state_size
                sketch_embedding_matrix += tf.concat(2, [embedding_update, sketch_update])
                return n_counter+1, sketch_embedding_matrix, a_j #, b_j

            S = tf.zeros(shape=[batch_size, L, state_size], dtype=tf.float32)
            a_n = tf.zeros(shape=[batch_size, L])
            HS = tf.concat(2, [H, S])
            sketches = []
            #b = tf.ones(shape=[batch_size, L], dtype=tf.float32)/L  # cumulative attention
            #b_n = tf.zeros(shape=[batch_size, L], dtype=tf.float32)  # cumulative attention
            #g = tf.Variable(tf.zeros(shape=[L]))

            padding_hs_col = tf.constant([[0, 0], [r, r], [0, 0]], name="padding_hs_col")
            n = tf.constant(1, dtype=tf.int32, name="n")

            if track_sketches:  # use for loop (slower, because more memory)
                if N > 0:
                    for i in xrange(N):
             #           n, HS, a_n, b_n = sketch_step(n, HS, a_n, b_n)
                        n, HS, a_n = sketch_step(n, HS, a_n)
                        sketch = tf.split(2, 2, HS)[1]
                        # append attention to sketch
                        sketch_attention = tf.concat(2, [sketch, tf.expand_dims(a_n, 2)])
                        #sketch_attention_cumulative = tf.concat(2, [sketch, tf.expand_dims(a_n, 2), tf.expand_dims(b_n, 2)])
                        #sketch_attention_cumulative = tf.concat(2, [tf.expand_dims(a_n, 2), tf.expand_dims(b_n, 2)])
                        #sketches.append(sketch_attention_cumulative)
                        sketches.append(sketch_attention)
            else:  # use while loop
                if N > 0:
                    (final_n, final_HS, _) = tf.while_loop(   # TODO add argument again if using b_n
                        cond=lambda n_counter, _1, _2: n_counter <= N,  # add argument
                        body=sketch_step,
                        loop_vars=(n, HS, a_n)#, b_n)
                    )
                    HS = final_HS

            sketches_tf = tf.pack(sketches)

        with tf.name_scope("scoring"):

            w_p = tf.get_variable(name="w_p", shape=[K],
                                   initializer=tf.random_uniform_initializer(dtype=tf.float32))
            if concat:
                wsp_size = 2*state_size
            else:
                wsp_size = state_size
            W_sp = tf.get_variable(name="W_sp", shape=[wsp_size, K],
                                   initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))

            if class_weights is not None:
                class_weights = tf.constant(class_weights, name="class_weights")

            def score(hs_j):
                """
                Score the word at index j, returns state vector for this word (column) across batch
                """
                if concat:
                    l = tf.matmul(tf.reshape(hs_j, [batch_size, 2*state_size]), W_sp) + w_p
                else:
                    l = tf.matmul(tf.reshape(hs_j, [batch_size, state_size]), W_sp) + w_p
                return l  # batch_size x K

            def score_predict_loss(score_input):
                """
                Predict a label for an input, compute the loss and return label and loss
                """
                [hs_i, y_words] = score_input
                word_label_score = score(hs_i)
                word_label_probs = tf.nn.softmax(word_label_score)
                word_preds = tf.argmax(word_label_probs, 1)
                y_words_full = tf.one_hot(tf.squeeze(y_words), depth=K, on_value=1.0, off_value=0.0)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(word_label_score,
                                                                                y_words_full)
                if class_weights is not None:
                    label_weights = tf.reduce_sum(tf.mul(y_words_full, class_weights), 1)
                    cross_entropy = tf.mul(cross_entropy, label_weights)
                return [word_preds, cross_entropy]

            if concat:
                S = HS
            else:
                S = tf.slice(HS, [0, 0, state_size], [batch_size, L, state_size])

            scores_pred = tf.map_fn(score_predict_loss,
                                    [tf.transpose(S, [1, 0, 2]), tf.transpose(y, [1,0])],
                                    dtype=[tf.int64, tf.float32])  # elems are unpacked along dim 0 -> L
            pred_labels = scores_pred[0]
            losses = scores_pred[1]

            pred_labels = mask*tf.transpose(pred_labels, [1, 0])  # masked, batch_size x L
            losses = tf.reduce_mean(tf.cast(mask, tf.float32)*tf.transpose(losses, [1, 0]),
                                    1)  # masked, batch_size x 1
            losses_reg = losses
            if l2_scale > 0:
                weights_list = [W_hss, W_sp]  # M_src, M_tgt word embeddings not included
                l2_loss = tf.contrib.layers.apply_regularization(
                    tf.contrib.layers.l2_regularizer(l2_scale), weights_list=weights_list)
                losses_reg += l2_loss
            if l1_scale > 0:
                weights_list = [W_hss, W_sp]
                l1_loss = tf.contrib.layers.apply_regularization(
                    tf.contrib.layers.l1_regularizer(l1_scale), weights_list=weights_list)
                losses_reg += l1_loss

        return losses, losses_reg, pred_labels, M_src, M_tgt, sketches_tf

    losses, losses_reg, predictions, M_src, M_tgt, sketches_tf = forward(inputs, labels, masks, seq_lens, class_weights)
    return losses, losses_reg, predictions, M_src, M_tgt, sketches_tf


def seq2seq(inputs, labels, masks, is_train, src_vocab_size, tgt_vocab_size, K, D, J, L, window_size,
           src_embeddings, tgt_embeddings, class_weights, l2_scale, keep_prob, l1_scale,
           keep_prob_sketch=1, attention_discount_factor=0.0, attention_temperature=1.0,
           lstm_units=0, bilstm=False, concat=False, r=0, N=0, seq_lens=None,
           activation=tf.nn.tanh, update_emb=True, track_sketches=False):

    """
    Encoder-decoder model for word-level QE predictions
    :param inputs:
    :param labels:
    :param masks:
    :param src_vocab_size:
    :param tgt_vocab_size:
    :param K:
    :param D:
    :param J:
    :param L:
    :param window_size:
    :param src_embeddings:
    :param tgt_embeddings:
    :param class_weights:
    :param l2_scale:
    :param keep_prob:
    :param l1_scale:
    :param keep_prob_sketch:
    :param lstm_units:
    :param bilstm:
    :param concat:
    :param r:
    :param N:
    :param seq_lens:
    :param activation:
    :param update_emb:
    :return:
    """
    batch_size = tf.shape(inputs)[0]
    with tf.name_scope("embedding"):
        if src_embeddings.table is None:
            #print "Random src embeddings of dimensionality %d" % D
            M_src = tf.get_variable(name="M_src", shape=[src_vocab_size, D],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
            emb_size = D
        else:
            M_src = tf.get_variable(name="M_src",
                            shape=[src_embeddings.table.shape[0],
                                   src_embeddings.table.shape[1]],
                            initializer=tf.constant_initializer(src_embeddings.table),
                            trainable=update_emb)
            D_loaded = len(tgt_embeddings.table[0])
            #print "Loading existing src embeddings of dimensionality %d" % D_loaded
            emb_size = D_loaded


        if tgt_embeddings.table is None:
            #print "Random tgt embeddings of dimensionality %d" % D
            M_tgt = tf.get_variable(name="M_tgt", shape=[tgt_vocab_size, D],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32))
            emb_size += D
        else:
            M_tgt = tf.get_variable(name="M_tgt",
                shape=[tgt_embeddings.table.shape[0],
                       tgt_embeddings.table.shape[1]],
                initializer=tf.constant_initializer(tgt_embeddings.table),
                trainable=update_emb)
            D_loaded = len(tgt_embeddings.table[0])
            #print "Loading existing tgt embeddings of dimensionality %d" % D_loaded
            emb_size += D_loaded

        # dropout for word embeddings ("pervasive dropout")
        M_tgt = tf.nn.dropout(M_tgt, keep_prob)
        M_src = tf.nn.dropout(M_src, keep_prob)

        # print "embedding size", emb_size
        x_src, x_tgt = tf.split(2, 2, inputs)  # split src and tgt part of input
        emb_tgt = tf.nn.embedding_lookup(M_tgt, x_src, name="emg_tgt")  # batch_size x L x window_size x emb_size
        emb_src = tf.nn.embedding_lookup(M_src, x_tgt, name="emb_src")  # batch_size x L x window_size x emb_size
        emb_comb = tf.concat(2, [emb_src, emb_tgt], name="emb_comb") # batch_size x L x 2*window_size x emb_size
        emb = tf.reshape(emb_comb, [batch_size, L, window_size*emb_size],
                         name="emb") # batch_size x L x window_size*emb_size

    with tf.name_scope("encoder"):
        encoder_inputs = emb
        #encoder_inputs = tf.reverse_sequence(emb, seq_lengths=seq_lens, seq_dim=1)  # TODO make param

        cell = tf.nn.rnn_cell.LSTMCell(lstm_units, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1, output_keep_prob=keep_prob)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=cell, inputs=encoder_inputs, sequence_length=seq_lens,
                                dtype=tf.float32, time_major=False)
        #print encoder_outputs, encoder_state

    with tf.name_scope("attention"): # see https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/python/ops/seq2seq.py
        attention_states = encoder_outputs

    with tf.name_scope("decoder"):
        output_size = K
        W = tf.get_variable(shape=[output_size, K], initializer=tf.contrib.layers.xavier_initializer(), name="output_W")
        b = tf.get_variable(initializer=tf.zeros_initializer(shape=[K]), name="output_b")
        output_projection = (W, b)
        feed_previous = ~is_train #tf.constant(True) #is_train  # scheduled sampling (greedy decoder), must be True during inference  # TODO
        initial_state_attention = True
        decoder_inputs = tf.transpose(labels, [1, 0])  # time-major
        decoder_inputs = tf.unpack(decoder_inputs)   #  decoder_inputs: A list of 1D int32 Tensors of shape [batch_size] --> split into time steps
        num_heads = 1  # number of attention heads that read from attention states
        embedding_size = J


        def decoder(feed_previous_bool):
            reuse = None if feed_previous_bool else True
            with tf.variable_scope(tf.get_variable_scope(),
                                             reuse=reuse):
                outputs, state = tf.nn.seq2seq.embedding_attention_decoder(
                    decoder_inputs=decoder_inputs, initial_state=encoder_state,
                    attention_states=attention_states, cell=cell, num_symbols=K,
                    embedding_size=embedding_size, num_heads=num_heads,
                    output_size=output_size, output_projection=output_projection,
                    feed_previous=feed_previous_bool, initial_state_attention=initial_state_attention)
                state_list = [state]
                if nest.is_sequence(state):
                    state_list = nest.flatten(state)
                return outputs + state_list

        outputs_and_state = tf.cond(feed_previous,  # two decoders, one with scheduled sampling for inference, one without
                                              lambda: decoder(True),
                                              lambda: decoder(False))
        outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
        state_list = outputs_and_state[outputs_len:]
        decoder_outputs = outputs_and_state[:outputs_len]

        #print "decoder outputs", decoder_outputs  # output projections: list of batch_size x K items
        #print "decoder state", decoder_state  # tuple: c; batch_size x lstm_units, batch_size x lstm_units
        decoder_packed = tf.pack(decoder_outputs, 0)  # time-major: L x batch_size x K

    logits = tf.transpose(decoder_packed, [1, 0, 2])  # batch-major
    #print "logits", logits
    labels_full = tf.one_hot(labels, depth=K, on_value=1.0, off_value=0.0)
    reshaped_logits = tf.reshape(logits, [batch_size*L, K])  # softmax needs 2D input
    softmax = tf.nn.softmax(reshaped_logits)


    # word-level cross-entropy
    xent = -labels_full*tf.log(tf.reshape(softmax, [batch_size, L, K]) + 1e-10)  # batch_size x L x K

    if class_weights is not None:
        class_weights = tf.constant(class_weights, name="class_weights")
        label_weights = tf.mul(labels_full, class_weights)  # batch_size x L x K
        xent = tf.mul(xent, label_weights)

    cross_entropy = tf.cast(masks, dtype=tf.float32)*tf.reduce_mean(xent, 2)  # batch_size x L

    # limit output to seq len
    pred_labels = masks*tf.argmax(tf.reshape(softmax, [batch_size, L, K]), 2)

    final_losses = cross_entropy

    return final_losses, final_losses, pred_labels, M_src, M_tgt, None  # TODO regularization?



def quetch(inputs, labels, masks, src_vocab_size, tgt_vocab_size, K, D, J, L, window_size,
           src_embeddings, tgt_embeddings, class_weights, l2_scale, keep_prob, l1_scale,
           keep_prob_sketch=1, attention_discount_factor=0.0, attention_temperature=1.0,
           lstm_units=0, bilstm=False, concat=False, r=0, N=0, seq_lens=None,
           activation=tf.nn.tanh, update_emb=True, track_sketches=False, is_train=False):
    """
    QUETCH model for word-level QE predictions  (MLP based on embeddings)
    :param inputs:
    :param labels:
    :param masks:
    :param seq_lens:
    :param src_vocab_size:
    :param tgt_vocab_size:
    :param K:
    :param D:
    :param N:
    :param J:
    :param L:
    :param r:
    :param lstm_units:
    :param bilstm:
    :param concat:
    :param window_size:
    :param src_embeddings:
    :param tgt_embeddings:
    :param class_weights:
    :param l2_scale:
    :param l1_scale:
    :param keep_prob:
    :return:
    """
    batch_size = tf.shape(inputs)[0]
    with tf.name_scope("embedding"):
        if src_embeddings.table is None:
            M_src = tf.get_variable(name="M_src", shape=[src_vocab_size, D],
                            initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, dtype=tf.float32))
            emb_size = D
        else:
            M_src = tf.get_variable(name="M_src",
                                    shape=[src_embeddings.table.shape[0],
                                           src_embeddings.table.shape[1]],
                                    initializer=tf.constant_initializer(src_embeddings.table),
                                    trainable=update_emb)
            D_loaded = len(tgt_embeddings.table[0])
            emb_size = D_loaded


        if tgt_embeddings.table is None:
            M_tgt = tf.get_variable(name="M_tgt", shape=[tgt_vocab_size, D],
                            initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True, dtype=tf.float32))
            emb_size += D
        else:
            M_tgt = tf.get_variable(name="M_tgt",
                                    shape=[tgt_embeddings.table.shape[0],
                                           tgt_embeddings.table.shape[1]],
                                    initializer=tf.constant_initializer(tgt_embeddings.table),
                                    trainable=update_emb)
            D_loaded = len(tgt_embeddings.table[0])
            emb_size += D_loaded

        # dropout for word embeddings ("pervasive dropout")
        #M_tgt = tf.nn.dropout(M_tgt, keep_prob)
        #M_src = tf.nn.dropout(M_src, keep_prob)

        x_tgt, x_src = tf.split(2, 2, inputs)  # split src and tgt part of input
        emb_tgt = tf.nn.embedding_lookup(M_tgt, x_tgt, name="emg_tgt")  # batch_size x L x window_size x emb_size
        emb_src = tf.nn.embedding_lookup(M_src, x_src, name="emb_src")  # batch_size x L x window_size x emb_size
        emb_comb = tf.concat(2, [emb_src, emb_tgt], name="emb_comb") # batch_size x L x 2*window_size x emb_size
        emb = tf.reshape(emb_comb, [batch_size, -1, window_size*emb_size], name="emb") # batch_size x L x window_size*emb_size

        emb = tf.nn.dropout(emb, keep_prob=keep_prob, name="drop_emb")

    with tf.name_scope("mlp"):
        inputs = tf.reshape(emb, [-1, window_size*emb_size])
        """
        # 1 hidden layer
        W_1 = tf.get_variable(shape=[window_size*emb_size, J],
                              initializer=tf.contrib.layers.xavier_initializer(
                                  uniform=True, dtype=tf.float32, ), name="W_1")
        W_2 = tf.get_variable(shape=[J, K],
                              initializer=tf.contrib.layers.xavier_initializer(
                                  uniform=True, dtype=tf.float32), name="W_2")
        b_1 = tf.get_variable(shape=[J], initializer=tf.random_uniform_initializer(
            dtype=tf.float32), name="b_1")
        b_2 = tf.get_variable(shape=[K], initializer=tf.random_uniform_initializer(
            dtype=tf.float32), name="b_2")
        hidden = activation(tf.matmul(inputs, W_1) + b_1)  # batch_size*emb_size, J
        hidden = tf.nn.dropout(hidden, keep_prob)
        out = tf.matmul(hidden, W_2) + b_2  # batch_size*emb_size, K
        """

        # 2 hidden layers
        J2 = 96
        W_1 = tf.get_variable(shape=[window_size*emb_size, J2],
                              initializer=tf.contrib.layers.xavier_initializer(
                                  uniform=True, dtype=tf.float32, ), name="W_1")
        W_2 = tf.get_variable(shape=[J2, J],
                              initializer=tf.contrib.layers.xavier_initializer(
                                  uniform=True, dtype=tf.float32), name="W_2")
        W_3 = tf.get_variable(shape=[J, K],
                              initializer=tf.contrib.layers.xavier_initializer(
                                  uniform=True, dtype=tf.float32), name="W_3")
        b_1 = tf.get_variable(shape=[J2], initializer=tf.random_uniform_initializer(
            dtype=tf.float32), name="b_1")
        b_2 = tf.get_variable(shape=[J], initializer=tf.random_uniform_initializer(
            dtype=tf.float32), name="b_2")
        b_3 = tf.get_variable(shape=[K], initializer=tf.random_uniform_initializer(
            dtype=tf.float32), name="b_3")
        hidden1 = activation(tf.matmul(inputs, W_1) + b_1)  # batch_size*emb_size, J2
        hidden = activation(tf.matmul(hidden1, W_2) + b_2)  # batch_size*emb_size, J
        hidden = tf.nn.dropout(hidden, keep_prob)
        out = tf.matmul(hidden, W_3) + b_3  # batch_size*emb_size, K

        logits = tf.reshape(out, [batch_size, L, K])
        softmax = tf.nn.softmax(out)  # batch_size*L, K
        pred_labels = masks*tf.argmax(logits, 2)  # batch_size x L
        y_full = tf.one_hot(labels, depth=K, on_value=1.0, off_value=0.0)  # batch_size x L x K
        xent = -y_full*tf.log(tf.reshape(softmax, [batch_size, L, K]) + 1e-10)  # batch_size x L x K

        if class_weights is not None:
            class_weights = tf.constant(class_weights, name="class_weights")
            label_weights = tf.mul(y_full, class_weights)  # batch_size x L x K
            xent = tf.mul(xent, label_weights)

        cross_entropy = tf.cast(masks, dtype=tf.float32)*tf.reduce_mean(xent, 2)  # batch_size x L

        losses = cross_entropy
        losses_reg = losses

        if l2_scale > 0:
            weights_list = [W_1, W_2]  # M_src, M_tgt,
            l2_loss = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l2_regularizer(l2_scale), weights_list=weights_list)
            losses_reg += l2_loss
        if l1_scale > 0:
            weights_list = [W_1, W_2]  # M_src, M_tgt,
            l1_loss = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l1_regularizer(l1_scale), weights_list=weights_list)
            losses_reg += l1_loss

    return losses, losses_reg, pred_labels, M_src, M_tgt, None
