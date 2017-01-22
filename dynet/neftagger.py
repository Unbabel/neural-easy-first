import dynet as dy
import numpy as np
from collections import Counter
import random
import argparse
import sys
import time
import util
import pdb

class NeuralEasyFirstTagger(object):
    def __init__(self, word_vocabulary, affix_length,
                 prefix_vocabularies, suffix_vocabularies,
                 tag_vocabulary, model_type,
                 attention_type, temperature, discount_factor, embedding_size,
                 affix_embedding_size,
                 hidden_size, preattention_size, sketch_size, context_size,
                 concatenate_last_layer,
                 sum_hidden_states_and_sketches,
                 share_attention_sketch_parameters,
                 use_sketch_losses,
                 use_max_pooling,
                 use_bilstm):
        self.word_vocabulary = word_vocabulary
        self.affix_length = affix_length
        self.prefix_vocabularies = prefix_vocabularies
        self.suffix_vocabularies = suffix_vocabularies
        self.tag_vocabulary = tag_vocabulary
        self.model_type = model_type
        self.attention_type = attention_type
        self.temperature = temperature
        self.discount_factor = discount_factor
        self.embedding_size = embedding_size
        self.affix_embedding_size = affix_embedding_size
        self.hidden_size = hidden_size
        self.preattention_size = preattention_size
        self.sketch_size = sketch_size
        self.context_size = context_size
        self.concatenate_last_layer = concatenate_last_layer
        self.sum_hidden_states_and_sketches = sum_hidden_states_and_sketches
        self.share_attention_sketch_parameters = \
            share_attention_sketch_parameters
        self.use_sketch_losses = use_sketch_losses
        self.use_max_pooling = use_max_pooling
        self.use_bilstm = use_bilstm
        self.model = None
        self.parameters = {}
        self.builders = []
        self.track_sketches = False
        self.sketch_file = None
        self.use_last_sketch = False #True

    def init_parameters(self, model, dims):
        assert len(dims) == 2
        return model.add_parameters(dims)
        #u = np.sqrt(6. / (dims[0] + dims[1]))
        #W = u * (2. * np.random.rand(dims[0], dims[1]) - 1.)
        #return model.parameters_from_numpy(W)

    def create_model(self, embeddings=None):
        model = dy.Model()
        parameters = {}
        if embeddings != None:
            num_words = len(set(embeddings.keys()).
                            union(set(self.word_vocabulary.w2i.keys())))
        else:
            num_words = self.word_vocabulary.size()
        num_tags = self.tag_vocabulary.size()
        window_size = 1+2*self.context_size
        parameters['E'] = model.add_lookup_parameters((num_words,
                                                       self.embedding_size))
        if embeddings != None:
            for word in embeddings:
                if word not in self.word_vocabulary.w2i:
                    self.word_vocabulary.w2i[word] = \
                        len(self.word_vocabulary.w2i.keys())
                parameters['E'].init_row(self.word_vocabulary.w2i[word],
                                         embeddings[word])

        for l in xrange(self.affix_length):
            num_prefixes = self.prefix_vocabularies[l].size()
            parameters['E_prefix_%d' % (l+1)] = model.add_lookup_parameters(
                (num_prefixes, self.affix_embedding_size))
            num_suffixes = self.suffix_vocabularies[l].size()
            parameters['E_suffix_%d' % (l+1)] = model.add_lookup_parameters(
                (num_suffixes, self.affix_embedding_size))

        if self.sum_hidden_states_and_sketches:
            assert not self.concatenate_last_layer
            assert self.sketch_size == 2*self.hidden_size
            state_size = self.sketch_size
        else:
            state_size = 2*self.hidden_size + self.sketch_size

        if self.use_last_sketch:
            parameters['W_sz'] = self.init_parameters(
                model, (self.preattention_size, self.sketch_size))
        parameters['W_cz'] = self.init_parameters(
            model, (self.preattention_size, window_size*state_size))
        parameters['w_z'] = model.parameters_from_numpy(
            np.zeros((self.preattention_size, 1)))
        parameters['v'] = self.init_parameters(
            model, (1, self.preattention_size))

        if self.share_attention_sketch_parameters:
            assert self.sketch_size == self.preattention_size
            parameters['W_cs'] = parameters['W_cz']
            parameters['w_s'] = parameters['w_z']
        else:
            parameters['W_cs'] = self.init_parameters(
                model, (self.sketch_size, window_size*state_size))
            parameters['w_s'] = model.parameters_from_numpy(
                np.zeros((self.sketch_size, 1)))
        if self.concatenate_last_layer:
            parameters['O'] = model.add_parameters(
                (num_tags, 2*self.hidden_size + self.sketch_size))
        else:
            parameters['O'] = model.add_parameters((num_tags,
                                                    self.sketch_size))

        self.model = model
        input_size = self.embedding_size + 2*self.affix_embedding_size
        if self.use_bilstm:
            self.builders = [dy.LSTMBuilder(1, input_size,
                                            self.hidden_size, self.model),
                             dy.LSTMBuilder(1, input_size,
                                            self.hidden_size, self.model)]
        else:
            self.builders = None
            parameters['W_xh'] = self.init_parameters(
                self.model, (2*self.hidden_size, input_size))
            parameters['w_h'] = model.parameters_from_numpy(
                np.zeros((2*self.hidden_size, 1)))
        self.parameters = parameters

    def squared_norm_of_parameters(self, print_norms=False):
        squared_norm = dy.scalarInput(0.)
        for key in self.parameters:
            if type(self.parameters[key]) == dy.LookupParameters:
                continue
                #w = self.parameters[key]
            elif self.share_attention_sketch_parameters \
                 and key in ['W_cs', 'w_s']:
                continue
            else:
                w = dy.parameter(self.parameters[key])
            tmp = dy.trace_of_product(w, w)
            if print_norms:
                print 'Norm of %s: %f' % (key, tmp.npvalue())
            squared_norm += tmp
            #squared_norm += dy.trace_of_product(w, w)
        return squared_norm

    def build_graph(self, instance, num_sketches=-1, noise_level=0.1,
                    training=True, epoch=-1, ordering=None):
        unk = self.word_vocabulary.w2i['_UNK_']
        words = [self.word_vocabulary.w2i.get(w, unk) for w, _ in instance]
        tags = [self.tag_vocabulary.w2i[t] for _, t in instance]

        if training:
            errs = []

        dy.renew_cg()

        E = self.parameters['E']
        wembs = [E[w] for w in words]
        wembs = [dy.noise(we, noise_level) for we in wembs]

        if self.affix_length:
            pembs = []
            for l in xrange(self.affix_length):
                E_prefix = self.parameters['E_prefix_%d' % (l+1)]
                punk = self.prefix_vocabularies[l].w2i['_UNK_']
                prefixes = [self.prefix_vocabularies[l].w2i.get(w[:(l+1)], punk) \
                            for w, _ in instance]
                pembs.append([E_prefix[p] for p in prefixes])
                pembs[l] = [dy.noise(pe, noise_level) for pe in pembs[l]]
            sembs = []
            for l in xrange(self.affix_length):
                E_suffix = self.parameters['E_suffix_%d' % (l+1)]
                sunk = self.suffix_vocabularies[l].w2i['_UNK_']
                suffixes = [self.suffix_vocabularies[l].w2i.get(w[-(l+1):], sunk) \
                            for w, _ in instance]
                sembs.append([E_suffix[s] for s in suffixes])
                sembs[l] = [dy.noise(se, noise_level) for se in sembs[l]]
            for i in xrange(len(wembs)):
                pemb = [pembs[l][i] for l in xrange(self.affix_length)]
                semb = [sembs[l][i] for l in xrange(self.affix_length)]
                wembs[i] = dy.concatenate([wembs[i],
                                           dy.esum(pemb),
                                           dy.esum(semb)])

        if self.use_bilstm:
            f_init, b_init = [b.initial_state() for b in self.builders]
            fw = [x.output() for x in f_init.add_inputs(wembs)]
            bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

            hidden_states = []
            for f, b in zip(fw, reversed(bw)):
                hidden_states.append(dy.concatenate([f, b]))
        else:
            #assert 2*self.hidden_size == self.embedding_size + \
            #    2*self.affix_embedding_size
            #hidden_states = wembs
            W_xh = dy.parameter(self.parameters['W_xh'])
            w_h = dy.parameter(self.parameters['w_h'])
            hidden_states = [dy.tanh(W_xh * x + w_h) for x in wembs]

        if self.use_last_sketch:
            W_sz = dy.parameter(self.parameters['W_sz'])
        W_cz = dy.parameter(self.parameters['W_cz'])
        w_z = dy.parameter(self.parameters['w_z'])
        v = dy.parameter(self.parameters['v'])
        W_cs = dy.parameter(self.parameters['W_cs'])
        w_s = dy.parameter(self.parameters['w_s'])
        O = dy.parameter(self.parameters['O'])

        # Initialize all word sketches as zero vectors.
        sketches = []
        last_sketch_vector = dy.vecInput(self.sketch_size)
        last_sketch_vector.set(np.zeros(self.sketch_size))
        for i, word in enumerate(words):
            if self.sum_hidden_states_and_sketches:
                assert self.sketch_size == 2*self.hidden_size
                sketch = hidden_states[i]
            else:
                sketch = dy.vecInput(self.sketch_size)
                sketch.set(np.zeros(self.sketch_size))
            sketches.append(sketch)

        # Make several sketch steps.
        if num_sketches < 0:
            num_sketches = len(words)
        cumulative_attention = dy.vecInput(len(words))
        cumulative_attention.set(np.zeros(len(words)))
        #cumulative_attention.set(np.ones(len(words)) / len(words))
        if self.sum_hidden_states_and_sketches:
            state_size = self.sketch_size
        else:
            state_size = 2*self.hidden_size + self.sketch_size
        for j in xrange(num_sketches):
            z = []
            states = []
            states_with_context = []
            for i in xrange(len(words)):
                if self.sum_hidden_states_and_sketches:
                    state = sketches[i]
                else:
                    state = dy.concatenate([hidden_states[i], sketches[i]])
                states.append(state)
            #pdb.set_trace()
            for i in xrange(len(words)):
                state_with_context = states[i]
                for l in xrange(1, self.context_size+1):
                    if i-l < 0:
                        state = dy.vecInput(state_size)
                        state.set(np.zeros(state_size)) # Note: should these be zeros?
                    else:
                        state = states[i-l]
                    state_with_context = dy.concatenate([state_with_context,
                                                         state])
                for l in xrange(1, self.context_size+1):
                    if i+l >= len(states):
                        state = dy.vecInput(state_size)
                        state.set(np.zeros(state_size)) # Note: should these be zeros?
                    else:
                        state = states[i+l]
                    state_with_context = dy.concatenate([state_with_context,
                                                         state])
                if self.use_max_pooling:
                    assert self.preattention_size == self.sketch_size
                    preattention = dy.tanh(W_cz * state_with_context + w_z)
                    if self.concatenate_last_layer:
                        r_i = O * dy.concatenate([hidden_states[i],
                                                  preattention])
                    else:
                        r_i = O * preattention
                    scores = dy.transpose(dy.concatenate_cols([r_i]))
                    z_i = dy.kmax_pooling(scores, 1)[0] # Best score.
                else:
                    if self.use_last_sketch:
                        z_i = v * dy.tanh(W_cz * state_with_context + \
                                          W_sz * last_sketch_vector + w_z)
                    else:
                        z_i = v * dy.tanh(W_cz * state_with_context + w_z)
                z.append(z_i)
                states_with_context.append(state_with_context)
            temperature = self.temperature #10. # 1.
            discount_factor = self.discount_factor #5. #10. #50. # 0.
            #attention_weights = dy.softmax(dy.concatenate(z)/temperature)
            if discount_factor == 0:
                scores = dy.concatenate(z)
            else:
                scores = dy.concatenate(z) - \
                         cumulative_attention * discount_factor
            if self.attention_type == 'softmax':
                attention_weights = dy.softmax(scores / temperature)
            elif self.attention_type == 'sparsemax':
                attention_weights = dy.sparsemax(scores / temperature)
            elif self.attention_type == 'constrained_softmax':
                constraints = -cumulative_attention + 1.
                attention_weights = dy.constrained_softmax(scores / temperature,
                                                           constraints)
            elif self.attention_type == 'left_to_right':
                attention_weights = dy.vecInput(len(words))
                a = np.zeros(len(words))
                a[j] = 1.
                attention_weights.set(a)
            elif self.attention_type == 'right_to_left':
                attention_weights = dy.vecInput(len(words))
                a = np.zeros(len(words))
                a[len(words)-1-j] = 1.
                attention_weights.set(a)
            elif self.attention_type == 'prescribed_order':
                attention_weights = dy.vecInput(len(words))
                a = np.zeros(len(words))
                a[ordering[j]] = 1.
                attention_weights.set(a)
            else:
                raise NotImplementedError
            if self.track_sketches:
                self.sketch_file.write('%s\n' % ' '.join(['{:.3f}'.format(p) \
                    for p in attention_weights.npvalue()]))
            cumulative_attention += attention_weights
            #cumulative_attention = \
            #    (attention_weights + cumulative_attention * j) / (j+1)
            #if not training:
            #    pdb.set_trace()
            if self.model_type == 'single_state':
                cbar = dy.esum([vector*attention_weight
                                for vector, attention_weight in
                                zip(states_with_context, attention_weights)])
                s_n = dy.tanh(W_cs * cbar + w_s)
                last_sketch_vector = s_n
                if training and self.use_sketch_losses:
                    state = s_n
                    if self.concatenate_last_layer:
                        state = dy.concatenate([hidden_states[i], s_n])
                    else:
                        state = s_n
                    r_t = O * state
                    for i, t in enumerate(tags):
                        err = dy.pickneglogsoftmax(r_t, t)
                        errs.append(err * attention_weights[i] / float(j+1))
                        #errs.append(err * attention_weights[i] / float(num_sketches))
                #if self.sum_hidden_states_and_sketches:
                #sketches = [sketch * (1-weight) + s_n * weight
                #            for sketch, weight in \
                #            zip(sketches, attention_weights)]
                #else:
                sketches = [sketch + s_n * weight
                            for sketch, weight in \
                            zip(sketches, attention_weights)]
            elif self.model_type == 'all_states':
                for i in xrange(len(words)):
                    s_n = dy.tanh(W_cs * states_with_context[i] + w_s)
                    sketches[i] += s_n * attention_weights[i]
                    #sketches[i] = sketches[i] * (1-attention_weights[i]) + \
                    #              s_n * attention_weights[i]
            else:
                raise NotImplementedError

        #pdb.set_trace()

        # Now use the last sketch to make a prediction.
        if training:
            predicted_tags = []
            for i, t in enumerate(tags):
                if self.concatenate_last_layer:
                    state = dy.concatenate([hidden_states[i], sketches[i]])
                else:
                    state = sketches[i]
                r_t = O * state
                err = dy.pickneglogsoftmax(r_t, t)
                #if self.use_sketch_losses and not self.concatenate_last_layer:
                #    errs.append(err * 0.001)
                #else:
                errs.append(err)
                chosen = np.argmax(r_t.npvalue())
                predicted_tags.append(self.tag_vocabulary.i2w[chosen])

            if self.track_sketches:
                self.sketch_file.write(' '.join([w for w, _ in instance]) +
                                       '\n')
                self.sketch_file.write(' '.join([t for _, t in instance]) +
                                       '\n')
                self.sketch_file.write(' '.join(predicted_tags) + '\n')
                self.sketch_file.write('\n')

            return dy.esum(errs), predicted_tags
        else:
            predicted_tags = []
            for i in xrange(len(words)):
                if self.concatenate_last_layer:
                    state = dy.concatenate([hidden_states[i], sketches[i]])
                else:
                    state = sketches[i]
                r_t = O * state
                #out = dy.softmax(r_t)
                chosen = np.argmax(r_t.npvalue())
                predicted_tags.append(self.tag_vocabulary.i2w[chosen])

            if self.track_sketches:
                self.sketch_file.write(' '.join([w for w, _ in instance]) +
                                       '\n')
                self.sketch_file.write(' '.join([t for _, t in instance]) +
                                       '\n')
                self.sketch_file.write(' '.join(predicted_tags) + '\n')
                self.sketch_file.write('\n')

            return predicted_tags


def read_dataset(fname, maximum_sentence_length=-1, read_ordering=False):
    sent = []
    ordering = None
    sentences = []
    orderings = []
    for line in file(fname):
        if ordering == None and read_ordering:
            line = line.lstrip('#')
            line = line.strip().split(' ')
            ordering = [int(index) for index in line]
            continue
        else:
            line = line.strip().split()
        if not line:
            if sent and (maximum_sentence_length < 0 or
                         len(sent) < maximum_sentence_length):
                if read_ordering:
                    sentences.append(sent)
                    orderings.append(ordering)
                else:
                    sentences.append(sent)
            sent = []
            ordering = None
        else:
            w, t = line[0], line[-1]
            #w, t = line[-2:]
            sent.append((w, t))
    if read_ordering:
        return sentences, orderings
    else:
        return sentences

def create_vocabularies(corpora, word_cutoff=0, affix_length=0):
    word_counter = Counter()
    tag_counter = Counter()
    prefix_counter = [Counter() for _ in xrange(affix_length)]
    suffix_counter = [Counter() for _ in xrange(affix_length)]
    word_counter['_UNK_'] = word_cutoff+1
    for l in xrange(affix_length):
        prefix_counter[l]['_UNK_'] = word_cutoff+1
        suffix_counter[l]['_UNK_'] = word_cutoff+1
    for corpus in corpora:
        for s in corpus:
            for w, t in s:
                word_counter[w] += 1
                tag_counter[t] += 1
                for l in xrange(affix_length):
                    prefix_counter[l][w[:(l+1)]] += 1
                    suffix_counter[l][w[-(l+1):]] += 1

    words = [w for w in word_counter if word_counter[w] > word_cutoff]
    tags = [t for t in tag_counter]

    word_vocabulary = util.Vocab.from_corpus([words])
    tag_vocabulary = util.Vocab.from_corpus([tags])

    prefix_vocabularies = []
    suffix_vocabularies = []
    for l in xrange(affix_length):
        prefixes = [p for p in prefix_counter[l] \
                    if prefix_counter[l][p] > word_cutoff]
        prefix_vocabularies.append(util.Vocab.from_corpus([prefixes]))
        suffixes = [s for s in suffix_counter[l] \
                    if suffix_counter[l][s] > word_cutoff]
        suffix_vocabularies.append(util.Vocab.from_corpus([suffixes]))

    print >> sys.stderr, 'Words: %d' % word_vocabulary.size()
    print >> sys.stderr, 'Tags: %d' % tag_vocabulary.size()

    if affix_length > 0:
        return word_vocabulary, prefix_vocabularies, suffix_vocabularies, \
            tag_vocabulary
    else:
        return word_vocabulary, tag_vocabulary

def load_embeddings(embeddings_file):
    embeddings = {}
    f = open(embeddings_file)
    for line in f:
        fields = line.split(' ')
        word = fields[0]
        v = [float(val) for val in fields[1:]]
        embeddings[word] = v
    f.close()
    return embeddings

def main():
    '''Main function.'''
    # Parse arguments.
    parser = argparse.ArgumentParser(
        prog='Neural Easy-First POS Tagger',
        description='Trains/test a neural easy-first POS tagger.')

    # Need to be here as an argument to allow specifying a seed.
    parser.add_argument('--dynet-seed', type=str, default=0)
    parser.add_argument('--dynet-mem', type=str, default=512)

    parser.add_argument('-train_file', type=str, default='')
    parser.add_argument('-dev_file', type=str, default='')
    parser.add_argument('-test_file', type=str, default='')
    parser.add_argument('-embeddings_file', type=str, default='')
    parser.add_argument('-affix_length', type=int, default=0)
    parser.add_argument('-noise_level', type=float, default=0.0)
    parser.add_argument('-concatenate_last_layer', type=int, default=1)
    parser.add_argument('-sum_hidden_states_and_sketches', type=int, default=0)
    parser.add_argument('-share_attention_sketch_parameters', type=int,
                        default=0)
    parser.add_argument('-use_sketch_losses', type=int, default=0)
    parser.add_argument('-use_max_pooling', type=int, default=0)
    parser.add_argument('-use_bilstm', type=int, default=1)
    parser.add_argument('-affix_embedding_size', type=int, default=0)
    parser.add_argument('-embedding_size', type=int, default=64) # 128
    parser.add_argument('-hidden_size', type=int, default=20) # 50
    parser.add_argument('-preattention_size', type=int, default=20) # 50
    parser.add_argument('-sketch_size', type=int, default=20) # 50
    parser.add_argument('-context_size', type=int, default=1) # 0
    parser.add_argument('-num_sketches', type=int, default=-1)
    parser.add_argument('-maximum_sentence_length', type=int, default=100)
    parser.add_argument('-num_epochs', type=int, default=50)
    parser.add_argument('-num_pretraining_epochs', type=int, default=0)
    parser.add_argument('-l2_regularization', type=float, default=0.)
    parser.add_argument('-model_type', type=str, default='single_state')
    parser.add_argument('-attention_type', type=str, default='softmax')
    parser.add_argument('-temperature', type=float, default=1.)
    parser.add_argument('-discount_factor', type=float, default=0.)
    parser.add_argument('-sketch_file_dev', type=str, required=True)
    parser.add_argument('-sketch_file_test', type=str, required=True)
    parser.add_argument('-metric', type=str, default='accuracy')
    parser.add_argument('-null_label', type=str, default='O')

    args = vars(parser.parse_args())
    print >> sys.stderr, args

    train_file = args['train_file']
    dev_file = args['dev_file']
    test_file = args['test_file']
    embeddings_file = args['embeddings_file']
    affix_length = args['affix_length']
    noise_level = args['noise_level']
    concatenate_last_layer = args['concatenate_last_layer']
    sum_hidden_states_and_sketches = args['sum_hidden_states_and_sketches']
    share_attention_sketch_parameters = \
        args['share_attention_sketch_parameters']
    use_sketch_losses = args['use_sketch_losses']
    use_max_pooling = args['use_max_pooling']
    use_bilstm = args['use_bilstm']
    embedding_size = args['embedding_size']
    affix_embedding_size = args['affix_embedding_size']
    hidden_size = args['hidden_size']
    preattention_size = args['preattention_size']
    sketch_size = args['sketch_size']
    context_size = args['context_size']
    num_sketches = args['num_sketches']
    maximum_sentence_length = args['maximum_sentence_length']
    num_epochs = args['num_epochs']
    num_pretraining_epochs = args['num_pretraining_epochs']
    l2_regularization = args['l2_regularization']
    model_type = args['model_type']
    attention_type = args['attention_type']
    temperature = args['temperature']
    discount_factor = args['discount_factor']
    sketch_file_dev = args['sketch_file_dev']
    sketch_file_test = args['sketch_file_test']
    metric = args['metric']
    null_label = args['null_label']

    np.random.seed(42)

#    suffix = 'model-%s_attention-%s_temp-%f_disc-%f_C-%f_sketches-%d_' \
#             'cat-%d_emb-%d_hid-%d_ctx-%d' % \
#             (model_type, attention_type, temperature, discount_factor,
#              l2_regularization, num_sketches, concatenate_last_layer,
#              embedding_size, hidden_size, context_size)

    # Read corpus (train, dev, test).
    read_ordering=False
    print >> sys.stderr
    print >> sys.stderr, 'Loading train/dev/test datasets...'
    if read_ordering:
        train_instances, train_orderings = read_dataset(
            train_file,
            maximum_sentence_length=maximum_sentence_length,
            read_ordering=True)
        dev_instances, dev_orderings = read_dataset(dev_file,
                                                    read_ordering=True)
        test_instances, test_orderings = read_dataset(test_file,
                                                      read_ordering=True)
    else:
        train_instances = read_dataset(
            train_file,
            maximum_sentence_length=maximum_sentence_length)
        dev_instances = read_dataset(dev_file)
        test_instances = read_dataset(test_file)


    if affix_length > 0:
        word_vocabulary, prefix_vocabularies, suffix_vocabularies, \
            tag_vocabulary = create_vocabularies([train_instances],
                                                 word_cutoff=1,
                                                 affix_length=affix_length)
    else:
        word_vocabulary, tag_vocabulary = create_vocabularies([train_instances],
                                                              word_cutoff=1)
        prefix_vocabularies = []
        suffix_vocabularies = []

    # Create model.
    tagger = NeuralEasyFirstTagger(word_vocabulary, affix_length,
                                   prefix_vocabularies, suffix_vocabularies,
                                   tag_vocabulary, model_type,
                                   attention_type, temperature, discount_factor,
                                   embedding_size, affix_embedding_size,
                                   hidden_size,
                                   preattention_size, sketch_size,
                                   context_size,
                                   concatenate_last_layer,
                                   sum_hidden_states_and_sketches,
                                   share_attention_sketch_parameters,
                                   use_sketch_losses,
                                   use_max_pooling, use_bilstm)
    if embeddings_file != '':
        embeddings = load_embeddings(embeddings_file)
    else:
        embeddings = None
    tagger.create_model(embeddings=embeddings)

    # Train.
    print >> sys.stderr
    print >> sys.stderr, 'Training...'
    tic = time.time()
    #trainer = dy.AdamTrainer(tagger.model, alpha=0.0001)
    trainer = dy.AdagradTrainer(tagger.model, e0=0.1)
    #trainer.set_clip_threshold(5.0)
    #trainer = dy.SimpleSGDTrainer(tagger.model, e0=0.1)
    best_epoch = -1
    best_dev_accuracy = 0.
    best_test_accuracy = 0.
    for epoch in xrange(num_epochs):
        num_numeric_issues = 0
        tagged = correct = loss = reg = 0
        matches = 0
        predicted = 0
        gold = 0
        #random.shuffle(train_instances)
        for i, instance in enumerate(train_instances, 1):
            if read_ordering:
                ordering = train_orderings[i-1]
            else:
                ordering = None
            gold_tags = [t for _, t in instance]
            attention_type = tagger.attention_type
            if epoch < num_pretraining_epochs:
                tagger.attention_type = 'prescribed_order'
                #if i%2:
                #    tagger.attention_type = 'left_to_right'
                #else:
                #    tagger.attention_type = 'right_to_left'
            sum_errs, predicted_tags = \
                tagger.build_graph(instance,
                                   num_sketches=num_sketches,
                                   noise_level=noise_level,
                                   ordering=ordering)
            tagger.attention_type = attention_type
            val = sum_errs.scalar_value()
            if np.isnan(val):
                print >> sys.stderr, \
                    'Numeric problems in sentence %d (%d words long; ' \
                    'previous sentence was %d words long)' % \
                    (i, len(instance), len(train_instances[i-2]))
                num_numeric_issues += 1
                if num_numeric_issues > 10:
                    assert False
                continue
            loss += val
            sum_errs += tagger.squared_norm_of_parameters() * \
                        l2_regularization
            reg += (sum_errs.scalar_value() - val)
            tagged += len(instance)
            correct += sum([int(g == p)
                            for g, p in zip(gold_tags, predicted_tags)])
            if metric == 'f1':
                matches += sum([int(g == p)
                                for g, p in zip(gold_tags, predicted_tags) \
                                if g != null_label])
                predicted += len([p for p in predicted_tags if p != null_label])
                gold += len([g for g in gold_tags if g != null_label])
            sum_errs.backward()
            #if len(gold_tags) <= 5: pdb.set_trace()
            trainer.update()

        if metric == 'f1':
            precision = float(matches) / predicted
            recall = float(matches) / gold
            f1 = 2.*precision*recall / (precision + recall)
            train_accuracy = f1
        else:
            train_accuracy = float(correct) / tagged

        # Check accuracy in dev set.
        tagger.track_sketches = True
        tagger.sketch_file = open(sketch_file_dev + '.tmp', 'w')
        correct = 0
        total = 0
        matches = 0
        predicted = 0
        gold = 0
        for i, instance in enumerate(dev_instances):
            if read_ordering:
                ordering = dev_orderings[i]
            else:
                ordering = None
            gold_tags = [t for _, t in instance]
            predicted_tags = tagger.build_graph(instance,
                                                num_sketches=num_sketches,
                                                noise_level=0.,
                                                training=False,
                                                epoch=epoch,
                                                ordering=ordering)
            correct += sum([int(g == p)
                            for g, p in zip(gold_tags, predicted_tags)])
            total += len(gold_tags)
            if metric == 'f1':
                matches += sum([int(g == p)
                                for g, p in zip(gold_tags, predicted_tags) \
                                if g != null_label])
                predicted += len([p for p in predicted_tags if p != null_label])
                gold += len([g for g in gold_tags if g != null_label])
        if metric == 'f1':
            precision = float(matches) / predicted
            recall = float(matches) / gold
            f1 = 2.*precision*recall / (precision + recall)
            dev_accuracy = f1
        else:
            dev_accuracy = float(correct) / total
        tagger.sketch_file.close()

        # Check accuracy in test set.
        tagger.sketch_file = open(sketch_file_test + '.tmp', 'w')
        correct = 0
        total = 0
        matches = 0
        predicted = 0
        gold = 0
        for i, instance in enumerate(test_instances):
            if read_ordering:
                ordering = test_orderings[i]
            else:
                ordering = None
            gold_tags = [t for _, t in instance]
            predicted_tags = tagger.build_graph(instance,
                                                num_sketches=num_sketches,
                                                noise_level=0.,
                                                training=False,
                                                epoch=epoch,
                                                ordering=ordering)
            correct += sum([int(g == p)
                            for g, p in zip(gold_tags, predicted_tags)])
            total += len(gold_tags)
            if metric == 'f1':
                matches += sum([int(g == p)
                                for g, p in zip(gold_tags, predicted_tags) \
                                if g != null_label])
                predicted += len([p for p in predicted_tags if p != null_label])
                gold += len([g for g in gold_tags if g != null_label])
        if metric == 'f1':
            precision = float(matches) / predicted
            recall = float(matches) / gold
            f1 = 2.*precision*recall / (precision + recall)
            test_accuracy = f1
        else:
            test_accuracy = float(correct) / total
        tagger.sketch_file.close()
        tagger.track_sketches = False

        # Check if this is the best model so far (on dev).
        if epoch == 0 or dev_accuracy > best_dev_accuracy:
            best_epoch = epoch
            best_dev_accuracy = dev_accuracy
            best_test_accuracy = test_accuracy
            from shutil import copyfile
            copyfile(sketch_file_dev + '.tmp', sketch_file_dev)
            copyfile(sketch_file_test + '.tmp', sketch_file_test)

        # Plot epoch statistics.
        trainer.status()
        print >> sys.stderr, \
            'Epoch: %d, Loss: %f, Reg: %f, Train acc: %f, Dev acc: %f, ' \
            'Best Dev acc: %f, Test acc: %f, Best Test acc: %f' \
            % (epoch+1,
               loss/tagged,
               reg/tagged,
               train_accuracy,
               dev_accuracy,
               best_dev_accuracy,
               test_accuracy,
               best_test_accuracy)

    toc = time.time()
    print >> sys.stderr, 'Final Dev Accuracy: %f.' % best_dev_accuracy
    print >> sys.stderr, 'Final Test Accuracy: %f.' % best_test_accuracy
    print >> sys.stderr, 'Training took %f miliseconds.' % (toc - tic)


if __name__ == "__main__":
    main()

