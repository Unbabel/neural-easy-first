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
    def __init__(self, word_vocabulary, tag_vocabulary, embedding_size,
                 hidden_size, context_size, concatenate_last_layer):
        self.word_vocabulary = word_vocabulary
        self.tag_vocabulary = tag_vocabulary
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.concatenate_last_layer = concatenate_last_layer
        self.model = None
        self.parameters = {}
        self.builders = []

    def create_model(self):
        model = dy.Model()
        parameters = {}
        num_words = self.word_vocabulary.size()
        num_tags = self.tag_vocabulary.size()
        window_size = 1+2*self.context_size
        parameters['E'] = model.add_lookup_parameters((num_words,
                                                       self.embedding_size))
        parameters['W_cz'] = model.add_parameters(
            (self.hidden_size, 3*window_size*self.hidden_size))
        parameters['w_z'] = model.add_parameters((self.hidden_size, 1))
        parameters['v'] = model.add_parameters((1, self.hidden_size))
        parameters['W_cs'] = model.add_parameters(
            (self.hidden_size, 3*window_size*self.hidden_size))
        parameters['w_s'] = model.add_parameters((self.hidden_size, 1))
        if self.concatenate_last_layer:
            parameters['O'] = model.add_parameters((num_tags,
                                                    3*self.hidden_size))
        else:
            parameters['O'] = model.add_parameters((num_tags,
                                                    self.hidden_size))
        self.model = model
        self.parameters = parameters
        self.builders = [dy.LSTMBuilder(1, self.embedding_size,
                                        self.hidden_size, self.model),
                         dy.LSTMBuilder(1, self.embedding_size,
                                        self.hidden_size, self.model)]

    def build_graph(self, instance, num_sketches=-1, noise_level=0.1,
                    training=True):
        unk = self.word_vocabulary.w2i['_UNK_']
        words = [self.word_vocabulary.w2i.get(w, unk) for w, _ in instance]
        tags = [self.tag_vocabulary.w2i[t] for _, t in instance]

        dy.renew_cg()
        f_init, b_init = [b.initial_state() for b in self.builders]

        E = self.parameters['E']
        wembs = [E[w] for w in words]
        wembs = [dy.noise(we, noise_level) for we in wembs]

        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        hidden_states = []
        for f, b in zip(fw, reversed(bw)):
            hidden_states.append(dy.concatenate([f, b]))

        W_cz = dy.parameter(self.parameters['W_cz'])
        w_z = dy.parameter(self.parameters['w_z'])
        v = dy.parameter(self.parameters['v'])
        W_cs = dy.parameter(self.parameters['W_cs'])
        w_s = dy.parameter(self.parameters['w_s'])

        # Initialize all word sketches as zero vectors.
        sketches = []
        for word in words:
            sketch = dy.vecInput(self.hidden_size)
            sketch.set(np.zeros(self.hidden_size))
            sketches.append(sketch)

        # Make several sketch steps.
        if num_sketches < 0:
            num_sketches = len(words)
        for j in xrange(num_sketches):
            z = []
            states = []
            states_with_context = []
            for i in xrange(len(words)):
                state = dy.concatenate([hidden_states[i], sketches[i]])
                states.append(state)
            #pdb.set_trace()
            for i in xrange(len(words)):
                state_with_context = state
                for l in xrange(1, self.context_size+1):
                    if i-l < 0:
                        state = dy.vecInput(3*self.hidden_size)
                        state.set(np.zeros(3*self.hidden_size))
                    else:
                        state = states[i-l]
                    state_with_context = dy.concatenate([state_with_context,
                                                         state])
                for l in xrange(1, self.context_size+1):
                    if i+l >= len(states):
                        state = dy.vecInput(3*self.hidden_size)
                        state.set(np.zeros(3*self.hidden_size))
                    else:
                        state = states[i+l]
                    state_with_context = dy.concatenate([state_with_context,
                                                         state])
                z_i = v * dy.tanh(W_cz * state_with_context + w_z)
                z.append(z_i)
                states_with_context.append(state_with_context)
            temperature = 10. # 1.
            #attention_weights = dy.softmax(dy.concatenate(z)/temperature)
            attention_weights = dy.sparsemax(dy.concatenate(z)/temperature)
            #if not training:
            #    pdb.set_trace()
            cbar = dy.esum([vector*attention_weight
                            for vector, attention_weight in
                            zip(states_with_context, attention_weights)])
            s_n = dy.tanh(W_cs * cbar + w_s)
            sketches = [sketch + s_n * weight
                        for sketch, weight in zip(sketches, attention_weights)]

        # Now use the last sketch to make a prediction.
        O = dy.parameter(self.parameters['O'])
        if training:
            errs = []
            for i, t in enumerate(tags):
                if self.concatenate_last_layer:
                    state = dy.concatenate([hidden_states[i], sketches[i]])
                else:
                    state = sketches[i]
                r_t = O * state
                err = dy.pickneglogsoftmax(r_t, t)
                errs.append(err)
            return dy.esum(errs)
        else:
            predicted_tags=[]
            for i in xrange(len(words)):
                if self.concatenate_last_layer:
                    state = dy.concatenate([hidden_states[i], sketches[i]])
                else:
                    state = sketches[i]
                r_t = O * state
                #out = dy.softmax(r_t)
                chosen = np.argmax(r_t.npvalue())
                predicted_tags.append(self.tag_vocabulary.i2w[chosen])
            return predicted_tags


def read_dataset(fname):
    sent = []
    for line in file(fname):
        line = line.strip().split()
        if not line:
            if sent: yield sent
            sent = []
        else:
            w, t = line
            sent.append((w, t))

def create_vocabularies(corpora, word_cutoff=0):
    word_counter = Counter()
    tag_counter = Counter()
    word_counter['_UNK_'] = word_cutoff+1
    for corpus in corpora:
        for s in corpus:
            for w, t in s:
                word_counter[w] += 1
                tag_counter[t] += 1
    words = [w for w in word_counter if word_counter[w] > word_cutoff]
    tags = [t for t in tag_counter]

    word_vocabulary = util.Vocab.from_corpus([words])
    tag_vocabulary = util.Vocab.from_corpus([tags])

    print >> sys.stderr, 'Words: %d' % word_vocabulary.size()
    print >> sys.stderr, 'Tags: %d' % tag_vocabulary.size()

    return word_vocabulary, tag_vocabulary

def main():
    '''Main function.'''
    # Parse arguments.
    parser = argparse.ArgumentParser(
        prog='Neural Easy-First POS Tagger',
        description='Trains/test a neural easy-first POS tagger.')

    parser.add_argument('-train_file', type=str, default='')
    parser.add_argument('-dev_file', type=str, default='')
    parser.add_argument('-test_file', type=str, default='')
    parser.add_argument('-concatenate_last_layer', type=int, default=1)
    parser.add_argument('-embedding_size', type=int, default=64) # 128
    parser.add_argument('-hidden_size', type=int, default=20) # 50
    parser.add_argument('-context_size', type=int, default=1) # 0
    parser.add_argument('-num_sketches', type=int, default=-1)
    parser.add_argument('-num_epochs', type=int, default=50)

    args = vars(parser.parse_args())
    print >> sys.stderr, args

    train_file = args['train_file']
    dev_file = args['dev_file']
    test_file = args['test_file']
    concatenate_last_layer = args['concatenate_last_layer']
    embedding_size = args['embedding_size']
    hidden_size = args['hidden_size']
    context_size = args['context_size']
    num_sketches = args['num_sketches']
    num_epochs = args['num_epochs']

    # Read corpus (train, dev, test).
    print >> sys.stderr
    print >> sys.stderr, 'Loading train/dev/test datasets...'
    train_instances = list(read_dataset(train_file))
    dev_instances = list(read_dataset(dev_file))
    test_instances = list(read_dataset(test_file))
    word_vocabulary, tag_vocabulary = create_vocabularies([train_instances])

    # Create model.
    tagger = NeuralEasyFirstTagger(word_vocabulary, tag_vocabulary,
                                   embedding_size, hidden_size, context_size,
                                   concatenate_last_layer)
    tagger.create_model()

    # Train.
    print >> sys.stderr
    print >> sys.stderr, 'Training...'
    tic = time.time()
    sgd = dy.SimpleSGDTrainer(tagger.model, e0=0.1)
    for epoch in xrange(num_epochs):
        tagged = loss = 0
        #random.shuffle(train_instances)
        for i, instance in enumerate(train_instances, 1):
            sum_errs = tagger.build_graph(instance, num_sketches=num_sketches,
                                          noise_level=0.1)
            loss += sum_errs.scalar_value()
            tagged += len(instance)
            sum_errs.backward()
            sgd.update()

        # Check accuracy in dev set.
        correct = 0
        total = 0
        for instance in dev_instances:
            gold_tags = [t for _, t in instance]
            predicted_tags = tagger.build_graph(instance,
                                                num_sketches=num_sketches,
                                                noise_level=0.,
                                                training=False)
            correct += sum([int(g == p)
                            for g, p in zip(gold_tags, predicted_tags)])
            total += len(gold_tags)
        dev_accuracy = float(correct) / total
        sgd.status()
        print >> sys.stderr, 'Epoch: %d, Loss: %f, Dev Acc: %f' % (epoch+1,
                                                                   loss/tagged,
                                                                   dev_accuracy)
    toc = time.time()
    print >> sys.stderr, 'Training took %f miliseconds.' % (toc - tic)


if __name__ == "__main__":
    main()

