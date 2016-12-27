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
    def __init__(self, word_vocabulary, tag_vocabulary, model_type,
                 attention_type, temperature, discount_factor, embedding_size,
                 hidden_size, context_size, concatenate_last_layer,
                 use_sketch_losses):
        self.word_vocabulary = word_vocabulary
        self.tag_vocabulary = tag_vocabulary
        self.model_type = model_type
        self.attention_type = attention_type
        self.temperature = temperature
        self.discount_factor = discount_factor
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.concatenate_last_layer = concatenate_last_layer
        self.use_sketch_losses = use_sketch_losses
        self.model = None
        self.parameters = {}
        self.builders = []
        self.track_sketches = False
        self.sketch_file = None

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
        #parameters['w_z'] = model.add_parameters((self.hidden_size, 1))
        parameters['w_z'] = model.parameters_from_numpy(
            np.zeros((self.hidden_size, 1)))
        parameters['v'] = model.add_parameters((1, self.hidden_size))
        #parameters['v'] = model.parameters_from_numpy(
        #    np.zeros((1, self.hidden_size)))
        parameters['W_cs'] = model.add_parameters(
            (self.hidden_size, 3*window_size*self.hidden_size))
        #parameters['w_s'] = model.add_parameters((self.hidden_size, 1))
        parameters['w_s'] = model.parameters_from_numpy(
            np.zeros((self.hidden_size, 1)))
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

    def squared_norm_of_parameters(self):
        squared_norm = dy.scalarInput(0.)
        for key in self.parameters:
            if type(self.parameters[key]) == dy.LookupParameters:
                continue
                #w = self.parameters[key]
            else:
                w = dy.parameter(self.parameters[key])
            squared_norm += dy.trace_of_product(w, w)
        return squared_norm

    def build_graph(self, instance, num_sketches=-1, noise_level=0.1,
                    training=True, epoch=-1):
        unk = self.word_vocabulary.w2i['_UNK_']
        words = [self.word_vocabulary.w2i.get(w, unk) for w, _ in instance]
        tags = [self.tag_vocabulary.w2i[t] for _, t in instance]

        if training:
            errs = []

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
        O = dy.parameter(self.parameters['O'])

        # Initialize all word sketches as zero vectors.
        sketches = []
        for word in words:
            sketch = dy.vecInput(self.hidden_size)
            sketch.set(np.zeros(self.hidden_size))
            sketches.append(sketch)

        # Make several sketch steps.
        if num_sketches < 0:
            num_sketches = len(words)
        cumulative_attention = dy.vecInput(len(words))
        cumulative_attention.set(np.ones(len(words)) / len(words))
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
                #if epoch > -2:
                #    print dy.tanh(W_cz * state_with_context + w_z).npvalue()
                z_i = v * dy.tanh(W_cz * state_with_context + w_z)
                z.append(z_i)
                states_with_context.append(state_with_context)
            temperature = self.temperature #10. # 1.
            discount_factor = self.discount_factor #5. #10. #50. # 0.
            #attention_weights = dy.softmax(dy.concatenate(z)/temperature)
            scores = dy.concatenate(z) - cumulative_attention * discount_factor
            if self.attention_type == 'softmax':
                attention_weights = dy.softmax(scores / temperature)
            elif self.attention_type == 'sparsemax':
                attention_weights = dy.sparsemax(scores / temperature)
            else:
                raise NotImplementedError
            if self.track_sketches:
                self.sketch_file.write('%s\n' % ' '.join(['{:.3f}'.format(p) \
                    for p in attention_weights.npvalue()]))
            cumulative_attention = \
                (attention_weights + cumulative_attention * i) / (i+1)
            #if not training:
            #    pdb.set_trace()
            if self.model_type == 'single_state':
                cbar = dy.esum([vector*attention_weight
                                for vector, attention_weight in
                                zip(states_with_context, attention_weights)])
                s_n = dy.tanh(W_cs * cbar + w_s)
                if training and self.use_sketch_losses:
                    assert not self.concatenate_last_layer
                    state = s_n
                    r_t = O * state
                    for i, t in enumerate(tags):
                        err = dy.pickneglogsoftmax(r_t, t)
                        errs.append(err * attention_weights[i] / float(num_sketches))
                sketches = [sketch + s_n * weight
                            for sketch, weight in \
                            zip(sketches, attention_weights)]
            elif self.model_type == 'all_states':
                for i in xrange(len(words)):
                    s_n = dy.tanh(W_cs * states_with_context[i] + w_s)
                    sketches[i] += s_n * attention_weights[i]
            else:
                raise NotImplementedError

        if self.track_sketches:
            self.sketch_file.write('\n')

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
                errs.append(err)
                chosen = np.argmax(r_t.npvalue())
                predicted_tags.append(self.tag_vocabulary.i2w[chosen])
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

    # Need to be here as an argument to allow specifying a seed.
    parser.add_argument('--dynet-seed', type=str, default=0)

    parser.add_argument('-train_file', type=str, default='')
    parser.add_argument('-dev_file', type=str, default='')
    parser.add_argument('-test_file', type=str, default='')
    parser.add_argument('-concatenate_last_layer', type=int, default=1)
    parser.add_argument('-use_sketch_losses', type=int, default=0)
    parser.add_argument('-embedding_size', type=int, default=64) # 128
    parser.add_argument('-hidden_size', type=int, default=20) # 50
    parser.add_argument('-context_size', type=int, default=1) # 0
    parser.add_argument('-num_sketches', type=int, default=-1)
    parser.add_argument('-num_epochs', type=int, default=50)
    parser.add_argument('-l2_regularization', type=float, default=0.)
    parser.add_argument('-model_type', type=str, default='single_state')
    parser.add_argument('-attention_type', type=str, default='softmax')
    parser.add_argument('-temperature', type=float, default=1.)
    parser.add_argument('-discount_factor', type=float, default=0.)
    parser.add_argument('-sketch_file', type=str, required=True)

    args = vars(parser.parse_args())
    print >> sys.stderr, args

    train_file = args['train_file']
    dev_file = args['dev_file']
    test_file = args['test_file']
    concatenate_last_layer = args['concatenate_last_layer']
    use_sketch_losses = args['use_sketch_losses']
    embedding_size = args['embedding_size']
    hidden_size = args['hidden_size']
    context_size = args['context_size']
    num_sketches = args['num_sketches']
    num_epochs = args['num_epochs']
    l2_regularization = args['l2_regularization']
    model_type = args['model_type']
    attention_type = args['attention_type']
    temperature = args['temperature']
    discount_factor = args['discount_factor']
    sketch_file = args['sketch_file']

#    suffix = 'model-%s_attention-%s_temp-%f_disc-%f_C-%f_sketches-%d_' \
#             'cat-%d_emb-%d_hid-%d_ctx-%d' % \
#             (model_type, attention_type, temperature, discount_factor,
#              l2_regularization, num_sketches, concatenate_last_layer,
#              embedding_size, hidden_size, context_size)

    # Read corpus (train, dev, test).
    print >> sys.stderr
    print >> sys.stderr, 'Loading train/dev/test datasets...'
    train_instances = list(read_dataset(train_file))
    dev_instances = list(read_dataset(dev_file))
    test_instances = list(read_dataset(test_file))
    word_vocabulary, tag_vocabulary = create_vocabularies([train_instances])

    # Create model.
    tagger = NeuralEasyFirstTagger(word_vocabulary, tag_vocabulary, model_type,
                                   attention_type, temperature, discount_factor,
                                   embedding_size, hidden_size, context_size,
                                   concatenate_last_layer, use_sketch_losses)
    tagger.create_model()

    # Train.
    print >> sys.stderr
    print >> sys.stderr, 'Training...'
    tic = time.time()
    #trainer = dy.AdamTrainer(tagger.model, alpha=0.0001)
    trainer = dy.AdagradTrainer(tagger.model, e0=0.1)
    #sgd = dy.SimpleSGDTrainer(tagger.model, e0=0.1)
    best_epoch = -1
    best_dev_accuracy = 0.
    for epoch in xrange(num_epochs):
        tagged = correct = loss = reg = 0
        #random.shuffle(train_instances)
        for i, instance in enumerate(train_instances, 1):
            gold_tags = [t for _, t in instance]
            sum_errs, predicted_tags = \
                tagger.build_graph(instance,
                                   num_sketches=num_sketches,
                                   noise_level=0.) #0.1)
            val = sum_errs.scalar_value()
            loss += val
            sum_errs += tagger.squared_norm_of_parameters() * \
                        l2_regularization
            reg += (sum_errs.scalar_value() - val)
            tagged += len(instance)
            correct += sum([int(g == p)
                            for g, p in zip(gold_tags, predicted_tags)])
            sum_errs.backward()
            trainer.update()
        train_accuracy = float(correct) / tagged

        # Check accuracy in dev set.
        tagger.track_sketches = True
        tagger.sketch_file = open(sketch_file + '.tmp', 'w')
        correct = 0
        total = 0
        for instance in dev_instances:
            gold_tags = [t for _, t in instance]
            predicted_tags = tagger.build_graph(instance,
                                                num_sketches=num_sketches,
                                                noise_level=0.,
                                                training=False,
                                                epoch=epoch)
            correct += sum([int(g == p)
                            for g, p in zip(gold_tags, predicted_tags)])
            total += len(gold_tags)
        dev_accuracy = float(correct) / total
        tagger.sketch_file.close()
        tagger.track_sketches = False

        # Check if this is the best model so far.
        if epoch == 0 or dev_accuracy > best_dev_accuracy:
            best_epoch = epoch
            best_dev_accuracy = dev_accuracy
            from shutil import copyfile
            copyfile(sketch_file + '.tmp', sketch_file)

        # Plot epoch statistics.
        trainer.status()
        print >> sys.stderr, \
            'Epoch: %d, Loss: %f, Reg: %f, Train acc: %f, Dev Acc: %f, ' \
            'Best Dev acc: %f' \
            % (epoch+1,
               loss/tagged,
               reg/tagged,
               train_accuracy,
               dev_accuracy,
               best_dev_accuracy)

    toc = time.time()
    print >> sys.stderr, 'Final Dev Accuracy: %f.' % best_dev_accuracy
    print >> sys.stderr, 'Training took %f miliseconds.' % (toc - tic)


if __name__ == "__main__":
    main()

