import numpy as np
import random
from hmmlearn import hmm
import cPickle as pkl

# generate dummy data for sequence tagging problems

def random_data(numbers_of_instances, instance_length, vocab_size, number_of_labels):
    """
    xs are random integers, ys are unrelated to xs and random
    :param numbers_of_instances:
    :param instance_length:
    :param vocab_size:
    :param number_of_labels:
    :return:
    """
    xs, ys = [], []
    for number_of_instances in numbers_of_instances:
        x = np.floor(np.random.rand(number_of_instances, instance_length)*vocab_size)
        y = np.floor(np.random.rand(number_of_instances, instance_length)*number_of_labels)
        xs.append(x)
        ys.append(np.array(y))
    return xs, ys


def random_deterministic_data(numbers_of_instances, instance_length, vocab_size, number_of_labels):
    """
    xs are random integers, ys are dependent on xs
    :param numbers_of_instances:
    :param instance_length:
    :param vocab_size:
    :param number_of_labels:
    :return:
    """
    xs, ys = [], []
    for number_of_instances in numbers_of_instances:
        x = np.floor(np.random.rand(number_of_instances, instance_length)*vocab_size)
        f = float(number_of_labels)/vocab_size
        y = [np.floor(x_i*f) for x_i in x]
        xs.append(x)
        ys.append(np.array(y))
    return xs, ys

def random_interdependent_data(numbers_of_instances, instance_length, vocab_size, number_of_labels):
    """
    xs are random integers, ys are dependent on xs and on previous xs (bi-gram)
    :param numbers_of_instances:
    :param instance_length:
    :param vocab_size:
    :param number_of_labels:
    :return:
    """
    xs, ys = [], []
    for number_of_instances in numbers_of_instances: # for each dataset
        x = np.floor(np.random.rand(number_of_instances, instance_length)*vocab_size)
        probs = np.random.randn(vocab_size, number_of_labels, number_of_labels)
        y = [[-1]*instance_length for x_i in x]
        for i in range(number_of_instances):
            label = random.randint(0, number_of_labels-1)
            label_prev = label
            y[i][0] = label
            for j in range(1, instance_length):
                x_i_j = int(x[i][j])
                # fixed label transitions
                # alternatively use probabilistic transitions: np.random.choice(range(labels), p=probs)
                label = np.argmax(probs[x_i_j][label_prev], 0)
                y[i][j] = label
                label_prev = label
        xs.append(x)
        ys.append(np.array(y))
    return xs, ys


def random_data_with_len(numbers_of_instances, instance_length, vocab_size, number_of_labels):
    """
    xs are random integers, ys are unrelated to xs and random
    xs has length up to instance_length
    :param numbers_of_instances:
    :param instance_length:
    :param vocab_size:
    :param number_of_labels:
    :return:
    """
    xs, ys = [], []
    for number_of_instances in numbers_of_instances:
        x = []
        y = []
        for i in range(number_of_instances):
            x_i_len = np.random.randint(1, instance_length+1)
            x_i = [np.random.randint(vocab_size) for j in range(x_i_len)]
            x.append(x_i)
            y_i = [np.random.randint(number_of_labels) for j in range(x_i_len)]
            y.append(y_i)
        xs.append(np.array(x))
        ys.append(np.array(y))
    return xs, ys


def random_deterministic_data_with_len(numbers_of_instances, instance_length, vocab_size, number_of_labels):
    """
    xs are random integers, ys are dependent on xs
    :param numbers_of_instances:
    :param instance_length:
    :param vocab_size:
    :param number_of_labels:
    :return:
    """
    xs, ys = [], []
    for number_of_instances in numbers_of_instances:
        x = []
        y = []
        for i in range(number_of_instances):
            x_i_len = np.random.randint(1, instance_length+1)
            x_i = [np.random.randint(vocab_size) for j in range(x_i_len)]
            x.append(x_i)
            f = float(number_of_labels)/vocab_size
            y_i = [np.floor(j*f) for j in x_i]
            y.append(y_i)
        xs.append(np.array(x))
        ys.append(np.array(y))
    return xs, ys

def random_interdependent_data_with_len(numbers_of_instances, instance_length, vocab_size, number_of_labels):
    """
    xs are random integers, ys are dependent on xs and on previous xs (bi-gram)
    :param numbers_of_instances:
    :param instance_length:
    :param vocab_size:
    :param number_of_labels:
    :return:
    """
    xs, ys = [], []
    for number_of_instances in numbers_of_instances: # for each dataset
        probs = np.random.randn(vocab_size, number_of_labels, number_of_labels)
        x = []
        y = []
        for i in range(number_of_instances):
            x_i_len = np.random.randint(1, instance_length+1)
            x_i = [np.random.randint(vocab_size) for j in range(x_i_len)]
            x.append(x_i)
            label = random.randint(0, number_of_labels-1)
            label_prev = label
            y_i = []
            for j in range(0, x_i_len):
                x_i_j = int(x[i][j])
                # fixed label transitions
                # alternatively use probabilistic transitions: np.random.choice(range(labels), p=probs)
                label = np.argmax(probs[x_i_j][label_prev], 0)
                y_ij = label
                label_prev = label
                y_i.append(y_ij)
            y.append(y_i)
        xs.append(np.array(x))
        ys.append(np.array(y))
    return xs, ys


def softmax_row(x, tau=1.):
    """
    softmax by row
    :param x:
    :return:
    """
    softmax_row = []
    for r in xrange(x.shape[0]):
        row = np.exp(x[r]/tau) / np.sum(np.exp(x[r]/tau), axis=0)
        softmax_row.append(row)
    return np.array(softmax_row)


def generate_hmm_data_probs(number_of_labels, vocab_size, dump_prefix):
    """
    Generate the probabilities for an hmm and pickle them
    :param number_of_labels:
    :param vocab_size:
    :param dump_file:
    :return:
    """
    # random start probabilities
    init = np.random.randn(1, number_of_labels)
    init_prob = softmax_row(init, 0.5)[0]
    print "start probs", init_prob

    # fixed transition probabilities
    # 1) equal
    #a_prob = np.ones(shape=(number_of_labels, number_of_labels), dtype=float)/number_of_labels

    # 2) strongly correlated within label class
    #a  = np.ones(shape=(number_of_labels, number_of_labels), dtype=float)/number_of_labels
    #imbalance = 2
    #a[np.diag_indices_from(a)] += imbalance
    #a_prob = softmax_row(a, 1)  # transition probabilities

    # 3) random
    a = np.random.randn(number_of_labels, number_of_labels)
    a_prob = softmax_row(a)  # transition probabilities
    print "transition prob", a_prob


    # random output probs
    o = np.random.randn(number_of_labels, vocab_size)
    o_prob = softmax_row(o, number_of_labels*2)  # output probabilities
    print "output prob", o_prob

    # pickle
    dump_file = "%s.labels%d.vocab%d.pkl" % (dump_prefix, number_of_labels, vocab_size)
    pkl.dump((init_prob, a_prob, o_prob), open(dump_file, "wb"))
    print "dumped to ", dump_file



def generate_hmm_data_from_fixed(numbers_of_instances, instance_length, prob_file):
    init_prob, a_prob, o_prob = pkl.load(prob_file)
    number_of_labels = len(init_prob)
    vocab_size = o_prob.shape[1]
    # create hmm
    hmm_model = hmm.MultinomialHMM(number_of_labels)
    hmm_model.transmat_ = a_prob
    hmm_model.startprob_ = init_prob
    hmm_model.emissionprob_ = o_prob
    xs, ys = [], []
    for number_of_instances in numbers_of_instances:
        x, y = [], []
        for i in range(number_of_instances):
            # generate input
            x_i_len = np.random.randint(1, instance_length+1)
            x_i = np.array([[np.random.randint(vocab_size)] for j in range(x_i_len)])
            y_i = hmm_model.predict(x_i)
            # or sample from model directly: x_i, y_i = hmm_model.sample(x_i_len)
            x.append(x_i.flatten())
            y.append(y_i)
        xs.append(x)
        ys.append(y)
    return xs, ys


def generate_hmm_data_random(numbers_of_instances, instance_length, vocab_size, number_of_labels):
    """
    hmm:
    y: states
    x: observations
    a: state transition probabilities
    o: output probabilities

    WARNING: in order to make this run, some sklearn hacking is required
    (due to internal version problems of sklearn):
    fix imports in the installed version of sklearn cluster, e.g.:
    sudo vi /usr/local/lib/python2.7/dist-packages/sklearn/cluster/bicluster/spectral.py
    -> from sklearn.utils.validation import check_array as check_arrays

    :param numbers_of_instances:
    :param instance_length:
    :param vocab_size:
    :param number_of_labels:
    :return:
    """
    # generate probabilities
    init = np.random.randn(1, number_of_labels)
    init_prob = softmax_row(init)[0]
    print "init prob", init_prob
    a = np.random.randn(number_of_labels, number_of_labels)
    a_prob = softmax_row(a)  # transition probabilities
    print "transition prob", a_prob
    o = np.random.randn(number_of_labels, vocab_size)
    o_prob = softmax_row(o)  # output probabilities
    print "output prob", o_prob
    # create hmm
    hmm_model = hmm.MultinomialHMM(number_of_labels)
    hmm_model.transmat_ = a_prob
    hmm_model.startprob_ = init_prob
    hmm_model.emissionprob_ = o_prob
    xs, ys = [], []
    for number_of_instances in numbers_of_instances:
        x, y = [], []
        for i in range(number_of_instances):
            # generate input
            x_i_len = np.random.randint(1, instance_length+1)
            x_i = np.array([[np.random.randint(vocab_size)] for j in range(x_i_len)])
            y_i = hmm_model.predict(x_i)
            # or sample from model directly: x_i, y_i = hmm_model.sample(x_i_len)
            x.append(x_i.flatten())
            y.append(y_i)
        xs.append(x)
        ys.append(y)
    return xs, ys


def extract_context_features(sequences, start_id, end_id, r=3):
    """
    extract features from window of size r around each target word
    e.g. r=3: one word to the left, actual word, one word to the right
    """
    if r%2 == 0:
        r += 1  # need to be uneven
    print "Extracting context within window of %d" % r
    features = []
    number_context_words = int(np.floor((r-1)/2.))
    for sequence in sequences:
        sequence_features = []
        for i, word in enumerate(sequence):
            context = []
            start = i-number_context_words
            end = i+number_context_words+1
            if start < 0:
                context.append(start_id)
                start = 0
            context.extend(sequence[start:end])
            if end > len(sequence):
                context.append(end_id)
            sequence_features.append(context)
        features.append(sequence_features)
        assert len(sequence) == len(sequence_features)
    return features


if __name__ == "__main__":
    # generate fixed distributions for hmm model
    #generate_hmm_data_probs(2, 20000, "../data/hmm/random")

    # generate fixed corpus from hmm
    dir = "../data/hmm/"
    for f in ["random.labels2.vocab200.pkl", "random.labels2.vocab20000.pkl"]: #["equal.labels2.vocab200.pkl", "equal.labels2.vocab20000.pkl", "imbalanced.labels2.vocab200.pkl", "imbalanced.labels2.vocab20000.pkl", "random.labels2.vocab200.pkl", "random.labels2.vocab20000.pkl"]:
        sets = ["train", "dev", "test"]
        xs, ys = generate_hmm_data_from_fixed([500000, 1000, 2000], 50, open(dir+f, "rb"))  # TODO continue here
        out_f = dir+f.split(".pkl")[0]
        for x, y, s in zip(xs, ys, sets):
            set_file = out_f+"."+s+".pkl"
            pkl.dump((x,y), open(set_file,"wb"))
            print "dumped %s data to %s" % (s, set_file)

    xs, ys = generate_hmm_data_random([5,2], 10, 4, 2)
    print "train: ", xs[0], ys[0]
    print "dev: ", xs[1], ys[1]

    print extract_context_features(xs[0], start_id=11, end_id=12)


    """
    print random_data([3], 4, 10, 3)
    print random_data_with_len([3], 4, 10, 3)
    print random_deterministic_data([3], 4, 10, 3)
    print random_deterministic_data_with_len([3], 4, 10, 3)
    print random_interdependent_data([2], 5, 4, 3)
    print random_interdependent_data_with_len([2], 5, 4, 3)
    """